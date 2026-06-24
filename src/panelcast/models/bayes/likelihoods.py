"""Plug-and-play likelihood-family registry for the score model.

Each family is one self-contained :class:`LikelihoodSpec`:

- ``sample_obs`` — how the family contributes the observation likelihood during
  inference (and generates the ``{prefix}y`` site under ``Predictive``);
- ``predict_draws`` — how it draws cold-start predictive samples for an unseen
  entity (the manual population-distribution path in ``predict.py``);
- ``cdf`` — for location-scale families, the CDF used by the discretization
  toggle to build an interval-censored likelihood. ``None`` for families that
  cannot be discretized (e.g. ``beta``).

The model's likelihood seam (``model._sample_likelihood``) and the new-artist
prediction dispatch (``predict.predict_new_artist``) both resolve a family by
name through :data:`REGISTRY`, so adding a family is a single new entry here
instead of edits scattered across ``model.py`` and ``predict.py``.

Discretization. ``PriorConfig.discretize_observation`` makes the observation
integer-aware via dequantization (:class:`DequantizedDistribution`): inference
conditions the continuous base on ``y + u``, ``u ~ Uniform(-0.5, 0.5)`` a single
fixed jitter, and draws are rounded. This keeps the gradient finite where the
interval-censored ``log(F(k+0.5) - F(k-0.5))`` underflowed (issue #4); that
marginalized form (:class:`RoundedDistribution`) is kept as a dormant fallback.
The continuous (toggle-off) path is byte-identical to the pre-registry code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.stats as jss
import numpyro
import numpyro.distributions as dist
from jax import lax, random
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform

# SinhArcsinhTransform lives in model.py and stays there (the public import path
# tests rely on). model.py imports REGISTRY lazily, so this top-level import does
# not create a cycle: by the time likelihoods.py is imported, model.py is loaded.
from panelcast.models.bayes.model import SinhArcsinhTransform
from panelcast.models.bayes.priors import DEFAULT_BETA_BOUNDARY_EPS, PriorConfig

__all__ = [
    "LikelihoodSpec",
    "DequantizedDistribution",
    "RoundedDistribution",
    "SplitNormal",
    "REGISTRY",
]

# Floor for interval-mass and Beta boundary clips, shared across the module.
_TINY = 1e-12

# "dequantize" (active, finite gradients) or the dormant "interval_cdf" fallback
# (marginalized log-CDF; see docs/LIKELIHOOD_CANDIDATES.md, issue #4).
_DISCRETIZE_MODE = "dequantize"

# Fixed realization, so the jitter is identical on every leapfrog step — redrawing
# per step would break NUTS.
_DEQUANT_JITTER_SEED = 0


# ---------------------------------------------------------------------------
# Custom distributions
# ---------------------------------------------------------------------------
class SplitNormal(dist.Distribution):
    """Two-piece (Fechner) normal: separate left/right scales about ``loc``.

    Density ``f(y) = sqrt(2/pi)/(sL+sR) * exp(-(y-loc)^2 / (2 s^2))`` with
    ``s = sL`` for ``y < loc`` and ``s = sR`` otherwise. Skew comes from
    ``sL != sR`` (``sL > sR`` => longer left tail). ``sL == sR`` recovers
    ``Normal(loc, sL)``.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale_left": constraints.positive,
        "scale_right": constraints.positive,
    }
    support = constraints.real
    reparametrized_params: list[str] = []

    def __init__(self, loc, scale_left, scale_right, *, validate_args=None):
        self.loc = loc
        self.scale_left = scale_left
        self.scale_right = scale_right
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale_left), jnp.shape(scale_right)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        scale = jnp.where(value < self.loc, self.scale_left, self.scale_right)
        log_norm = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(self.scale_left + self.scale_right)
        return log_norm - 0.5 * jnp.square((value - self.loc) / scale)

    def sample(self, key, sample_shape=()):
        k_side, k_half = random.split(key)
        shape = tuple(sample_shape) + self.batch_shape
        total = self.scale_left + self.scale_right
        p_left = self.scale_left / total
        pick_left = random.uniform(k_side, shape) < p_left
        half = jnp.abs(random.normal(k_half, shape))
        return jnp.where(
            pick_left,
            self.loc - self.scale_left * half,
            self.loc + self.scale_right * half,
        )

    def cdf(self, value):
        total = self.scale_left + self.scale_right
        left = (
            2.0 * self.scale_left / total
            * jss.norm.cdf(value, loc=self.loc, scale=self.scale_left)
        )
        right = self.scale_left / total + 2.0 * self.scale_right / total * (
            jss.norm.cdf(value, loc=self.loc, scale=self.scale_right) - 0.5
        )
        return jnp.where(value < self.loc, left, right)


class RoundedDistribution(dist.Distribution):
    """Integer-rounded view of a continuous base distribution.

    ``log_prob(k) = log(F(k+0.5) - F(k-0.5))`` (interval censoring) and samples
    are ``round(base.sample())``. Used with ``obs=`` so a single site is both an
    interval-censored likelihood (inference) and an integer generator (PPC).
    """

    arg_constraints: dict = {}
    support = constraints.real

    def __init__(self, base, cdf_fn, *, validate_args=None):
        self._base = base
        self._cdf_fn = cdf_fn
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        return jnp.round(self._base.sample(key, sample_shape))

    def log_prob(self, value):
        hi = self._cdf_fn(value + 0.5)
        lo = self._cdf_fn(value - 0.5)
        return jnp.log(jnp.maximum(hi - lo, _TINY))


class DequantizedDistribution(dist.Distribution):
    """Dequantized view of a continuous base distribution.

    ``log_prob`` passes through to the continuous base (finite gradient, scored at
    ``y + u``); ``sample`` rounds the base draw so replicated ``y_rep`` stays
    integer. Used with ``obs=y+u`` for inference and ``obs=None`` for generation.
    """

    arg_constraints: dict = {}
    support = constraints.real

    def __init__(self, base, *, validate_args=None):
        self._base = base
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        return jnp.round(self._base.sample(key, sample_shape))

    def log_prob(self, value):
        return self._base.log_prob(value)


def _dequant_jitter(y):
    """Single fixed Uniform(-0.5, 0.5) jitter realization, constant across steps."""
    return random.uniform(
        random.PRNGKey(_DEQUANT_JITTER_SEED), jnp.shape(y), minval=-0.5, maxval=0.5
    )


# ---------------------------------------------------------------------------
# Family CDFs (value -> P(Y <= value)); ``params`` carries sampled globals.
# ---------------------------------------------------------------------------
def _studentt_cdf(value, mu, sigma, df, params):
    # jax.scipy.stats.t has no cdf; use the regularized incomplete beta identity
    # P(T<=t) = 0.5*I_z(df/2, 1/2) for t<=0 (and 1 - that for t>0), z=df/(df+t^2).
    x = (value - mu) / sigma
    z = df / (df + x * x)
    half = 0.5 * jsp.betainc(df / 2.0, 0.5, z)
    return jnp.where(x <= 0.0, half, 1.0 - half)


def _normal_cdf(value, mu, sigma, df, params):
    return jss.norm.cdf(value, loc=mu, scale=sigma)


def _skewnormal_cdf(value, mu, sigma, df, params):
    skew = params["skewness"]
    tw = params["tailweight"]
    z = jnp.sinh(jnp.arcsinh((value - mu) / sigma) / tw - skew)
    return jss.norm.cdf(z)


def _splitnormal_cdf(value, mu, sigma, df, params):
    log_ratio = params["split_log_ratio"]
    scale_left = sigma * jnp.exp(-0.5 * log_ratio)
    scale_right = sigma * jnp.exp(0.5 * log_ratio)
    return SplitNormal(mu, scale_left, scale_right).cdf(value)


def _emit_obs(prefix, base, n_obs, y, *, discretize, cdf_fn):
    """Emit the ``{prefix}y`` site, integer-aware when ``discretize`` is on.

    ``discretize=False`` keeps the original ``sample(f"{prefix}y", base, obs=y)``
    byte-identical. ``discretize=True`` dequantizes (condition on ``y + u``, round
    on generation); the ``interval_cdf`` mode is the dormant marginalized fallback.
    """
    if not discretize:
        d, obs = base, y
    elif _DISCRETIZE_MODE == "interval_cdf":
        d, obs = RoundedDistribution(base, cdf_fn), y
    else:
        d = DequantizedDistribution(base)
        obs = None if y is None else y + _dequant_jitter(y)
    with numpyro.plate(f"{prefix}obs", n_obs):
        numpyro.sample(f"{prefix}y", d, obs=obs)


def _reject_discretization(priors: PriorConfig, family: str) -> None:
    if priors.discretize_observation:
        raise ValueError(
            f"likelihood_family='{family}' does not support "
            "discretize_observation=True (no usable observation CDF). Use a "
            "location-scale family (studentt, normal, skew_normal, split_normal)."
        )


# ---------------------------------------------------------------------------
# sample_obs (inference / generation): one per family
# ---------------------------------------------------------------------------
def _studentt_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    base = dist.StudentT(df, mu, sigma)
    _emit_obs(
        prefix, base, n_obs, y,
        discretize=priors.discretize_observation,
        cdf_fn=lambda v: _studentt_cdf(v, mu, sigma, df, {}),
    )


def _normal_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    base = dist.Normal(mu, sigma)
    _emit_obs(
        prefix, base, n_obs, y,
        discretize=priors.discretize_observation,
        cdf_fn=lambda v: _normal_cdf(v, mu, sigma, df, {}),
    )


def _skew_studentt_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    _reject_discretization(priors, "skew_studentt")
    skewness = numpyro.sample(f"{prefix}skewness", dist.Normal(priors.skew_loc, priors.skew_scale))
    base = dist.TransformedDistribution(
        dist.StudentT(df, jnp.zeros_like(mu), jnp.ones_like(sigma)),
        [SinhArcsinhTransform(skewness, priors.skew_tailweight), AffineTransform(mu, sigma)],
    )
    with numpyro.plate(f"{prefix}obs", n_obs):
        numpyro.sample(f"{prefix}y", base, obs=y)


def _skew_normal_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    skewness = numpyro.sample(f"{prefix}skewness", dist.Normal(priors.skew_loc, priors.skew_scale))
    base = dist.TransformedDistribution(
        dist.Normal(jnp.zeros_like(mu), jnp.ones_like(sigma)),
        [SinhArcsinhTransform(skewness, priors.skew_tailweight), AffineTransform(mu, sigma)],
    )
    params = {"skewness": skewness, "tailweight": priors.skew_tailweight}
    _emit_obs(
        prefix, base, n_obs, y,
        discretize=priors.discretize_observation,
        cdf_fn=lambda v: _skewnormal_cdf(v, mu, sigma, df, params),
    )


def _split_normal_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    log_ratio = numpyro.sample(
        f"{prefix}split_log_ratio",
        dist.Normal(priors.split_scale_ratio_loc, priors.split_scale_ratio_scale),
    )
    scale_left = sigma * jnp.exp(-0.5 * log_ratio)
    scale_right = sigma * jnp.exp(0.5 * log_ratio)
    base = SplitNormal(mu, scale_left, scale_right)
    params = {"split_log_ratio": log_ratio}
    _emit_obs(
        prefix, base, n_obs, y,
        discretize=priors.discretize_observation,
        cdf_fn=lambda v: _splitnormal_cdf(v, mu, sigma, df, params),
    )


def _beta_sample_obs(prefix, mu, sigma, y, n_obs, df, priors, bounds, n_reviews):
    _reject_discretization(priors, "beta")
    if priors.target_transform not in ("identity", None):
        raise ValueError(
            "likelihood_family='beta' requires target_transform='identity' "
            f"(got '{priors.target_transform}'); the Beta likelihood assumes "
            "mu is on the score scale."
        )
    low, high = float(bounds[0]), float(bounds[1])
    span = high - low
    eps = priors.beta_boundary_eps
    mu01 = jnp.clip((mu - low) / span, eps, 1.0 - eps)
    phi = numpyro.sample(
        f"{prefix}phi",
        dist.Gamma(priors.beta_precision_concentration, priors.beta_precision_rate),
    )
    a = mu01 * phi
    b = (1.0 - mu01) * phi
    beta = dist.TransformedDistribution(dist.Beta(a, b), AffineTransform(low, span))
    y_obs = None if y is None else jnp.clip(y, low + span * eps, high - span * eps)
    with numpyro.plate(f"{prefix}obs", n_obs):
        numpyro.sample(f"{prefix}y", beta, obs=y_obs)


# ---------------------------------------------------------------------------
# predict_draws (cold-start population path): one per family
# ---------------------------------------------------------------------------
def _symmetric_draws(key, mu_pred, sigma_scaled, *, df, transform, skew, tailweight, student_base):
    """Shared location-scale draw: Student-t base for df<100 (legacy), else Normal.

    ``skew`` (or None) applies the sinh-arcsinh tilt; ``student_base`` selects the
    df-dependent base used by studentt/normal/skew_studentt (preserves the
    pre-registry behavior verbatim, including normal's Student-t predictive base).
    """
    if student_base and df < 100:
        base_noise = random.t(key, df, mu_pred.shape)
    else:
        base_noise = random.normal(key, mu_pred.shape)
    if skew is not None:
        base_noise = jnp.sinh((jnp.arcsinh(base_noise) + skew[:, None]) * tailweight)
    return transform.inverse(mu_pred + sigma_scaled * base_noise)


def _studentt_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    y = _symmetric_draws(
        key, mu_pred, sigma_scaled, df=df, transform=transform,
        skew=None, tailweight=skew_tailweight, student_base=True,
    )
    return jnp.round(y) if discretize else y


def _normal_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    # student_base=True preserves the pre-registry predictive base for "normal".
    y = _symmetric_draws(
        key, mu_pred, sigma_scaled, df=df, transform=transform,
        skew=None, tailweight=skew_tailweight, student_base=True,
    )
    return jnp.round(y) if discretize else y


def _skew_studentt_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    skew = sites.get("skewness")
    if skew is None:
        raise ValueError(
            "skew_studentt likelihood requires '{prefix}skewness' in "
            "posterior_samples; train with --likelihood-family skew_studentt."
        )
    return _symmetric_draws(
        key, mu_pred, sigma_scaled, df=df, transform=transform,
        skew=skew, tailweight=skew_tailweight, student_base=True,
    )


def _skew_normal_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    skew = sites.get("skewness")
    if skew is None:
        raise ValueError(
            "skew_normal likelihood requires '{prefix}skewness' in "
            "posterior_samples; train with --likelihood-family skew_normal."
        )
    y = _symmetric_draws(
        key, mu_pred, sigma_scaled, df=df, transform=transform,
        skew=skew, tailweight=skew_tailweight, student_base=False,
    )
    return jnp.round(y) if discretize else y


def _split_normal_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    log_ratio = sites.get("split_log_ratio")
    if log_ratio is None:
        raise ValueError(
            "split_normal likelihood requires '{prefix}split_log_ratio' in "
            "posterior_samples; train with --likelihood-family split_normal."
        )
    scale_left = sigma_scaled * jnp.exp(-0.5 * log_ratio[:, None])
    scale_right = sigma_scaled * jnp.exp(0.5 * log_ratio[:, None])
    y = transform.inverse(SplitNormal(mu_pred, scale_left, scale_right).sample(key))
    return jnp.round(y) if discretize else y


def _beta_predict_draws(
    key, mu_pred, sigma_scaled, *, sites, df, bounds, skew_tailweight, transform, discretize
):
    phi = sites.get("phi")
    if phi is None:
        raise ValueError(
            "beta likelihood requires '{prefix}phi' in posterior_samples; "
            "the model must have been trained with --likelihood-family beta."
        )
    low, high = float(bounds[0]), float(bounds[1])
    span = high - low
    # Mirror the inference-side default clip (PriorConfig.beta_boundary_eps); the
    # predict path has no priors object, so it tracks the shared default constant.
    eps = DEFAULT_BETA_BOUNDARY_EPS
    mu01 = jnp.clip((mu_pred - low) / span, eps, 1.0 - eps)
    phi = phi[:, None]
    beta01 = random.beta(key, mu01 * phi, (1.0 - mu01) * phi)
    return low + span * beta01


# ---------------------------------------------------------------------------
# Spec + registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LikelihoodSpec:
    """Everything the model + prediction paths need to use one likelihood family."""

    name: str
    required_sites: tuple[str, ...]
    supports_discretization: bool
    sample_obs: Callable
    predict_draws: Callable
    cdf: Callable | None = None


REGISTRY: dict[str, LikelihoodSpec] = {
    "studentt": LikelihoodSpec(
        name="studentt",
        required_sites=(),
        supports_discretization=True,
        sample_obs=_studentt_sample_obs,
        predict_draws=_studentt_predict_draws,
        cdf=_studentt_cdf,
    ),
    "normal": LikelihoodSpec(
        name="normal",
        required_sites=(),
        supports_discretization=True,
        sample_obs=_normal_sample_obs,
        predict_draws=_normal_predict_draws,
        cdf=_normal_cdf,
    ),
    "skew_studentt": LikelihoodSpec(
        name="skew_studentt",
        required_sites=("skewness",),
        supports_discretization=False,
        sample_obs=_skew_studentt_sample_obs,
        predict_draws=_skew_studentt_predict_draws,
        cdf=None,
    ),
    "beta": LikelihoodSpec(
        name="beta",
        required_sites=("phi",),
        supports_discretization=False,
        sample_obs=_beta_sample_obs,
        predict_draws=_beta_predict_draws,
        cdf=None,
    ),
    "skew_normal": LikelihoodSpec(
        name="skew_normal",
        required_sites=("skewness",),
        supports_discretization=True,
        sample_obs=_skew_normal_sample_obs,
        predict_draws=_skew_normal_predict_draws,
        cdf=_skewnormal_cdf,
    ),
    "split_normal": LikelihoodSpec(
        name="split_normal",
        required_sites=("split_log_ratio",),
        supports_discretization=True,
        sample_obs=_split_normal_sample_obs,
        predict_draws=_split_normal_predict_draws,
        cdf=_splitnormal_cdf,
    ),
}
