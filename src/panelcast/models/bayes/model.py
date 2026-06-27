"""Bayesian hierarchical model for album score prediction.

This module defines the core NumPyro models with:
- Hierarchical artist effects with partial pooling
- Time-varying artist effects via random walk (career trajectory modeling)
- AR(1) structure for album-to-album score dependencies
- Non-centered parameterization to avoid funnel geometry
- Fixed effects for covariates (genre PCA, release year, etc.)
- Student-t likelihood for robustness to extreme scores
- Soft clipping to bound predictions within [0, 100]
- Factory pattern for user_score and critic_score model variants

The extended model structure:
    y_ij ~ StudentT(df, mu_ij, sigma_obs)   [or Normal when df >= 100]
    mu_ij = soft_clip(artist_effect_jt + X_ij @ beta + rho * prev_score_ij)

    # Time-varying artist effect via random walk
    artist_effect_j1 ~ Normal(mu_artist, sigma_artist)  # initial effect
    artist_effect_jt = artist_effect_j(t-1) + N(0, sigma_rw)  # random walk

    # Hyperpriors
    mu_artist ~ Normal(mu_artist_loc, mu_artist_scale)
    sigma_artist ~ HalfNormal(sigma_artist_scale)
    sigma_rw ~ HalfNormal(sigma_rw_scale)
    rho ~ TruncatedNormal(rho_loc, rho_scale, -0.99, 0.99)

    beta ~ Normal(beta_loc, beta_scale)  # fixed effects
    sigma_obs ~ HalfNormal(sigma_obs_scale)  # observation noise

Non-centered parameterization is applied via LocScaleReparam to transform
the init_artist_effect sampling for efficient NUTS sampling.

Model variants:
- user_score_model: For user score prediction (prefix: "user_")
- critic_score_model: For critic score prediction (prefix: "critic_")
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from panelcast.models.bayes.model_math import _CLIP_SHARPNESS, soft_clip  # noqa: F401
from panelcast.models.bayes.priors import PriorConfig, get_default_priors
from panelcast.models.bayes.transforms import get_transform

__all__ = [
    "compute_sigma_scaled",
    "soft_clip",
    "make_score_model",
    "user_score_model",
    "critic_score_model",
    "album_score_model",
]


# Guard for log-space arithmetic: prevents -inf when sigma_obs is near-zero
# during NUTS warmup exploration.  The specific value is irrelevant because
# min_sigma (inside compute_sigma_scaled) is the real floor — this just stops
# -inf from propagating through JAX's autodiff before reaching the clamp.
_LOG_EPS = 1e-8


def compute_sigma_scaled(
    sigma_obs: float,
    n_reviews: jnp.ndarray,
    exponent: float,
    single_review_multiplier: float = 2.0,
    min_sigma: float = 0.01,
) -> jnp.ndarray:
    """Compute per-observation sigma scaled by review count.

    Implements heteroscedastic observation noise where albums with more reviews
    have lower noise (more reliable scores). Uses log-space arithmetic to
    avoid numerical underflow/overflow with extreme review counts.

    The scaling formula is:
        sigma_scaled = sigma_obs / n_reviews^exponent

    Special cases:
        - n_reviews=1: Applies multiplier (default 2x) for unreliable single reviews
        - exponent=0: Returns sigma_obs unchanged (homoscedastic mode)
        - Large n_reviews: Floored at min_sigma to prevent numerical issues

    Args:
        sigma_obs: Base observation noise scale (scalar).
        n_reviews: Array of review counts per observation. Values < 1 are
            clamped to 1.0 defensively.
        exponent: Power-law exponent for scaling. Typically in [0.3, 0.7].
            exponent=0 gives homoscedastic (constant) noise.
            exponent=0.5 gives square-root scaling.
        single_review_multiplier: Multiplier applied to sigma_obs when
            n_reviews=1. Default 2.0 reflects that single reviews are
            unreliable indicators of true album quality.
        min_sigma: Minimum sigma floor for numerical stability. Default 0.01
            prevents underflow with very large review counts.

    Returns:
        Array of scaled sigma values, same shape as n_reviews.

    Notes:
        Log-space arithmetic is used to compute sigma_obs / n^exponent as:
            exp(log(sigma_obs) - exponent * log(n))
        This avoids overflow/underflow for extreme n values (e.g., n=100,000).

    Example:
        >>> import jax.numpy as jnp
        >>> sigma = compute_sigma_scaled(1.0, jnp.array([100.0]), 0.5)
        >>> print(f"{sigma[0]:.2f}")  # ~0.10 (1.0 / sqrt(100))
        0.10
        >>> sigma = compute_sigma_scaled(1.0, jnp.array([1.0]), 0.5)
        >>> print(f"{sigma[0]:.2f}")  # 2.0 (single review penalty)
        2.00
    """
    # Clamp n_reviews to minimum of 1.0 (defensive against invalid data)
    n_safe = jnp.maximum(n_reviews, 1.0)

    # Log-space computation: sigma_obs / n^exponent = exp(log(sigma_obs) - exponent * log(n))
    log_sigma = jnp.log(jnp.maximum(sigma_obs, _LOG_EPS)) - exponent * jnp.log(n_safe)
    sigma_scaled = jnp.exp(log_sigma)

    # Apply single-review penalty (n=1 is unreliable)
    # Use robust comparison instead of exact float equality
    is_single_review = jnp.isclose(n_safe, 1.0, rtol=1e-6, atol=1e-6)
    # DESIGN: Only apply single-review penalty in heteroscedastic mode (exponent > 0).
    # In homoscedastic mode (exponent=0), per-observation noise scaling doesn't
    # exist — all observations share the same sigma_obs.  Adding a single-review
    # penalty would make the model internally inconsistent (claiming homoscedastic
    # while varying sigma).  To penalize single reviews, use heteroscedastic mode.
    # TODO(model-v2): single-review handling is asymmetric across the model.
    # Response side: downweight n=1 albums via observation weights instead of
    # noise scaling (above). Predictor side (worse): the AR(1) term regresses on
    # the *observed* lagged score as if noise-free (`ar_term`, below) while that
    # same score is the review-count-noisy response here, so conditioning on the
    # noisy regressor attenuates rho. A latent-state AR is the principled fix.
    # Tracked under issue #14 (model-v2).
    apply_penalty = jnp.logical_and(is_single_review, exponent > 0)
    sigma_scaled = jnp.where(apply_penalty, sigma_obs * single_review_multiplier, sigma_scaled)

    # Apply minimum floor for numerical stability
    sigma_scaled = jnp.maximum(sigma_scaled, min_sigma)

    return sigma_scaled


def _sample_sigma_rw(prefix: str, priors: PriorConfig) -> jnp.ndarray:
    """Sample the random-walk innovation scale under the configured prior."""
    if priors.sigma_rw_prior_type == "lognormal":
        return numpyro.sample(
            f"{prefix}sigma_rw",
            dist.LogNormal(priors.sigma_rw_lognormal_loc, priors.sigma_rw_lognormal_sigma),
        )
    if priors.sigma_rw_prior_type == "halfnormal":
        return numpyro.sample(
            f"{prefix}sigma_rw",
            dist.HalfNormal(priors.sigma_rw_scale),
        )
    raise ValueError(
        f"Invalid sigma_rw_prior_type: '{priors.sigma_rw_prior_type}'. "
        f"Must be 'lognormal' or 'halfnormal'."
    )


def _sample_sigma_artist(prefix: str, priors: PriorConfig) -> jnp.ndarray:
    """Sample the between-artist scale under the configured prior.

    Mirrors the sigma_rw pattern: "halfnormal" is the legacy default;
    "lognormal" removes the zero-boundary pile-up that throttles NUTS
    mixing for sigma_artist (published ESS 561 < 800 gate).
    """
    if priors.sigma_artist_prior_type == "lognormal":
        return numpyro.sample(
            f"{prefix}sigma_artist",
            dist.LogNormal(
                priors.sigma_artist_lognormal_loc,
                priors.sigma_artist_lognormal_sigma,
            ),
        )
    if priors.sigma_artist_prior_type == "halfnormal":
        return numpyro.sample(
            f"{prefix}sigma_artist",
            dist.HalfNormal(priors.sigma_artist_scale),
        )
    raise ValueError(
        f"Invalid sigma_artist_prior_type: '{priors.sigma_artist_prior_type}'. "
        f"Must be 'lognormal' or 'halfnormal'."
    )


def _sample_sigma_obs(prefix: str, priors: PriorConfig) -> jnp.ndarray:
    """Sample the base observation-noise scale under the configured prior.

    Mirrors the ``_sample_sigma_artist`` / ``_sample_sigma_rw`` pattern:
    "halfnormal" is the legacy default; "lognormal" removes the zero-boundary
    pile-up that lets NUTS collapse sigma_obs onto 0 on heavily zero-inflated
    targets (the econ log-citation failure, sci_sigma_obs -> 0.004). The site
    name ``{prefix}sigma_obs`` is identical under both branches, so every
    downstream consumer (evaluation, prediction, the saved ``.nc``) is
    untouched, and the "halfnormal" branch is the exact legacy sample call,
    preserving the gate-off RNG draw.
    """
    if priors.sigma_obs_prior_type == "lognormal":
        return numpyro.sample(
            f"{prefix}sigma_obs",
            dist.LogNormal(
                priors.sigma_obs_lognormal_loc,
                priors.sigma_obs_lognormal_sigma,
            ),
        )
    if priors.sigma_obs_prior_type == "halfnormal":
        return numpyro.sample(
            f"{prefix}sigma_obs",
            dist.HalfNormal(priors.sigma_obs_scale),
        )
    raise ValueError(
        f"Invalid sigma_obs_prior_type: '{priors.sigma_obs_prior_type}'. "
        f"Must be 'lognormal' or 'halfnormal'."
    )


def _apply_entity_overdispersion(
    prefix: str,
    sigma_scaled: jnp.ndarray,
    artist_idx: jnp.ndarray,
    n_artists: int,
    priors: PriorConfig,
) -> jnp.ndarray:
    """Inflate per-observation noise by a learned per-entity factor (gated).

    Active only when ``priors.heteroscedastic_entity_obs`` is True. Adds a
    non-centered multiplicative log-normal noise term on top of the existing
    ``sigma_scaled``:

        entity_log_scale_e = tau_entity * entity_obs_raw_e
        sigma_scaled_i    *= exp(entity_log_scale[artist_idx_i])

    with ``entity_obs_raw ~ Normal(0, 1)`` (plate over n_artists) and
    ``tau_entity ~ HalfNormal(tau_entity_scale)``. The per-entity
    ``{prefix}entity_log_scale`` is exported as a deterministic (the
    interpretable quantity); the high-cardinality unit-normal
    ``{prefix}entity_obs_raw`` is wired into the train stage's
    exclude_from_collection/idata machinery exactly like ``rw_raw``.

    Contract (the load-bearing safety property): the two NEW sample sites
    ``{prefix}tau_entity`` and ``{prefix}entity_obs_raw`` are created here and
    ONLY here, after every other model site has been sampled. Because NumPyro's
    ``seed`` handler splits PRNG keys in site-execution order, appending sites
    at the end leaves every earlier site's draw bit-identical to the gate-off
    path. The caller must therefore invoke this immediately before the
    likelihood and never on the gate-off branch.
    """
    tau_entity = numpyro.sample(
        f"{prefix}tau_entity",
        dist.HalfNormal(priors.tau_entity_scale),
    )
    entity_obs_raw = numpyro.sample(
        f"{prefix}entity_obs_raw",
        dist.Normal(0.0, 1.0).expand([n_artists]).to_event(1),
    )
    # Export the interpretable per-entity log-inflation; keep the raw site
    # (excluded downstream) only as the non-centered reparameterization.
    entity_log_scale = numpyro.deterministic(
        f"{prefix}entity_log_scale", tau_entity * entity_obs_raw
    )
    return sigma_scaled * jnp.exp(entity_log_scale[artist_idx])


def _sample_init_artist_effect(
    prefix: str,
    n_artists: int,
    mu_artist: jnp.ndarray,
    sigma_artist: jnp.ndarray,
    priors: PriorConfig,
) -> jnp.ndarray:
    """Sample the initial artist effects under the configured parameterization.

    Parameterizations:
        "noncentered": legacy Normal(mu_artist, sigma_artist) plate; the
            factory applies LocScaleReparam to this site.
        "zerosum": effects = mu_artist + sigma_artist * z with
            z ~ ZeroSumNormal — the deviations sum to zero by construction,
            removing the mu_artist <-> effects location ridge (the exported
            "{prefix}init_artist_effect" becomes a deterministic site).
    """
    if priors.artist_effect_param == "zerosum":
        z = numpyro.sample(
            f"{prefix}artist_effect_z",
            dist.ZeroSumNormal(1.0, event_shape=(n_artists,)),
        )
        return numpyro.deterministic(f"{prefix}init_artist_effect", mu_artist + sigma_artist * z)
    if priors.artist_effect_param == "noncentered":
        with numpyro.plate(f"{prefix}artists", n_artists):
            return numpyro.sample(
                f"{prefix}init_artist_effect",
                dist.Normal(mu_artist, sigma_artist),
            )
    raise ValueError(
        f"Invalid artist_effect_param: '{priors.artist_effect_param}'. "
        f"Must be 'noncentered' or 'zerosum'."
    )


def _build_latent_effects(
    prefix: str,
    n_artists: int,
    max_seq: int,
    init_artist_effect: jnp.ndarray,
    sigma_rw: jnp.ndarray,
    priors: PriorConfig,
) -> jnp.ndarray:
    """Build the time-varying artist-effect matrix.

    Contract: returns artist effects with shape ``(max_seq, n_artists)``;
    row t is the effect of every artist at album sequence t+1. Any latent
    process registered here must honor that shape so evaluation/prediction
    stay untouched.

    Processes:
        "rw": non-centered Gaussian random walk (legacy default) —
            cumulative sum of sigma_rw-scaled unit-normal innovations on
            top of the initial effect.
        "ar1": stationary deviations around the initial effect —
            dev_t = phi * dev_{t-1} + sigma_rw * eps_t with dev_1 = 0 and
            |phi| < 1 (new sample site ``{prefix}phi``). Nests the random
            walk exactly at phi = 1, and the sequence-1 effect equals
            init_artist_effect under both processes, so LOO comparisons
            are apples-to-apples. Unlike the random walk, deviation
            variance is bounded by sigma_rw^2 / (1 - phi^2).
    """
    if priors.latent_process not in ("rw", "ar1"):
        raise ValueError(
            f"Unknown latent_process: '{priors.latent_process}'. Registered: ['rw', 'ar1']."
        )

    # Only one time step: no trajectory needed under either process.
    if max_seq <= 1:
        return init_artist_effect[None, :]

    # Non-centered parameterization for trajectory innovations: sample unit
    # normal, then scale by sigma_rw to decouple geometry (avoids Neal's
    # funnel between sigma_rw and the innovations). The site name stays
    # "{prefix}rw_raw" for BOTH processes so the train stage's idata memory
    # exclusion (exclude_from_idata) applies regardless of latent_process.
    rw_raw = numpyro.sample(
        f"{prefix}rw_raw",
        dist.Normal(0, 1).expand([n_artists, max_seq - 1]).to_event(2),
    )
    innovations = sigma_rw * rw_raw  # (n_artists, max_seq - 1)

    if priors.latent_process == "ar1":
        phi = numpyro.sample(
            f"{prefix}phi",
            dist.TruncatedNormal(
                loc=priors.phi_loc,
                scale=priors.phi_scale,
                low=-0.99,
                high=0.99,
            ),
        )

        def _ar1_step(dev: jnp.ndarray, eps_t: jnp.ndarray):
            dev_new = phi * dev + eps_t
            return dev_new, dev_new

        # Scan over time: innovations.T is (max_seq-1, n_artists).
        _, deviations = lax.scan(_ar1_step, jnp.zeros(n_artists), innovations.T)
        trajectory = deviations  # (max_seq - 1, n_artists)
    else:
        # Random walk: cumulative sum of innovations (phi = 1 limit).
        trajectory = jnp.cumsum(innovations, axis=1).T  # (max_seq - 1, n_artists)

    # Full artist effects, shape (max_seq, n_artists): row 0 is the initial
    # effect; later rows add the latent trajectory.
    return jnp.vstack(
        [
            init_artist_effect[None, :],
            init_artist_effect[None, :] + trajectory,
        ]
    )


def _apply_target_transform(
    mu_raw: jnp.ndarray,
    priors: PriorConfig,
    target_bounds: tuple[float, float],
) -> jnp.ndarray:
    """Map the raw linear predictor to the likelihood location parameter."""
    transform = get_transform(
        priors.target_transform,
        target_bounds=target_bounds,
        offset=priors.logit_offset,
    )
    return transform.transform_mu(mu_raw)


def _log_cosh(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable ``log(cosh(x))``."""
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)


class SinhArcsinhTransform(Transform):
    """Jones-Pewsey sinh-arcsinh transform of a base real variable.

    ``y = sinh((arcsinh(x) + skewness) * tailweight)``. With ``skewness = 0`` and
    ``tailweight = 1`` this is the identity, so applying it to a StudentT base
    and then locating/scaling nests the symmetric Student-t exactly. A nonzero
    skewness tilts the density (negative -> longer left tail), giving the
    skew-t needed for the left-skewed score distribution.
    """

    domain = constraints.real
    codomain = constraints.real

    def __init__(self, skewness, tailweight=1.0):
        self.skewness = skewness
        self.tailweight = tailweight
        super().__init__()

    def __call__(self, x):
        return jnp.sinh((jnp.arcsinh(x) + self.skewness) * self.tailweight)

    def _inverse(self, y):
        return jnp.sinh(jnp.arcsinh(y) / self.tailweight - self.skewness)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        inner = (jnp.arcsinh(x) + self.skewness) * self.tailweight
        return jnp.log(jnp.abs(self.tailweight)) + _log_cosh(inner) - 0.5 * jnp.log1p(x * x)

    def tree_flatten(self):
        return (self.skewness, self.tailweight), (("skewness", "tailweight"), {})

    def __eq__(self, other):
        return (
            isinstance(other, SinhArcsinhTransform)
            and self.tailweight == other.tailweight
            and jnp.array_equal(self.skewness, other.skewness)
        )


def _sample_likelihood(
    prefix: str,
    mu: jnp.ndarray,
    sigma_scaled: jnp.ndarray,
    y: jnp.ndarray | None,
    n_obs: int,
    likelihood_df: float,
    priors: PriorConfig,
    target_bounds: tuple[float, float] = (0.0, 100.0),
    n_reviews: jnp.ndarray | None = None,
) -> None:
    """Dispatch the observation likelihood to the configured family's spec.

    The family math lives in :mod:`panelcast.models.bayes.likelihoods` — one
    self-contained ``LikelihoodSpec`` per family — so this seam only resolves the
    family by name and forwards. ``{prefix}y`` is on the score scale under every
    family, so evaluation, prediction, and the saved idata are untouched by the
    choice. Student-t with ``df >= 100`` degrades to Normal (legacy rule).
    """
    # Lazy import breaks the cycle: likelihoods.py imports SinhArcsinhTransform
    # from this module, so this module must not import it at load time.
    from panelcast.models.bayes.likelihoods import REGISTRY

    family = priors.likelihood_family
    if family == "studentt" and likelihood_df >= 100:
        family = "normal"
    try:
        spec = REGISTRY[family]
    except KeyError:
        raise ValueError(
            f"Unknown likelihood_family: '{priors.likelihood_family}'. "
            f"Registered: {sorted(REGISTRY)}."
        ) from None
    spec.sample_obs(
        prefix,
        mu,
        sigma_scaled,
        y,
        n_obs,
        likelihood_df,
        priors,
        target_bounds,
        n_reviews,
    )


def make_score_model(score_type: str) -> Callable:
    """Factory function to create score prediction models.

    Creates a NumPyro model function with score-type-specific parameter prefixes.
    This allows fitting separate models for user scores and critic scores with
    distinct posterior distributions.

    Parameters
    ----------
    score_type : str
        Either "user" or "critic" to create score-specific models.
        The score_type becomes a prefix for all sample site names
        (e.g., "user_beta", "critic_rho").

    Returns
    -------
    Callable
        NumPyro model function with non-centered parameterization.
        The returned function has signature: model(artist_idx, album_seq,
            prev_score, X, y=None, n_artists=None, max_seq=None, priors=None)

    Example
    -------
    >>> user_model = make_score_model("user")
    >>> critic_model = make_score_model("critic")
    >>>
    >>> # User model samples will have prefixes like "user_beta", "user_rho"
    >>> # Critic model samples will have prefixes like "critic_beta", "critic_rho"
    """
    # Any identifier works as a posterior-site prefix; the descriptor supplies
    # the domain's prefix (AOTY: "user"/"critic", aero example: "perf").
    if not isinstance(score_type, str) or not score_type.isidentifier():
        raise ValueError(
            f"score_type must be a non-empty identifier usable as a posterior-site "
            f"prefix (e.g. 'user', 'critic', 'perf'), got {score_type!r}"
        )

    prefix = f"{score_type}_"

    def _score_model_centered(
        artist_idx: jnp.ndarray,
        album_seq: jnp.ndarray,
        prev_score: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray | None = None,
        n_artists: int | None = None,
        max_seq: int | None = None,
        priors: PriorConfig | None = None,
        n_reviews: jnp.ndarray | None = None,
        n_ref: float | None = None,
        n_exponent: float = 0.0,
        learn_n_exponent: bool = False,
        n_exponent_prior: str = "logit-normal",
        likelihood_df: float = 4.0,
        target_bounds: tuple[float, float] = (0.0, 100.0),
        ar_center: float | jnp.ndarray = 0.0,
        prev_meas_sigma: jnp.ndarray | None = None,
    ) -> None:
        """Centered parameterization of score model (internal).

        This function defines the model in centered form for the initial
        artist effect. Use the reparameterized version returned by
        make_score_model for actual inference.

        Args:
            artist_idx: Integer array of shape (n_obs,) mapping each observation
                to an artist index in [0, n_artists).
            album_seq: Integer array of shape (n_obs,) with album sequence numbers
                for each observation (1 = first album, 2 = second, etc.).
                Used for time-varying artist effects.
            prev_score: Float array of shape (n_obs,) with previous album scores.
                Debut albums (no previous release) are filled with the
                training global mean score by the data pipeline, not 0.
            X: Feature matrix of shape (n_obs, n_features) containing covariates
                such as genre PCA components, release year, etc.
            y: Optional target scores of shape (n_obs,). Pass None for prior
                predictive sampling or posterior predictive on new data.
            n_artists: Number of unique artists. Must be provided.
            max_seq: Maximum album sequence number in the data. Must be provided
                for JAX tracing. Compute as int(album_seq.max()) before calling.
            priors: Prior configuration. If None, uses get_default_priors().
            n_reviews: Optional array of shape (n_obs,) with per-observation
                review counts. Used for heteroscedastic noise scaling. If None,
                homoscedastic noise is used (scalar sigma_obs for all observations).
            n_ref: Reference review count for sigma-ref reparameterization.
                If provided (non-None) and heteroscedastic mode is active, the
                model samples sigma_ref instead of sigma_obs and derives
                sigma_obs = sigma_ref * n_ref^n_exponent as a deterministic
                site. Computed as median(n_reviews) from training data. Must
                be > 1 when provided.
            n_exponent: Fixed exponent for heteroscedastic noise scaling.
                Default 0.0 gives homoscedastic (constant) noise. Higher values
                give more noise reduction for albums with many reviews.
            learn_n_exponent: If True, sample the exponent from a prior distribution
                instead of using the fixed n_exponent value.
            n_exponent_prior: Prior type for learned n_exponent. Options:
                - "logit-normal" (default): Uses TransformedDistribution with
                  Normal(loc, scale) and SigmoidTransform. Avoids funnel geometry.
                - "beta" (legacy): Uses Beta(alpha, beta) distribution. May cause
                  divergences due to funnel geometry in the likelihood.
            ar_center: Value subtracted from prev_score before the AR(1)
                term: ar_term = rho * (prev_score - ar_center). Scalar for
                global centering, per-observation array for running-mean
                centering, 0.0 (default) for the legacy uncentered form.
                Must live on the same scale as prev_score (i.e. the model's
                training scale under a target transform). When the debut
                prev_score fill equals ar_center, debut AR terms are
                exactly zero and rho decorrelates from mu_artist.
            prev_meas_sigma: Optional per-observation measurement-error scale
                for the lagged score, on the same scale as prev_score. Required
                when priors.errors_in_variables is True; ignored otherwise.
                Debuts must be pinned to 0 so their de-noised regressor equals
                the prev_score fill (keeping debut AR terms exactly zero).

        Model structure:
            - Population-level hyperpriors for artist effect distribution
            - Time-varying artist effects via random walk
            - AR(1) term for album-to-album dependency
            - Fixed effects for covariates
            - Observation-level noise (optionally heteroscedastic)
        """
        # Get prior configuration
        if priors is None:
            priors = get_default_priors()

        if n_artists is None:
            raise ValueError("n_artists must be provided")

        if max_seq is None:
            raise ValueError(
                "max_seq must be provided (compute as int(album_seq.max()) before calling)"
            )

        n_features = X.shape[1]

        # === Population-level hyperpriors ===
        # Mean of artist quality distribution
        mu_artist = numpyro.sample(
            f"{prefix}mu_artist",
            dist.Normal(priors.mu_artist_loc, priors.mu_artist_scale),
        )

        # Between-artist standard deviation (controls pooling strength)
        sigma_artist = _sample_sigma_artist(prefix, priors)

        # Random walk innovation scale (controls career trajectory smoothness)
        sigma_rw = _sample_sigma_rw(prefix, priors)

        # AR(1) coefficient for album-to-album dependency
        # Truncated to ensure stationarity
        rho = numpyro.sample(
            f"{prefix}rho",
            dist.TruncatedNormal(
                loc=priors.rho_loc,
                scale=priors.rho_scale,
                low=-0.99,
                high=0.99,
            ),
        )

        # === Initial artist effects (partial pooling; parameterization seam) ===
        init_artist_effect = _sample_init_artist_effect(
            prefix, n_artists, mu_artist, sigma_artist, priors
        )

        # === Time-varying artist effects (latent process seam) ===
        artist_effects = _build_latent_effects(
            prefix, n_artists, max_seq, init_artist_effect, sigma_rw, priors
        )

        # album_seq is 1-indexed -> 0-indexed. Clipping at max_seq-1 reuses the
        # final latent step for any album past the longest training trajectory,
        # freezing the random walk there: prediction adds no further RW
        # innovations beyond max_seq, so deep multi-step-ahead variance is
        # understated by ~(h-max_seq)*sigma_rw^2. Immaterial for the one-step
        # flagship use (next album).
        seq_idx = jnp.clip(album_seq - 1, 0, max_seq - 1).astype(jnp.int32)
        obs_artist_effect = artist_effects[seq_idx, artist_idx]

        # === Fixed effects for covariates ===
        beta = numpyro.sample(
            f"{prefix}beta",
            dist.Normal(priors.beta_loc, priors.beta_scale).expand([n_features]).to_event(1),
        )

        # === AR(1) term for album-to-album dependency ===
        # With ar_center=0 (legacy) debuts carry the training global mean as
        # prev_score, so rho also absorbs the overall score level. Centering
        # on that same mean makes debut AR terms exactly zero and removes
        # the rho <-> mu_artist location confound.
        ar_term = rho * (prev_score - ar_center)

        # === Mean prediction (target-transform seam) ===
        # identity: soft-clip into bounds (differentiable, NUTS-friendly);
        # offset_logit: mu stays unconstrained, the sigmoid back-transform
        # guarantees bounds by construction. The transform is RNG-free, so the
        # mu call is deferred to just before the likelihood -- after the gated
        # EIV resample below -- without perturbing any draw on the gate-off path.
        mu_raw = obs_artist_effect + X @ beta + ar_term

        # === Heteroscedastic mode validation ===
        # Note: When learn_n_exponent=True, n_exp is a traced JAX value and cannot
        # be used in Python conditionals. We use the Python-level learn_n_exponent
        # flag to determine if we should apply heteroscedastic scaling.
        # When learning, we always apply scaling (that's why we're learning it).
        # When fixed, we can check if n_exp != 0 to skip unnecessary computation.
        heteroscedastic_requested = learn_n_exponent or n_exponent != 0
        if heteroscedastic_requested and n_reviews is None:
            raise ValueError(
                f"Heteroscedastic noise scaling requires n_reviews data. "
                f"Got learn_n_exponent={learn_n_exponent}, n_exponent={n_exponent}, "
                f"but n_reviews is None. Either provide n_reviews or set both "
                f"learn_n_exponent=False and n_exponent=0 for homoscedastic mode."
            )
        use_heteroscedastic = heteroscedastic_requested  # n_reviews guaranteed non-None here

        # === Exponent for heteroscedastic noise (fixed or learned) ===
        if learn_n_exponent:
            if n_exponent_prior == "logit-normal":
                # Logit-normal: sample in unbounded space, transform via sigmoid
                # This avoids funnel geometry that causes divergences with Beta prior
                n_exp = numpyro.sample(
                    f"{prefix}n_exponent",
                    dist.TransformedDistribution(
                        dist.Normal(priors.n_exponent_loc, priors.n_exponent_scale),
                        [dist.transforms.SigmoidTransform()],
                    ),
                )
            elif n_exponent_prior == "beta":
                # Beta prior (legacy, may cause divergences)
                n_exp = numpyro.sample(
                    f"{prefix}n_exponent",
                    dist.Beta(priors.n_exponent_alpha, priors.n_exponent_beta),
                )
            else:
                raise ValueError(
                    f"Invalid n_exponent_prior: '{n_exponent_prior}'. "
                    f"Must be 'logit-normal' or 'beta'."
                )
        else:
            n_exp = n_exponent  # Use fixed value from config

        # === Observation-level noise ===
        # Determine parameterization mode:
        # - sigma_ref mode: n_ref is provided AND heteroscedastic is requested
        # - Original mode: n_ref=None (homoscedastic backward compat) or no
        #   heteroscedastic scaling
        use_sigma_ref = (n_ref is not None) and heteroscedastic_requested

        if use_sigma_ref:
            if n_ref <= 0:
                raise ValueError(
                    f"n_ref must be positive for sigma-ref parameterization, got {n_ref}"
                )
            # SIGMA-REF PARAMETERIZATION: sample noise at reference review count
            # This breaks the multiplicative funnel between sigma_obs and n_exponent
            sigma_ref = numpyro.sample(
                f"{prefix}sigma_ref",
                dist.HalfNormal(priors.sigma_ref_scale),
            )
            # Derive sigma_obs deterministically (no Jacobian correction needed --
            # numpyro.deterministic only records a value, no log-density contribution)
            sigma_obs = numpyro.deterministic(
                f"{prefix}sigma_obs",
                sigma_ref * jnp.power(n_ref, n_exp),
            )
        else:
            # ORIGINAL PARAMETERIZATION: sample sigma_obs directly
            # Used when n_ref=None (homoscedastic/backward compat). The prior
            # family ("halfnormal" legacy default, or "lognormal") is selected
            # by the helper; the "halfnormal" branch is the exact legacy call,
            # so the default-config draw is bit-identical.
            sigma_obs = _sample_sigma_obs(prefix, priors)

        # === Per-observation noise scaling ===
        if use_heteroscedastic:
            sigma_scaled = compute_sigma_scaled(sigma_obs, n_reviews, n_exp)
        else:
            # Homoscedastic mode: use scalar sigma_obs for all observations
            sigma_scaled = sigma_obs

        # === Entity-level overdispersion (gated; appended after every site) ===
        # New sites live ONLY on this branch and AFTER all existing sites, so
        # the gate-off RNG draw sequence (and every published number) is
        # unchanged. When homoscedastic, sigma_scaled is a scalar; jnp.exp of
        # the per-entity factor broadcasts it to per-observation shape.
        if priors.heteroscedastic_entity_obs:
            sigma_scaled = jnp.broadcast_to(sigma_scaled, (len(artist_idx),))
            sigma_scaled = _apply_entity_overdispersion(
                prefix, sigma_scaled, artist_idx, n_artists, priors
            )

        # === Errors-in-variables on the AR regressor (gated; appended last) ===
        # De-noise the lagged score the AR term regresses on. The new site lives
        # ONLY here, after every existing site, so the gate-off draw sequence is
        # bit-identical (NumPyro splits PRNG keys in site-execution order).
        # Debuts carry prev_meas_sigma == 0, so prev_latent == prev_score there
        # and debut AR terms stay exactly zero.
        if priors.errors_in_variables:
            if prev_meas_sigma is None:
                raise ValueError(
                    "errors_in_variables=True requires prev_meas_sigma "
                    "(per-observation measurement-error scale for prev_score)."
                )
            prev_latent_raw = numpyro.sample(
                f"{prefix}prev_latent_raw",
                dist.Normal(0.0, 1.0).expand([len(artist_idx)]).to_event(1),
            )
            prev_latent = prev_score + prev_meas_sigma * prev_latent_raw
            # Rebuild mu_raw with the latent regressor. Valid only because the base
            # AR term is exactly rho * (prev_score - ar_center); if ar_term ever
            # becomes compound, this substitution must be updated to match.
            mu_raw = obs_artist_effect + X @ beta + rho * (prev_latent - ar_center)

        mu = _apply_target_transform(mu_raw, priors, target_bounds)

        # === Likelihood (family seam) ===
        _sample_likelihood(
            prefix,
            mu,
            sigma_scaled,
            y,
            len(artist_idx),
            likelihood_df,
            priors,
            target_bounds=target_bounds,
            n_reviews=n_reviews,
        )

    # Apply non-centered reparameterization to init_artist_effect
    reparam_config = {
        f"{prefix}init_artist_effect": LocScaleReparam(centered=0),
    }
    reparameterized_model = reparam(_score_model_centered, config=reparam_config)

    # Add docstring to reparameterized model
    reparameterized_model.__doc__ = f"""Non-centered hierarchical model for {score_type} scores.

This model includes:
- Time-varying artist effects via random walk (career trajectory)
- AR(1) structure for album-to-album score dependencies
- Hierarchical partial pooling of artist effects
- Non-centered parameterization via LocScaleReparam
- Optional heteroscedastic observation noise (per-observation sigma)

All sample site names are prefixed with "{prefix}" (e.g., "{prefix}beta", "{prefix}rho").

Args:
    artist_idx: Integer array of shape (n_obs,) mapping each observation
        to an artist index in [0, n_artists).
    album_seq: Integer array of shape (n_obs,) with album sequence numbers
        (1 = first album, 2 = second, etc.). Used for time-varying effects.
    prev_score: Float array of shape (n_obs,) with previous album scores.
        Debut albums are filled with the training global mean score by the
        data pipeline, not 0.
    X: Feature matrix of shape (n_obs, n_features) containing covariates.
        Features should be standardized for the default priors to be appropriate.
    y: Optional target scores of shape (n_obs,). Pass None for prior
        predictive sampling or posterior predictive on new data.
    n_artists: Number of unique artists. Required.
    max_seq: Maximum album sequence number. Required for JAX tracing.
        Compute as int(album_seq.max()) before calling the model.
    priors: Prior configuration. If None, uses get_default_priors().
    n_reviews: Optional array of shape (n_obs,) with per-observation
        review counts. Used for heteroscedastic noise scaling. If None,
        homoscedastic noise is used (scalar sigma_obs for all observations).
    n_ref: Reference review count for sigma-ref reparameterization.
        If provided (non-None) and heteroscedastic mode is active, the model
        samples sigma_ref instead of sigma_obs and derives
        sigma_obs = sigma_ref * n_ref^n_exponent as a deterministic site.
        Computed as median(n_reviews) from training data. Must be > 1.
    n_exponent: Fixed exponent for heteroscedastic noise scaling.
        Default 0.0 gives homoscedastic (constant) noise.
    learn_n_exponent: If True, sample the exponent from a prior distribution
        instead of using the fixed n_exponent value.
    n_exponent_prior: Prior type for learned n_exponent. Options:
        - "logit-normal" (default): Uses TransformedDistribution with
          Normal(loc, scale) and SigmoidTransform. Recommended to avoid
          funnel geometry that causes divergences.
        - "beta" (legacy): Uses Beta(alpha, beta) distribution. May cause
          divergences due to challenging posterior geometry.
    likelihood_df: Degrees of freedom for Student-t likelihood. Default 4.0
        gives heavier tails than Normal, reducing overprediction for extreme
        scores (0-30 range). Set >= 100 to recover Normal likelihood.
    ar_center: Centering value for the AR(1) term:
        ar_term = rho * (prev_score - ar_center). Default 0.0 keeps the
        legacy uncentered behavior. Must be on the model's training scale.
    prev_meas_sigma: Per-observation measurement-error scale for prev_score,
        required only when priors.errors_in_variables is True (debuts pinned
        to 0). Ignored on the default (gate-off) path.

Returns:
    None. Model samples are tracked by NumPyro internally.

Sample sites (all prefixed with "{prefix}"):
    - {prefix}mu_artist: Population mean of artist effects
    - {prefix}sigma_artist: Between-artist standard deviation
    - {prefix}sigma_rw: Random walk innovation scale
    - {prefix}rho: AR(1) coefficient
    - {prefix}init_artist_effect: Initial artist effects (partial pooling)
    - {prefix}rw_raw: Unit-normal random walk innovations (n_artists x max_seq-1)
    - {prefix}prev_latent_raw: Unit-normal AR-regressor measurement-error
        innovations, shape (n_obs,) (only when priors.errors_in_variables=True)
    - {prefix}beta: Fixed effect coefficients
    - {prefix}sigma_ref: Noise at reference review count (only when n_ref is
        provided and heteroscedastic mode active)
    - {prefix}sigma_obs: Observation noise base scale. Sampled directly in
        original mode; deterministic (derived from sigma_ref) when n_ref is
        provided and heteroscedastic mode is active.
    - {prefix}n_exponent: Heteroscedastic scaling exponent (only when learn_n_exponent=True)
    - {prefix}y: Observed/predicted scores

Example:
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from numpyro.infer import MCMC, NUTS
    >>> from panelcast.models.bayes.model import {score_type}_score_model
    >>>
    >>> # Prepare data
    >>> n_obs, n_features, n_artists = 100, 5, 20
    >>> artist_idx = jnp.array([i % n_artists for i in range(n_obs)])
    >>> album_seq = jnp.array([(i // n_artists) + 1 for i in range(n_obs)])
    >>> max_seq = int(album_seq.max())  # Compute before tracing
    >>> prev_score = jnp.full(n_obs, 75.0)  # pipeline fills debuts with global mean
    >>> X = random.normal(random.PRNGKey(0), (n_obs, n_features))
    >>> y = random.normal(random.PRNGKey(1), (n_obs,)) * 10 + 70
    >>>
    >>> # Run MCMC
    >>> mcmc = MCMC(NUTS({score_type}_score_model), num_warmup=100, num_samples=100)
    >>> mcmc.run(
    ...     random.PRNGKey(2),
    ...     artist_idx=artist_idx,
    ...     album_seq=album_seq,
    ...     prev_score=prev_score,
    ...     X=X,
    ...     y=y,
    ...     n_artists=n_artists,
    ...     max_seq=max_seq
    ... )
    >>> samples = mcmc.get_samples()
    >>> print("{prefix}rho" in samples)
    True
"""

    return reparameterized_model


# Create the two exported model functions
user_score_model = make_score_model("user")
critic_score_model = make_score_model("critic")

# Backwards compatibility alias
album_score_model = user_score_model
