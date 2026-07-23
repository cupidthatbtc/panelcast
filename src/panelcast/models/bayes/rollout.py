"""Multi-step-ahead forecasting via ancestral rollout (#157).

Every held-out number the evaluate stage publishes is one-step-ahead: the
model conditions on the *observed* previous score (teacher forcing) and the
latent trajectory is frozen past the training horizon. ``predict_horizon``
produces genuine h-step forecasts by integrating over the model's own
predictions: per posterior draw, it samples y_{t+1}, feeds it back as the AR
lag for t+2, and compounds a fresh latent innovation per step — so
uncertainty grows through both the latent process and the AR channel.

Conventions (documented so the honesty story doesn't leak):

- Covariates are supplied by the caller per step (``X_future``). Evaluation
  uses the realized held-out covariates; production callers must hold at
  last-known values or supply their own futures.
- The terminal latent state is re-sampled from the prior conditional on the
  posterior hyperparameters — the same marginalization the saved posterior
  forces on ``predict_next`` (``rw_raw`` is excluded from collection), so a
  1-step rollout matches the existing evaluation in distribution.
- The rollout always propagates latent innovations past the training horizon;
  the ``propagate_rw_horizon`` gate exists to give the Predictive-based
  one-step path the same behavior and has no separate role here.
- The fed-back AR lag is the sampled score clipped to ``target_bounds``
  (observed scores are bounded by construction; an identity-transform
  Student-t draw is not) and is treated as exact — the errors-in-variables
  measurement model applies to *observed* lags, not self-generated ones.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, random

from panelcast.models.bayes.likelihoods import available_families, find_likelihood
from panelcast.models.bayes.model import compute_sigma_scaled
from panelcast.models.bayes.transforms import get_transform

__all__ = ["predict_horizon"]


def predict_horizon(
    posterior_samples: dict,
    *,
    artist_idx: jnp.ndarray,
    n_train_events: jnp.ndarray,
    y_last: jnp.ndarray,
    X_future: jnp.ndarray,
    n_reviews_future: jnp.ndarray | None = None,
    dynamic_mask: jnp.ndarray | None = None,
    prefix: str = "user_",
    seed: int = 7,
    ar_center: float = 0.0,
    target_bounds: tuple[float, float] = (0.0, 100.0),
    likelihood_df: float = 4.0,
    target_transform: str = "identity",
    logit_offset: float = 0.5,
    likelihood_family: str = "studentt",
    skew_tailweight: float = 1.0,
    discretize_observation: bool = False,
    latent_process: str = "rw",
    fixed_n_exponent: float | None = None,
) -> dict:
    """Ancestral h-step rollout for entities seen in training.

    Args:
        posterior_samples: Flattened posterior dict (``mcmc.get_samples()``
            layout: leading axis is the draw).
        artist_idx: (n_entities,) int indices into the fitted entity axis.
        n_train_events: (n_entities,) training-history length per entity; the
            terminal latent deviation is re-sampled with the variance the
            latent process accumulates over ``n_train_events - 1`` steps.
        y_last: (n_entities,) last observed score per entity on the MODEL
            scale (forward-transformed under a non-identity transform) — the
            h=1 AR lag.
        X_future: (H, n_entities, n_features) standardized covariates per
            step. Realized values for evaluation; futures for production.
        n_reviews_future: (H, n_entities) observation counts per step.
            Required for heteroscedastic fits and count-based likelihoods.
        dynamic_mask: (n_entities,) bool; False freezes the latent effect
            (entities below ``min_albums_filter`` train with a static effect
            and must stay static here). Default all-dynamic.
        ar_center: AR centering value on the model scale (scalar).
        fixed_n_exponent: heteroscedastic exponent for fits trained with a
            fixed non-zero exponent (learned exponents ride the posterior).

    Returns:
        dict with ``y`` and ``mu``, both (n_draws, H, n_entities) on the
        score scale.
    """
    spec = find_likelihood(likelihood_family)
    if spec is None:
        raise ValueError(
            f"Unknown likelihood_family: '{likelihood_family}'. "
            f"Registered: {list(available_families())}."
        )

    init_effect = posterior_samples[f"{prefix}init_artist_effect"][:, artist_idx]
    sigma_rw = posterior_samples[f"{prefix}sigma_rw"]
    rho = posterior_samples[f"{prefix}rho"]
    beta = posterior_samples[f"{prefix}beta"]
    sigma_obs = posterior_samples[f"{prefix}sigma_obs"]
    family_sites = {
        name: posterior_samples.get(f"{prefix}{name}") for name in spec.required_sites
    }

    n_samples = init_effect.shape[0]
    n_entities = init_effect.shape[1]
    H = X_future.shape[0]
    if H < 1:
        raise ValueError(f"X_future must cover at least one step, got H={H}.")

    if latent_process == "ar1":
        damp = posterior_samples[f"{prefix}phi"]
    elif latent_process == "rw":
        damp = jnp.ones((n_samples,))
    else:
        raise ValueError(
            f"Unknown latent_process: '{latent_process}'. Registered: ['rw', 'ar1']."
        )

    n_exp = None
    if f"{prefix}n_exponent" in posterior_samples:
        n_exp = posterior_samples[f"{prefix}n_exponent"]
    elif fixed_n_exponent is not None and fixed_n_exponent != 0:
        n_exp = jnp.full((n_samples,), fixed_n_exponent)
    if n_reviews_future is None:
        if n_exp is not None:
            raise ValueError(
                "n_reviews_future is required for rollout under heteroscedastic "
                "noise (learned or fixed n_exponent)."
            )
        if spec.requires_aggregation_count:
            raise ValueError(
                f"n_reviews_future is required for the {likelihood_family} family."
            )
        n_reviews_future = jnp.ones((H, n_entities), dtype=jnp.float32)

    # Entity overdispersion: seen entities keep their fitted inflation when the
    # per-entity site survived collection; otherwise fall back to no inflation
    # (the marginal treatment belongs to cold-start prediction, not rollout).
    entity_log_scale = posterior_samples.get(f"{prefix}entity_log_scale")
    overdisp = (
        jnp.exp(entity_log_scale[:, artist_idx]) if entity_log_scale is not None else 1.0
    )

    if dynamic_mask is None:
        dynamic_mask = jnp.ones((n_entities,), dtype=bool)
    dyn = jnp.asarray(dynamic_mask, dtype=jnp.float32)

    transform = get_transform(target_transform, target_bounds=target_bounds, offset=logit_offset)
    low, high = float(target_bounds[0]), float(target_bounds[1])

    # Terminal latent deviation, re-sampled in distribution: over T-1 steps the
    # random walk accumulates variance (T-1)*sigma_rw^2; the AR(1) deviation
    # accumulates sigma_rw^2 * (1 - phi^(2(T-1))) / (1 - phi^2).
    steps = jnp.maximum(jnp.asarray(n_train_events, dtype=jnp.float32) - 1.0, 0.0) * dyn
    if latent_process == "ar1":
        phi2 = damp[:, None] ** 2
        var_ratio = (1.0 - phi2 ** steps[None, :]) / (1.0 - phi2)
        term_sd = sigma_rw[:, None] * jnp.sqrt(var_ratio)
    else:
        term_sd = sigma_rw[:, None] * jnp.sqrt(steps)[None, :]

    key = random.key(seed)
    key, k_term = random.split(key)
    dev0 = term_sd * random.normal(k_term, (n_samples, n_entities))

    innov_scale = sigma_rw[:, None] * dyn[None, :]
    y_prev0 = jnp.broadcast_to(
        jnp.asarray(y_last, dtype=jnp.float32)[None, :], (n_samples, n_entities)
    )
    step_keys = random.split(key, H)

    def _step(carry, xs):
        dev, y_prev = carry
        X_h, nrev_h, k = xs
        k_eps, k_y = random.split(k)
        dev = damp[:, None] * dev + innov_scale * random.normal(k_eps, dev.shape)
        mu_raw = (
            init_effect
            + dev
            + jnp.einsum("sf,ef->se", beta, X_h)
            + rho[:, None] * (y_prev - ar_center)
        )
        mu = transform.transform_mu(mu_raw)
        if n_exp is not None:
            sigma = compute_sigma_scaled(sigma_obs[:, None], nrev_h[None, :], n_exp[:, None])
        else:
            sigma = jnp.broadcast_to(sigma_obs[:, None], mu.shape)
        sigma = sigma * overdisp
        y_score = spec.predict_draws(
            k_y,
            mu,
            sigma,
            sites=family_sites,
            df=likelihood_df,
            bounds=target_bounds,
            skew_tailweight=skew_tailweight,
            transform=transform,
            discretize=discretize_observation,
            n_reviews=nrev_h,
        )
        y_prev_next = transform.forward(jnp.clip(y_score, low, high))
        return (dev, y_prev_next), (y_score, transform.inverse(mu))

    X_future = jnp.asarray(X_future, dtype=jnp.float32)
    n_reviews_future = jnp.asarray(n_reviews_future, dtype=jnp.float32)
    _, (ys, mus) = lax.scan(_step, (dev0, y_prev0), (X_future, n_reviews_future, step_keys))

    return {
        "y": jnp.moveaxis(ys, 0, 1),
        "mu": jnp.moveaxis(mus, 0, 1),
    }
