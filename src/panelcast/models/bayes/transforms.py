"""Target-transform registry for the score model.

A target transform defines how the bounded observed score relates to the
scale the likelihood operates on:

- ``forward(y)``: data scale -> model scale (applied to y and prev_score
  during training preparation).
- ``inverse(z)``: model scale -> data scale (applied to predictive draws).
- ``log_jacobian(y)``: log |d forward / d y| at data-scale y; needed to put
  per-observation log-likelihoods of different transforms on a common
  (data) scale for LOO/WAIC comparison.
- ``transform_mu(mu_raw)``: how the model maps the raw linear predictor to
  the likelihood location parameter (e.g. identity soft-clips into bounds;
  a logit-scale model leaves mu unconstrained).

Registering a new transform is a single :func:`register_transform` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from panelcast.models.bayes.model_math import soft_clip


@dataclass(frozen=True)
class TargetTransform:
    """A target transform bound to concrete bounds/offset parameters."""

    name: str
    bounds: tuple[float, float]
    offset: float
    forward: Callable
    inverse: Callable
    log_jacobian: Callable
    transform_mu: Callable


_TRANSFORM_FACTORIES: dict[str, Callable[[tuple[float, float], float], TargetTransform]] = {}


def register_transform(
    name: str,
) -> Callable[[Callable[[tuple[float, float], float], TargetTransform]], Callable]:
    """Register a transform factory under a name."""

    def _decorator(factory: Callable[[tuple[float, float], float], TargetTransform]):
        _TRANSFORM_FACTORIES[name] = factory
        return factory

    return _decorator


def get_transform(
    name: str,
    target_bounds: tuple[float, float] = (0.0, 100.0),
    offset: float = 0.5,
) -> TargetTransform:
    """Resolve a registered transform by name with concrete parameters."""
    try:
        factory = _TRANSFORM_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown target_transform: '{name}'. Registered: {sorted(_TRANSFORM_FACTORIES)}"
        ) from None
    return factory(tuple(target_bounds), float(offset))


@register_transform("identity")
def _identity_transform(bounds: tuple[float, float], offset: float) -> TargetTransform:
    """Legacy behavior: scores stay on their natural scale; the model
    soft-clips the linear predictor into the target bounds."""
    low, high = bounds

    return TargetTransform(
        name="identity",
        bounds=bounds,
        offset=offset,
        forward=lambda y: y,
        inverse=lambda z: z,
        log_jacobian=lambda y: jnp.zeros_like(jnp.asarray(y, dtype=jnp.result_type(float))),
        transform_mu=lambda mu_raw: soft_clip(mu_raw, low=low, high=high),
    )


@register_transform("offset_logit")
def _offset_logit_transform(bounds: tuple[float, float], offset: float) -> TargetTransform:
    """Smithson-Verkuilen offset logit.

    ``z = logit((y - low + offset) / (high - low + 2*offset))`` maps the
    closed interval [low, high] strictly inside the logit's domain, so
    boundary scores stay finite. The model runs entirely on the logit
    scale; the sigmoid back-transform guarantees predictions inside the
    bounds *by construction*, regenerating ceiling skew on the score scale
    from a symmetric likelihood — which is exactly why this transform fixes
    the pinned PPC skew/max statistics.

    transform_mu is the identity: no soft-clipping on the logit scale.
    """
    low, high = bounds
    span = high - low + 2.0 * offset

    def forward(y):
        p = (jnp.asarray(y) - low + offset) / span
        return jnp.log(p) - jnp.log1p(-p)

    def inverse(z):
        return low - offset + span * jax.nn.sigmoid(jnp.asarray(z))

    def log_jacobian(y):
        p = (jnp.asarray(y) - low + offset) / span
        return -jnp.log(span) - jnp.log(p) - jnp.log1p(-p)

    return TargetTransform(
        name="offset_logit",
        bounds=bounds,
        offset=offset,
        forward=forward,
        inverse=inverse,
        log_jacobian=log_jacobian,
        transform_mu=lambda mu_raw: mu_raw,
    )
