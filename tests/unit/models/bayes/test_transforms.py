"""Tests for the target-transform registry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.transforms import get_transform


class TestRegistry:
    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown target_transform"):
            get_transform("bogus")

    def test_identity_and_offset_logit_registered(self):
        assert get_transform("identity").name == "identity"
        assert get_transform("offset_logit").name == "offset_logit"


class TestIdentity:
    def test_forward_inverse_are_noops(self):
        t = get_transform("identity", (0.0, 100.0))
        y = jnp.array([0.0, 50.0, 100.0])
        assert np.allclose(t.forward(y), y)
        assert np.allclose(t.inverse(y), y)
        assert np.allclose(t.log_jacobian(y), 0.0)

    def test_transform_mu_soft_clips(self):
        t = get_transform("identity", (0.0, 100.0))
        # Values near the bounds saturate smoothly; far outside they clamp
        # to the bound in float32.
        mu = t.transform_mu(jnp.array([-1.0, 50.0, 101.0]))
        assert 0.0 <= float(mu[0]) < 1.0
        assert 99.0 < float(mu[2]) <= 100.0
        assert float(mu[1]) == pytest.approx(50.0, abs=1e-3)


class TestOffsetLogit:
    @pytest.mark.parametrize("bounds", [(0.0, 100.0), (0.0, 10.0), (-5.0, 5.0)])
    def test_round_trip(self, bounds):
        t = get_transform("offset_logit", bounds, offset=0.5)
        y = jnp.linspace(bounds[0], bounds[1], 41)
        back = t.inverse(t.forward(y))
        assert np.allclose(back, y, atol=1e-4)

    def test_matches_smithson_verkuilen_aoty_form(self):
        """For (0,100), offset 0.5: z = logit((y+0.5)/101)."""
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        y = jnp.array([0.0, 75.0, 100.0])
        p = (y + 0.5) / 101.0
        expected = np.log(p) - np.log1p(-p)
        assert np.allclose(t.forward(y), expected, atol=1e-6)

    def test_bounds_map_to_finite_values(self):
        t = get_transform("offset_logit", (0.0, 100.0))
        z = t.forward(jnp.array([0.0, 100.0]))
        assert np.all(np.isfinite(np.asarray(z)))

    def test_inverse_always_inside_bounds_extension(self):
        """sigmoid output stays within [low-offset, high+offset]."""
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        z = jnp.array([-50.0, 0.0, 50.0])
        y = np.asarray(t.inverse(z))
        # Extreme z saturate to the closed extension bounds in float32.
        assert np.all(y >= -0.5) and np.all(y <= 100.5)
        assert y[1] == pytest.approx(50.0, abs=0.01)

    def test_log_jacobian_matches_autodiff(self):
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        y_points = jnp.array([1.0, 25.0, 75.0, 99.0])
        grad_fn = jax.grad(lambda y: t.forward(y))
        for y in y_points:
            expected = float(jnp.log(jnp.abs(grad_fn(y))))
            actual = float(t.log_jacobian(y))
            assert actual == pytest.approx(expected, rel=1e-5)

    def test_transform_mu_is_identity(self):
        t = get_transform("offset_logit", (0.0, 100.0))
        mu = jnp.array([-3.0, 0.0, 3.0])
        assert np.allclose(t.transform_mu(mu), mu)
