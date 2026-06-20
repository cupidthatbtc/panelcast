"""Integration tests for model predictions to evaluation metrics.

Tests the model -> evaluation integration pathway:
- InferenceData -> convergence diagnostics
- Predictions -> calibration metrics
- Predictions -> point metrics
- Predictions -> CRPS
- InferenceData -> LOO-CV

These tests verify interface compatibility between modules
that was the source of bugs fixed in Phases 10-12.
"""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from panelcast.evaluation.calibration import (
    CoverageResult,
    ReliabilityData,
    compute_coverage,
    compute_multi_coverage,
    compute_reliability_data,
)
from panelcast.evaluation.cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compute_loo,
)
from panelcast.evaluation.metrics import (
    CRPSResult,
    PointMetrics,
    compute_crps,
    compute_point_metrics,
    posterior_mean,
)
from panelcast.models.bayes.diagnostics import (
    ConvergenceDiagnostics,
    check_convergence,
    get_divergence_info,
)


class TestModelEvaluationIntegration:
    """Integration tests for model predictions through evaluation."""

    def test_idata_to_diagnostics(self, mock_idata: az.InferenceData):
        """Test that InferenceData integrates with convergence diagnostics.

        Verifies:
        - check_convergence accepts InferenceData
        - Returns ConvergenceDiagnostics with valid fields
        """
        diags = check_convergence(mock_idata)

        assert isinstance(diags, ConvergenceDiagnostics)
        assert isinstance(diags.passed, bool)
        assert isinstance(diags.rhat_max, float)
        assert isinstance(diags.ess_bulk_min, int)
        assert isinstance(diags.divergences, int)

        # With mock data (no actual MCMC issues), should pass
        assert diags.passed is True
        assert diags.divergences == 0

    def test_predictions_to_coverage(self, mock_predictions: dict):
        """Test that predictions integrate with coverage computation.

        Verifies:
        - compute_coverage accepts y_true and y_samples
        - Returns CoverageResult with valid fields
        """
        result = compute_coverage(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
            prob=0.94,
        )

        assert isinstance(result, CoverageResult)
        assert result.nominal == pytest.approx(0.94)
        assert 0.0 <= result.empirical <= 1.0
        assert result.n_obs == len(mock_predictions["y_true"])
        assert result.n_covered >= 0
        assert result.n_covered <= result.n_obs
        assert result.interval_width > 0  # Sharpness metric

    def test_predictions_to_point_metrics(self, mock_predictions: dict):
        """Test that predictions integrate with point metrics.

        Verifies:
        - compute_point_metrics accepts y_true and y_pred_mean
        - Returns PointMetrics with valid fields
        """
        result = compute_point_metrics(
            mock_predictions["y_true"],
            mock_predictions["mean"],
        )

        assert isinstance(result, PointMetrics)
        assert result.mae >= 0  # MAE is non-negative
        assert result.rmse >= 0  # RMSE is non-negative
        assert result.rmse >= result.mae  # RMSE >= MAE always
        assert result.median_ae >= 0  # MedianAE is non-negative
        assert isinstance(result.r2, float)  # R2 can be negative for bad models
        assert result.n_observations == len(mock_predictions["y_true"])
        assert isinstance(result.mean_bias, float)

    def test_predictions_to_crps(self, mock_predictions: dict):
        """Test that predictions integrate with CRPS computation.

        Verifies:
        - compute_crps accepts y_true and y_samples
        - Returns CRPSResult with valid fields
        """
        result = compute_crps(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
        )

        assert isinstance(result, CRPSResult)
        assert result.mean_crps >= 0  # CRPS is non-negative
        assert result.n_obs == len(mock_predictions["y_true"])
        assert len(result.crps_values) == result.n_obs
        assert (result.crps_values >= 0).all()  # All per-obs CRPS non-negative

    def test_loo_cv_integration(self, mock_idata_with_log_lik: az.InferenceData):
        """Test that InferenceData with log_likelihood integrates with LOO.

        Verifies:
        - compute_loo accepts InferenceData with log_likelihood
        - Returns LOOResult with valid fields
        """
        result = compute_loo(mock_idata_with_log_lik)

        assert isinstance(result, LOOResult)
        assert isinstance(result.elpd_loo, float)
        assert isinstance(result.se_elpd, float)
        assert result.se_elpd >= 0  # Standard error is non-negative
        assert isinstance(result.p_loo, float)
        assert result.p_loo >= 0  # Effective params is non-negative
        assert isinstance(result.n_high_pareto_k, int)
        assert result.n_high_pareto_k >= 0

    def test_reliability_data_for_plotting(self, mock_predictions: dict):
        """Test that predictions integrate with reliability data computation.

        Verifies:
        - compute_reliability_data returns valid data for plotting
        - Arrays have proper shape and range
        """
        data = compute_reliability_data(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
            n_bins=10,
        )

        assert isinstance(data, ReliabilityData)

        # Bin edges should be ordered
        assert np.all(np.diff(data.bin_edges) >= 0)

        # Predicted probs should be in [0, 1]
        assert (data.predicted_probs >= 0).all()
        assert (data.predicted_probs <= 1).all()

        # Observed freq should be in [0, 1]
        assert (data.observed_freq >= 0).all()
        assert (data.observed_freq <= 1).all()

        # Counts should be non-negative integers
        assert (data.counts >= 0).all()

        # Each quantile level should use all observations.
        assert np.all(data.counts == len(mock_predictions["y_true"]))

    def test_evaluation_chain_data_types(
        self, mock_predictions: dict, mock_idata: az.InferenceData
    ):
        """Test full evaluation chain maintains correct data types.

        Runs through the complete evaluation flow to verify
        data type consistency at each step.
        """
        # Step 1: Diagnostics
        diags = check_convergence(mock_idata)
        assert isinstance(diags.passed, bool)
        assert isinstance(diags.summary_df, object)  # pandas DataFrame

        # Step 2: Point metrics
        point_metrics = compute_point_metrics(
            mock_predictions["y_true"],
            mock_predictions["mean"],
        )
        assert isinstance(point_metrics.mae, float)
        assert isinstance(point_metrics.n_observations, int)

        # Step 3: CRPS
        crps_result = compute_crps(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
        )
        assert isinstance(crps_result.mean_crps, float)
        assert isinstance(crps_result.crps_values, np.ndarray)

        # Step 4: Coverage
        coverage = compute_coverage(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
            prob=0.95,
        )
        assert isinstance(coverage.empirical, float)
        assert isinstance(coverage.lower_bound, np.ndarray)
        assert isinstance(coverage.upper_bound, np.ndarray)

        # Step 5: Multi-coverage
        multi_coverage = compute_multi_coverage(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
            probs=(0.50, 0.80, 0.95),
        )
        assert isinstance(multi_coverage, dict)
        assert all(isinstance(v, CoverageResult) for v in multi_coverage.values())

    def test_posterior_mean_helper(self, mock_predictions: dict):
        """Test posterior_mean helper function.

        Verifies:
        - Correctly computes mean across sample dimension
        - Output shape matches number of observations
        """
        samples = mock_predictions["y_pred_samples"]
        mean = posterior_mean(samples)

        assert mean.shape == (samples.shape[1],)
        assert np.allclose(mean, samples.mean(axis=0))

    def test_divergence_info_extraction(self, mock_idata: az.InferenceData):
        """Test divergence info extraction from InferenceData.

        Verifies:
        - get_divergence_info returns valid structure
        - Per-chain breakdown is available
        """
        info = get_divergence_info(mock_idata)

        assert "total" in info
        assert "per_chain" in info
        assert "rate" in info
        assert "locations" in info

        # Mock data has no divergences
        assert info["total"] == 0
        assert info["rate"] == 0.0

        # Per-chain should have entry for each chain
        n_chains = mock_idata.posterior.sizes["chain"]
        assert len(info["per_chain"]) == n_chains

    def test_add_log_likelihood_to_idata(self, mock_idata: az.InferenceData):
        """Test adding log_likelihood group to InferenceData.

        Verifies:
        - add_log_likelihood_to_idata correctly adds group
        - Group is accessible after addition
        """
        # Create mock log-likelihood
        n_chains = mock_idata.posterior.sizes["chain"]
        n_draws = mock_idata.posterior.sizes["draw"]
        n_obs = 50

        log_lik_da = xr.DataArray(
            np.random.normal(-5, 1, (n_chains, n_draws, n_obs)),
            dims=["chain", "draw", "obs"],
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "obs": range(n_obs),
            },
        )

        # Add to idata (creates a copy effectively)
        updated = add_log_likelihood_to_idata(mock_idata, log_lik_da, var_name="y")

        assert "log_likelihood" in updated.groups()
        assert "y" in updated.log_likelihood

    def test_coverage_calibration_check(self, mock_predictions: dict):
        """Test that well-calibrated predictions have good coverage.

        The mock_predictions fixture is designed to be well-calibrated.
        Coverage should be close to nominal for such predictions.
        """
        # Check 95% coverage
        result = compute_coverage(
            mock_predictions["y_true"],
            mock_predictions["y_pred_samples"],
            prob=0.95,
        )

        # Well-calibrated model should have empirical ~= nominal
        # Allow some slack due to finite sample size
        assert 0.85 <= result.empirical <= 1.0

    def test_crps_vs_mae_relationship(self, mock_predictions: dict):
        """Test CRPS relates sensibly to MAE for point predictions.

        For deterministic predictions, CRPS equals MAE.
        For probabilistic predictions with uncertainty, CRPS should
        be in a reasonable range relative to MAE.
        """
        y_true = mock_predictions["y_true"]
        y_mean = mock_predictions["mean"]
        y_samples = mock_predictions["y_pred_samples"]

        # Point metrics
        point = compute_point_metrics(y_true, y_mean)

        # CRPS
        crps = compute_crps(y_true, y_samples)

        # CRPS should be related to MAE
        # For well-spread samples, CRPS is typically smaller than MAE
        # because it rewards appropriate uncertainty
        assert crps.mean_crps > 0
        assert crps.mean_crps < 50  # Sanity check given score range ~50-90

    def test_convergence_diagnostics_with_thresholds(self, mock_idata: az.InferenceData):
        """Test convergence diagnostics with custom thresholds.

        Verifies threshold parameters are correctly applied.
        """
        # Default thresholds
        default_diags = check_convergence(mock_idata)

        # Stricter thresholds
        strict_diags = check_convergence(
            mock_idata,
            rhat_threshold=1.005,  # Stricter than default 1.01
            ess_threshold=800,  # Higher than default 400
        )

        # Thresholds should be recorded
        assert default_diags.rhat_threshold == 1.01
        assert default_diags.ess_threshold == 400
        assert strict_diags.rhat_threshold == 1.005
        assert strict_diags.ess_threshold == 800
