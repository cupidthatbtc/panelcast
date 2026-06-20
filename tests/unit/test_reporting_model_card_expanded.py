"""Expanded tests for reporting/model_card.py: ModelCardData, generate, write, update."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from panelcast.reporting.model_card import (
    ModelCardData,
    _latex_escape,
    create_default_model_card_data,
    generate_model_card,
    update_model_card_with_results,
    write_model_card,
)


@pytest.fixture
def minimal_data():
    """Create minimal ModelCardData."""
    return ModelCardData(
        model_name="Min Model",
        model_version="0.0.1",
        model_type="Linear",
        authors=["A"],
        created_date="2026-01-01",
        last_updated="2026-01-01",
        dataset_name="Test",
        dataset_size=10,
        dataset_description="desc",
        data_preprocessing="none",
        architecture_summary="arch",
        priors_description="priors",
    )


class TestModelCardDataExpanded:
    """Expanded tests for ModelCardData."""

    def test_mutable(self, minimal_data):
        """ModelCardData is not frozen, so fields can be set."""
        minimal_data.model_name = "Updated"
        assert minimal_data.model_name == "Updated"

    def test_empty_limitations(self, minimal_data):
        assert minimal_data.limitations == []

    def test_empty_ethical_considerations(self, minimal_data):
        assert minimal_data.ethical_considerations == []

    def test_default_loo_elpd_none(self, minimal_data):
        assert minimal_data.loo_elpd is None

    def test_hyperparameters_default_empty(self, minimal_data):
        assert minimal_data.hyperparameters == {}

    def test_with_all_optional_fields(self):
        data = ModelCardData(
            model_name="Full",
            model_version="1.0.0",
            model_type="Bayesian",
            authors=["A", "B"],
            created_date="2026-01-01",
            last_updated="2026-02-01",
            dataset_name="DS",
            dataset_size=5000,
            dataset_description="d",
            data_preprocessing="p",
            architecture_summary="a",
            priors_description="pr",
            hyperparameters={"lr": 0.01},
            convergence_summary="Converged",
            calibration_summary="Good",
            predictive_summary="MAE=2",
            loo_elpd=-100.5,
            limitations=["L1"],
            ethical_considerations=["E1"],
            intended_use="Research",
            out_of_scope_use="Production",
            load_example="load()",
            predict_example="predict()",
            interpret_example="print()",
        )
        assert data.loo_elpd == -100.5
        assert data.limitations == ["L1"]


class TestGenerateModelCardExpanded:
    """Expanded tests for generate_model_card."""

    def test_markdown_has_elpd(self):
        data = create_default_model_card_data()
        data.loo_elpd = -1234.5
        # Need to regenerate to test
        data2 = ModelCardData(
            model_name="T",
            model_version="0.1.0",
            model_type="T",
            authors=["A"],
            created_date="2026-01-01",
            last_updated="2026-01-01",
            dataset_name="D",
            dataset_size=100,
            dataset_description="d",
            data_preprocessing="p",
            architecture_summary="a",
            priors_description="pr",
            loo_elpd=-1234.5,
        )
        card = generate_model_card(data2, format="markdown")
        assert "ELPD (LOO-CV):" in card
        assert "-1234.5" in card

    def test_markdown_no_elpd(self, minimal_data):
        card = generate_model_card(minimal_data, format="markdown")
        assert "ELPD (LOO-CV):" not in card

    def test_markdown_no_hyperparameters(self, minimal_data):
        card = generate_model_card(minimal_data, format="markdown")
        assert "No hyperparameters specified." in card

    def test_markdown_with_hyperparameters(self, minimal_data):
        minimal_data.hyperparameters = {"alpha": 0.5, "beta": 1.0}
        card = generate_model_card(minimal_data, format="markdown")
        assert "| alpha | 0.5 |" in card
        assert "| beta | 1.0 |" in card

    def test_markdown_no_limitations(self, minimal_data):
        card = generate_model_card(minimal_data, format="markdown")
        assert "No limitations documented." in card

    def test_markdown_with_limitations(self, minimal_data):
        minimal_data.limitations = ["Limit A", "Limit B"]
        card = generate_model_card(minimal_data, format="markdown")
        assert "- Limit A" in card
        assert "- Limit B" in card

    def test_markdown_no_ethical(self, minimal_data):
        card = generate_model_card(minimal_data, format="markdown")
        assert "No ethical considerations documented." in card

    def test_latex_has_booktabs(self, minimal_data):
        minimal_data.hyperparameters = {"k": 1}
        card = generate_model_card(minimal_data, format="latex")
        assert "\\toprule" in card
        assert "\\bottomrule" in card

    def test_latex_no_hyperparams(self, minimal_data):
        card = generate_model_card(minimal_data, format="latex")
        assert "No hyperparameters specified." in card

    def test_latex_limitations_itemize(self, minimal_data):
        minimal_data.limitations = ["L1"]
        card = generate_model_card(minimal_data, format="latex")
        assert "\\begin{itemize}" in card

    def test_dataset_size_comma_formatted(self):
        data = ModelCardData(
            model_name="T",
            model_version="0.1.0",
            model_type="T",
            authors=["A"],
            created_date="2026-01-01",
            last_updated="2026-01-01",
            dataset_name="D",
            dataset_size=12345,
            dataset_description="d",
            data_preprocessing="p",
            architecture_summary="a",
            priors_description="pr",
        )
        card = generate_model_card(data, format="markdown")
        assert "12,345" in card


class TestLatexEscapeExpanded:
    """Expanded tests for _latex_escape."""

    def test_tilde(self):
        assert "\\textasciitilde{}" in _latex_escape("~")

    def test_caret(self):
        assert "\\textasciicircum{}" in _latex_escape("^")

    def test_multiple_specials(self):
        result = _latex_escape("$100 & 50%")
        assert "\\$" in result
        assert "\\&" in result
        assert "\\%" in result

    def test_empty_string(self):
        assert _latex_escape("") == ""

    def test_digits_unchanged(self):
        assert _latex_escape("12345") == "12345"


class TestWriteModelCardExpanded:
    """Expanded write_model_card tests."""

    def test_md_content_readable(self, minimal_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mc"
            paths = write_model_card(minimal_data, out, formats=("md",))
            content = paths[0].read_text(encoding="utf-8")
            assert "# Model Card:" in content

    def test_tex_content_readable(self, minimal_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mc"
            paths = write_model_card(minimal_data, out, formats=("tex",))
            content = paths[0].read_text(encoding="utf-8")
            assert "\\begin{document}" in content

    def test_overwrite_existing(self, minimal_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mc"
            write_model_card(minimal_data, out, formats=("md",))
            # Write again - should overwrite without error
            paths = write_model_card(minimal_data, out, formats=("md",))
            assert paths[0].exists()


class TestUpdateModelCardExpanded:
    """Expanded update_model_card_with_results tests."""

    def test_convergence_single_chain_note(self, minimal_data):
        """Single chain has non-finite rhat/ess, should add note."""
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = float("nan")
        convergence.ess_bulk_min = float("nan")
        convergence.ess_tail_min = float("nan")
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(minimal_data, convergence=convergence)
        assert "unavailable" in updated.convergence_summary
        assert ">=2 chains" in updated.convergence_summary

    def test_convergence_with_divergences(self, minimal_data):
        convergence = MagicMock()
        convergence.passed = False
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 15
        convergence.failing_params = []

        updated = update_model_card_with_results(minimal_data, convergence=convergence)
        assert "15" in updated.convergence_summary

    def test_point_metrics_with_loo(self, minimal_data):
        """Both point_metrics and loo_result should combine in predictive_summary."""
        pm = MagicMock()
        pm.mae = 3.0
        pm.rmse = 4.0
        pm.r2 = 0.85

        loo = MagicMock()
        loo.elpd_loo = -500.0
        loo.se_elpd = 20.0

        updated = update_model_card_with_results(minimal_data, point_metrics=pm, loo_result=loo)
        assert "MAE: 3.00" in updated.predictive_summary
        assert "ELPD" in updated.predictive_summary
        assert updated.loo_elpd == -500.0

    def test_ppc_with_non_numeric(self, minimal_data):
        """PPC summary with non-numeric values should still render."""
        ppc_summary = {
            "mean": {"observed": "N/A", "p_value": "N/A", "mc_se": "N/A"},
        }
        updated = update_model_card_with_results(minimal_data, ppc_summary=ppc_summary)
        assert "mean" in updated.calibration_summary

    def test_empty_ppc_summary(self, minimal_data):
        """Empty ppc_summary dict should not crash."""
        updated = update_model_card_with_results(minimal_data, ppc_summary={})
        assert "Posterior Predictive" in updated.calibration_summary

    def test_all_updates_together(self, minimal_data):
        """Test providing all update arguments at once."""
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.001
        convergence.ess_bulk_min = 3000
        convergence.ess_tail_min = 2500
        convergence.divergences = 0
        convergence.failing_params = []

        cov = MagicMock()
        cov.empirical = 0.95
        cov.interval_width = 10.0

        pm = MagicMock()
        pm.mae = 2.0
        pm.rmse = 3.0
        pm.r2 = 0.9

        loo = MagicMock()
        loo.elpd_loo = -800.0
        loo.se_elpd = 15.0

        updated = update_model_card_with_results(
            minimal_data,
            convergence=convergence,
            coverage_results={0.95: cov},
            point_metrics=pm,
            loo_result=loo,
            ppc_summary={"mean": {"observed": 70.0, "p_value": 0.5, "mc_se": 0.01}},
            prior_justification="Custom priors text.",
        )

        assert "PASSED" in updated.convergence_summary
        assert "95.0%" in updated.calibration_summary
        assert "MAE: 2.00" in updated.predictive_summary
        assert updated.loo_elpd == -800.0
        assert updated.priors_description == "Custom priors text."
