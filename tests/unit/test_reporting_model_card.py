"""Unit tests for model card generation."""

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
def sample_model_card_data():
    """Create sample ModelCardData for testing."""
    return ModelCardData(
        model_name="Test Model",
        model_version="0.1.0",
        model_type="Test Type",
        authors=["Test Author"],
        created_date="2026-01-19",
        last_updated="2026-01-19",
        dataset_name="Test Dataset",
        dataset_size=1000,
        dataset_description="Test description",
        data_preprocessing="Test preprocessing",
        architecture_summary="Test architecture",
        priors_description="Test priors",
        hyperparameters={"param1": 1.0, "param2": "value"},
        convergence_summary="R-hat < 1.01",
        calibration_summary="95% coverage: 94%",
        predictive_summary="MAE: 5.0",
        loo_elpd=-1234.5,
        limitations=["Limitation 1", "Limitation 2"],
        ethical_considerations=["Consideration 1", "Consideration 2"],
        intended_use="Test use case",
        out_of_scope_use="Not for production",
        load_example="model = load()",
        predict_example="pred = predict()",
        interpret_example="print(pred)",
    )


class TestModelCardData:
    """Tests for ModelCardData dataclass."""

    def test_dataclass_fields_present(self, sample_model_card_data):
        """All required fields should be present."""
        data = sample_model_card_data

        # Model identity
        assert hasattr(data, "model_name")
        assert hasattr(data, "model_version")
        assert hasattr(data, "model_type")

        # Authors and dates
        assert hasattr(data, "authors")
        assert hasattr(data, "created_date")
        assert hasattr(data, "last_updated")

        # Training data
        assert hasattr(data, "dataset_name")
        assert hasattr(data, "dataset_size")
        assert hasattr(data, "dataset_description")
        assert hasattr(data, "data_preprocessing")

        # Model details
        assert hasattr(data, "architecture_summary")
        assert hasattr(data, "priors_description")
        assert hasattr(data, "hyperparameters")

        # Evaluation metrics
        assert hasattr(data, "convergence_summary")
        assert hasattr(data, "calibration_summary")
        assert hasattr(data, "predictive_summary")
        assert hasattr(data, "loo_elpd")

        # Limitations and ethics
        assert hasattr(data, "limitations")
        assert hasattr(data, "ethical_considerations")
        assert hasattr(data, "intended_use")
        assert hasattr(data, "out_of_scope_use")

        # Code examples
        assert hasattr(data, "load_example")
        assert hasattr(data, "predict_example")
        assert hasattr(data, "interpret_example")

    def test_default_values(self):
        """Optional fields should have sensible defaults."""
        data = ModelCardData(
            model_name="Test",
            model_version="0.1.0",
            model_type="Test",
            authors=["Author"],
            created_date="2026-01-19",
            last_updated="2026-01-19",
            dataset_name="Test",
            dataset_size=100,
            dataset_description="Test",
            data_preprocessing="Test",
            architecture_summary="Test",
            priors_description="Test",
        )

        # Check defaults
        assert data.hyperparameters == {}
        assert data.convergence_summary == "Not yet evaluated"
        assert data.calibration_summary == "Not yet evaluated"
        assert data.predictive_summary == "Not yet evaluated"
        assert data.loo_elpd is None
        assert data.limitations == []
        assert data.ethical_considerations == []
        assert data.intended_use == ""
        assert data.out_of_scope_use == ""
        assert data.load_example == ""
        assert data.predict_example == ""
        assert data.interpret_example == ""

    def test_create_default_returns_valid_data(self):
        """create_default_model_card_data should return valid ModelCardData."""
        data = create_default_model_card_data()

        assert isinstance(data, ModelCardData)
        assert data.model_name == "AOTY Artist Score Prediction Model"
        assert data.model_type == "Bayesian Hierarchical Regression with Time-Varying Effects"
        assert len(data.authors) > 0
        assert len(data.limitations) > 0
        assert len(data.ethical_considerations) > 0

    def test_create_default_has_code_examples(self):
        """Default model card should have code examples."""
        data = create_default_model_card_data()

        assert len(data.load_example) > 0
        assert len(data.predict_example) > 0
        assert len(data.interpret_example) > 0
        assert "load_model" in data.load_example
        assert "predict" in data.predict_example.lower()
        assert "extract_posterior_samples" in data.predict_example
        assert "posterior_samples=" in data.predict_example
        assert "pred['y']" in data.interpret_example

    def test_default_descriptor_returns_aoty_card(self):
        """Explicitly passing the default descriptor keeps the AOTY card."""
        from panelcast.config.descriptor import DatasetDescriptor

        data = create_default_model_card_data(DatasetDescriptor())
        assert data.model_name == "AOTY Artist Score Prediction Model"
        assert data.dataset_name == "Album of the Year (AOTY)"

    def test_non_default_descriptor_templates_prose(self):
        """A non-AOTY descriptor templates the card from its fields."""
        from tests.helpers.aero_data import make_aero_descriptor

        data = create_default_model_card_data(make_aero_descriptor())
        assert "Airframe" in data.model_name
        assert data.dataset_name == "aero"
        assert "Perf_Score" in data.dataset_description
        assert "[0, 10]" in data.dataset_description
        assert 'manifest.current["perf_score"]' in data.load_example
        assert 'prefix="perf_"' in data.predict_example
        # Architecture text is shared model documentation, AOTY-free prose.
        assert "Album of the Year" not in data.architecture_summary


class TestDefaultCardMatchesCurrentModel:
    """The static AOTY card must describe the shipped 0.5.0+/0.6.0+ defaults."""

    def test_architecture_describes_current_defaults(self):
        data = create_default_model_card_data()
        assert "y_ij ~ StudentT(df=4" in data.architecture_summary
        assert "y_ij ~ Normal(" not in data.architecture_summary
        assert "offset-logit transformed" in data.architecture_summary
        assert "gbm_offset" in data.architecture_summary

    def test_limitations_reflect_offset_logit_default(self):
        data = create_default_model_card_data()
        text = "\n".join(data.limitations)
        # Pre-0.5.0 claims: the transform was "HELD" and "the clip stays".
        assert "is HELD" not in text
        assert "the clip stays" not in text
        assert "default since 0.5.0" in text
        assert "no clip is applied" in text


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_markdown_format(self, sample_model_card_data):
        """Should generate valid markdown structure."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert isinstance(card, str)
        assert "# Model Card:" in card
        assert "## Model Details" in card
        assert "## Intended Use" in card

    def test_latex_format(self, sample_model_card_data):
        """Should generate valid LaTeX structure."""
        card = generate_model_card(sample_model_card_data, format="latex")

        assert isinstance(card, str)
        assert "\\documentclass{article}" in card
        assert "\\begin{document}" in card
        assert "\\end{document}" in card
        assert "\\section*{Model Card:" in card

    def test_invalid_format_raises(self, sample_model_card_data):
        """Should raise ValueError for invalid format."""
        with pytest.raises(ValueError, match="format must be"):
            generate_model_card(sample_model_card_data, format="html")

    def test_contains_model_name(self, sample_model_card_data):
        """Model name should appear in output."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert sample_model_card_data.model_name in card

    def test_contains_all_sections(self, sample_model_card_data):
        """All major sections should be present in markdown."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        expected_sections = [
            "## Model Details",
            "## Intended Use",
            "## Training Data",
            "## Model Architecture",
            "### Prior Distributions",
            "### Hyperparameters",
            "## Evaluation Results",
            "### Convergence Diagnostics",
            "### Calibration",
            "### Predictive Performance",
            "## Limitations",
            "## Ethical Considerations",
            "## How to Use",
            "### Loading the Model",
            "### Making Predictions",
            "### Interpreting Results",
        ]

        for section in expected_sections:
            assert section in card, f"Missing section: {section}"

    def test_code_blocks_formatted(self, sample_model_card_data):
        """Code examples should have proper fencing."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        # Should have python code blocks
        assert "```python" in card
        assert "```" in card

        # Code examples should be included
        assert sample_model_card_data.load_example in card
        assert sample_model_card_data.predict_example in card
        assert sample_model_card_data.interpret_example in card

    def test_hyperparameters_table(self, sample_model_card_data):
        """Hyperparameters should be formatted as table in markdown."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert "| Parameter | Value |" in card
        assert "|-----------|-------|" in card
        assert "param1" in card
        assert "1.0" in card

    def test_latex_code_listing(self, sample_model_card_data):
        """LaTeX output should use lstlisting for code."""
        card = generate_model_card(sample_model_card_data, format="latex")

        assert "\\begin{lstlisting}" in card
        assert "\\end{lstlisting}" in card

    def test_latex_escapes_special_chars(self):
        """LaTeX generator should escape special characters."""
        data = ModelCardData(
            model_name="Test_Model & Co.",
            model_version="0.1.0",
            model_type="Test Type with $special% chars",
            authors=["Author #1"],
            created_date="2026-01-19",
            last_updated="2026-01-19",
            dataset_name="Test",
            dataset_size=100,
            dataset_description="Test with 100% coverage",
            data_preprocessing="Test",
            architecture_summary="Test",
            priors_description="Test",
        )

        card = generate_model_card(data, format="latex")

        # Special characters should be escaped
        assert "\\_" in card  # underscore
        assert "\\&" in card  # ampersand
        assert "\\$" in card  # dollar
        assert "\\%" in card  # percent
        assert "\\#" in card  # hash


class TestWriteModelCard:
    """Tests for write_model_card function."""

    def test_creates_markdown_file(self, sample_model_card_data):
        """Should create .md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path, formats=("md",))

            assert len(paths) == 1
            assert paths[0].suffix == ".md"
            assert paths[0].exists()

    def test_creates_latex_file(self, sample_model_card_data):
        """Should create .tex file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path, formats=("tex",))

            assert len(paths) == 1
            assert paths[0].suffix == ".tex"
            assert paths[0].exists()

    def test_creates_both_formats(self, sample_model_card_data):
        """Should create both .md and .tex files by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert len(paths) == 2
            suffixes = {p.suffix for p in paths}
            assert ".md" in suffixes
            assert ".tex" in suffixes

    def test_returns_paths(self, sample_model_card_data):
        """Should return list of created paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert isinstance(paths, list)
            assert all(isinstance(p, Path) for p in paths)
            assert all(p.exists() for p in paths)

    def test_creates_parent_directories(self, sample_model_card_data):
        """Should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deep" / "nested" / "path" / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert all(p.exists() for p in paths)

    def test_unsupported_format_raises(self, sample_model_card_data):
        """Should raise ValueError for unsupported format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            with pytest.raises(ValueError, match="Unsupported format"):
                write_model_card(sample_model_card_data, output_path, formats=("html",))


class TestUpdateWithResults:
    """Tests for update_model_card_with_results function."""

    def test_fills_convergence(self, sample_model_card_data):
        """Should populate convergence summary from diagnostics."""
        # Create mock convergence diagnostics
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        assert "PASSED" in updated.convergence_summary
        assert "1.0050" in updated.convergence_summary
        assert "2,000" in updated.convergence_summary

    def test_fills_convergence_failed(self, sample_model_card_data):
        """Should show FAILED status when convergence fails."""
        convergence = MagicMock()
        convergence.passed = False
        convergence.rhat_max = 1.02
        convergence.ess_bulk_min = 500
        convergence.ess_tail_min = 400
        convergence.divergences = 10
        convergence.failing_params = ["param1", "param2"]

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        assert "FAILED" in updated.convergence_summary
        assert "param1" in updated.convergence_summary

    def test_missing_ess_tail_renders_unavailable(self, sample_model_card_data):
        """A None ess_tail_min must print 'unavailable', never the bulk value."""
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2100
        convergence.ess_tail_min = None
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        assert "ESS tail (min): unavailable" in updated.convergence_summary
        assert "ESS tail (min): 2,100" not in updated.convergence_summary
        assert "ESS bulk (min): 2,100" in updated.convergence_summary

    def test_failing_params_line_rendered(self, sample_model_card_data):
        """failing_params from the diagnostics payload should render."""
        convergence = MagicMock()
        convergence.passed = False
        convergence.rhat_max = 1.02
        convergence.ess_bulk_min = 300
        convergence.ess_tail_min = 250
        convergence.divergences = 0
        convergence.failing_params = ["user_sigma_artist", "user_rho"]

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        assert "Failing parameters: user_sigma_artist, user_rho" in updated.convergence_summary

    def test_fills_calibration(self, sample_model_card_data):
        """Should populate calibration summary from coverage results."""
        # Create mock coverage results
        coverage_50 = MagicMock()
        coverage_50.empirical = 0.48
        coverage_95 = MagicMock()
        coverage_95.empirical = 0.94

        coverage_results = {0.50: coverage_50, 0.95: coverage_95}

        updated = update_model_card_with_results(
            sample_model_card_data, coverage_results=coverage_results
        )

        assert "50% CI:" in updated.calibration_summary
        assert "95% CI:" in updated.calibration_summary
        assert "48.0%" in updated.calibration_summary
        assert "94.0%" in updated.calibration_summary

    def test_fills_elpd(self, sample_model_card_data):
        """Should fill LOO ELPD value from LOOResult."""
        loo_result = MagicMock()
        loo_result.elpd_loo = -5432.1
        loo_result.se_elpd = 123.4

        updated = update_model_card_with_results(sample_model_card_data, loo_result=loo_result)

        assert updated.loo_elpd == -5432.1
        assert "-5432.1" in updated.predictive_summary
        assert "123.4" in updated.predictive_summary

    def test_fills_point_metrics(self, sample_model_card_data):
        """Should fill point metrics summary."""
        point_metrics = MagicMock()
        point_metrics.mae = 4.56
        point_metrics.rmse = 6.78
        point_metrics.r2 = 0.789

        updated = update_model_card_with_results(
            sample_model_card_data, point_metrics=point_metrics
        )

        assert "MAE: 4.56" in updated.predictive_summary
        assert "RMSE: 6.78" in updated.predictive_summary
        assert "R-squared: 0.789" in updated.predictive_summary

    def test_preserves_other_fields(self, sample_model_card_data):
        """Should preserve fields not being updated."""
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        # Should preserve non-updated fields
        assert updated.model_name == sample_model_card_data.model_name
        assert updated.limitations == sample_model_card_data.limitations
        assert updated.load_example == sample_model_card_data.load_example

    def test_updates_last_updated_date(self, sample_model_card_data):
        """Should update last_updated to today's date."""
        from datetime import date

        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(sample_model_card_data, convergence=convergence)

        assert updated.last_updated == date.today().isoformat()

    def test_no_updates_returns_copy(self, sample_model_card_data):
        """Calling with no updates should return data with only last_updated changed."""
        updated = update_model_card_with_results(sample_model_card_data)

        # Should be a different object
        assert updated is not sample_model_card_data
        # But with same content (except last_updated)
        assert updated.model_name == sample_model_card_data.model_name
        assert updated.convergence_summary == sample_model_card_data.convergence_summary


class TestPPCAndPriorJustification:
    """Tests for PPC summary and prior justification in model card."""

    def test_ppc_summary_appended_to_calibration(self, sample_model_card_data):
        """PPC summary dict should appear in calibration summary."""
        ppc_summary = {
            "mean": {"observed": 72.3, "p_value": 0.45, "mc_se": 0.015},
            "sd": {"observed": 8.1, "p_value": 0.62, "mc_se": 0.014},
        }
        updated = update_model_card_with_results(sample_model_card_data, ppc_summary=ppc_summary)
        assert "Posterior Predictive Checks" in updated.calibration_summary
        assert "mean" in updated.calibration_summary
        assert "72.30" in updated.calibration_summary
        assert "0.450" in updated.calibration_summary

    def test_prior_justification_sets_priors_description(self, sample_model_card_data):
        """prior_justification should replace priors_description."""
        justification = "Custom prior justification text with actual values."
        updated = update_model_card_with_results(
            sample_model_card_data, prior_justification=justification
        )
        assert updated.priors_description == justification

    def test_no_ppc_preserves_calibration(self, sample_model_card_data):
        """When ppc_summary is None, calibration summary should be unchanged."""
        updated = update_model_card_with_results(sample_model_card_data)
        assert "Posterior Predictive" not in updated.calibration_summary

    def test_no_prior_justification_preserves_priors(self, sample_model_card_data):
        """When prior_justification is None, priors_description unchanged."""
        updated = update_model_card_with_results(sample_model_card_data)
        assert updated.priors_description == sample_model_card_data.priors_description

    def test_coverage_with_interval_width(self, sample_model_card_data):
        """Coverage results with interval_width should include width in summary."""
        coverage_95 = MagicMock()
        coverage_95.empirical = 0.94
        coverage_95.interval_width = 15.3
        coverage_results = {0.95: coverage_95}

        updated = update_model_card_with_results(
            sample_model_card_data, coverage_results=coverage_results
        )
        assert "mean width=15.30" in updated.calibration_summary

    def test_coverage_without_interval_width(self, sample_model_card_data):
        """Coverage results without interval_width should still work."""
        coverage_95 = MagicMock(spec=["empirical"])
        coverage_95.empirical = 0.94
        coverage_results = {0.95: coverage_95}

        updated = update_model_card_with_results(
            sample_model_card_data, coverage_results=coverage_results
        )
        assert "94.0%" in updated.calibration_summary
        assert "mean width" not in updated.calibration_summary


class TestLatexEscape:
    """Tests for _latex_escape helper."""

    def test_latex_escapes_backslash(self):
        """Backslash should become \\textbackslash{} without corrupted braces."""
        assert _latex_escape("a\\b") == "a\\textbackslash{}b"

    def test_latex_escapes_braces(self):
        """Curly braces should be escaped."""
        assert _latex_escape("{test}") == "\\{test\\}"

    def test_latex_escapes_backslash_and_braces(self):
        """Backslash followed by braces should not double-escape."""
        result = _latex_escape("\\{x}")
        assert result == "\\textbackslash{}\\{x\\}"

    def test_latex_escapes_common_specials(self):
        """Common LaTeX specials should all be escaped."""
        assert _latex_escape("a_b") == "a\\_b"
        assert _latex_escape("a&b") == "a\\&b"
        assert _latex_escape("100%") == "100\\%"
        assert _latex_escape("$x$") == "\\$x\\$"
        assert _latex_escape("#1") == "\\#1"

    def test_latex_escapes_plain_text(self):
        """Plain text without specials should pass through unchanged."""
        assert _latex_escape("hello world") == "hello world"


# --- from unit/test_reporting_model_card_expanded.py ---


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
        assert "Held-out ELPD (test lppd):" in card
        assert "-1234.5" in card

    def test_markdown_no_elpd(self, minimal_data):
        card = generate_model_card(minimal_data, format="markdown")
        assert "Held-out ELPD (test lppd):" not in card

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


class TestFlaggedSlicesSection:
    """Flagged calibration slices table in the model card (#181)."""

    def _data(self, **overrides):
        from panelcast.reporting.model_card import create_default_model_card_data

        data = create_default_model_card_data()
        for key, value in overrides.items():
            setattr(data, key, value)
        return data

    def test_flagged_table_rendered(self):
        from panelcast.reporting.model_card import generate_model_card

        data = self._data(
            flagged_slices=[
                {
                    "dimension": "group",
                    "label": "pop",
                    "n": 45,
                    "levels": {
                        "0.95": {
                            "nominal": 0.95, "empirical": 0.82,
                            "wilson_lo": 0.68, "wilson_hi": 0.90,
                            "mean_interval_width": 20.0, "flagged": True,
                        },
                    },
                },
            ],
            expected_false_flags=1.2,
        )
        card = generate_model_card(data, format="markdown")
        assert "Flagged calibration slices" in card
        assert "| group | pop | 45 | 0.95 | 0.820 | [0.680, 0.900] |" in card
        assert "~1.2" in card

    def test_section_absent_without_flags(self):
        from panelcast.reporting.model_card import generate_model_card

        card = generate_model_card(self._data(), format="markdown")
        assert "Flagged calibration slices" not in card


class TestRankingSummaryLine:
    """Ranking one-liner in Predictive Performance (#182)."""

    def test_rendered_when_set(self):
        from panelcast.reporting.model_card import (
            create_default_model_card_data,
            generate_model_card,
        )

        data = create_default_model_card_data()
        data.ranking_summary = "Spearman 0.712, precision@10 0.60 (single-slate, descriptive)."
        card = generate_model_card(data, format="markdown")
        assert "**Ranking:** Spearman 0.712" in card

    def test_absent_by_default(self):
        from panelcast.reporting.model_card import (
            create_default_model_card_data,
            generate_model_card,
        )

        card = generate_model_card(create_default_model_card_data(), format="markdown")
        assert "**Ranking:**" not in card
