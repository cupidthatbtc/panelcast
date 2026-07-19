"""Tests for DatasetDescriptor plumbing through stages and orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.errors import PipelineError
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import (
    StageContext,
    build_pipeline_stages,
    make_stage_data,
    make_stage_splits,
)


def _aero_descriptor() -> DatasetDescriptor:
    return DatasetDescriptor(
        name="aero",
        entity_col="Airframe",
        target_col="Perf_Score",
        target_bounds=(0.0, 10.0),
        model_prefix="perf",
        n_obs_col="Sensor_Samples",
        secondary_target_col=None,
        secondary_prefix=None,
        secondary_n_obs_col=None,
        processed_name_template="perf_minobs_{min_ratings}",
        raw_path_env="AERO_DATASET_PATH",
        raw_path_default="data/raw/flights.csv",
    )


class TestDefaultStagePaths:
    """Default (no descriptor) must reproduce today's exact paths."""

    def test_data_stage_outputs_unchanged(self):
        stage = make_stage_data()
        assert [str(p).replace("\\", "/") for p in stage.output_paths] == [
            "data/processed/cleaned_all.parquet",
            "data/processed/user_score_minratings_5.parquet",
            "data/processed/user_score_minratings_10.parquet",
            "data/processed/user_score_minratings_25.parquet",
            "data/processed/critic_score.parquet",
        ]

    def test_splits_stage_input_unchanged(self):
        stage = make_stage_splits(min_ratings=10)
        assert [str(p).replace("\\", "/") for p in stage.input_paths] == [
            "data/processed/user_score_minratings_10.parquet"
        ]

    def test_build_pipeline_stages_default_names(self):
        names = [s.name for s in build_pipeline_stages()]
        assert names == ["data", "splits", "features", "train", "evaluate", "predict", "report"]


class TestDescriptorDrivenPaths:
    def test_data_stage_outputs_from_descriptor(self):
        stage = make_stage_data(descriptor=_aero_descriptor())
        outputs = [str(p).replace("\\", "/") for p in stage.output_paths]
        assert "data/processed/perf_minobs_10.parquet" in outputs
        # Secondary model disabled -> no critic dataset.
        assert not any("critic" in p for p in outputs)

    def test_data_stage_input_from_descriptor_env(self, monkeypatch):
        monkeypatch.delenv("AERO_DATASET_PATH", raising=False)
        stage = make_stage_data(descriptor=_aero_descriptor())
        assert str(stage.input_paths[0]).replace("\\", "/") == "data/raw/flights.csv"
        monkeypatch.setenv("AERO_DATASET_PATH", "elsewhere/flights.csv")
        stage = make_stage_data(descriptor=_aero_descriptor())
        assert str(stage.input_paths[0]).replace("\\", "/") == "elsewhere/flights.csv"

    def test_descriptor_yaml_in_input_paths(self, tmp_path):
        yaml_path = tmp_path / "aero.yaml"
        yaml_path.write_text("name: aero\n", encoding="utf-8")
        stage = make_stage_data(descriptor=_aero_descriptor(), descriptor_path=yaml_path)
        assert yaml_path in stage.input_paths

    def test_splits_input_from_descriptor(self):
        stage = make_stage_splits(min_ratings=5, descriptor=_aero_descriptor())
        assert [str(p).replace("\\", "/") for p in stage.input_paths] == [
            "data/processed/perf_minobs_5.parquet"
        ]


class TestStageContextDescriptor:
    def test_default_descriptor_is_aoty(self):
        ctx = StageContext(
            run_dir=Path("outputs/test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.descriptor == DatasetDescriptor()


class TestResumeDescriptorGuard:
    def _orchestrator_with_manifest(self, flags: dict) -> PipelineOrchestrator:
        orch = PipelineOrchestrator(PipelineConfig())
        manifest = MagicMock()
        manifest.flags = flags
        orch.manifest = manifest
        return orch

    def test_resume_roundtrip_restores_dataset(self):
        recorded = DatasetDescriptor().descriptor_hash()
        orch = self._orchestrator_with_manifest(
            {"dataset": None, "dataset_descriptor_hash": recorded}
        )
        orch._restore_config_from_manifest()
        assert orch.config.dataset is None
        assert orch.descriptor == DatasetDescriptor()

    def test_resume_descriptor_hash_mismatch_raises(self):
        orch = self._orchestrator_with_manifest(
            {"dataset": None, "dataset_descriptor_hash": "deadbeef" * 8}
        )
        with pytest.raises(PipelineError, match="descriptor changed"):
            orch._restore_config_from_manifest()

    def test_legacy_manifest_without_hash_warns_and_defaults(self):
        orch = self._orchestrator_with_manifest({})
        orch._restore_config_from_manifest()
        assert orch.descriptor == DatasetDescriptor()

    def test_fresh_manifest_records_descriptor_hash(self, tmp_path):
        config = PipelineConfig(dry_run=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch._setup_run()
        assert orch.manifest is not None
        assert orch.manifest.flags["dataset"] is None
        assert (
            orch.manifest.flags["dataset_descriptor_hash"] == DatasetDescriptor().descriptor_hash()
        )


class TestPriorAndInitKnobPlumbing:
    """sigma_artist prior, artist-effect param, and init strategy must reach the
    StageContext and, from there, PriorConfig / MCMCConfig the way train_bayes
    consumes them (attribute-name wiring is easy to break silently)."""

    def test_context_carries_knobs(self, tmp_path):
        config = PipelineConfig(
            sigma_artist_prior_type="lognormal",
            artist_effect_param="zerosum",
            init_strategy="median",
        )
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()
        assert ctx.sigma_artist_prior_type == "lognormal"
        assert ctx.artist_effect_param == "zerosum"
        assert ctx.init_strategy == "median"

    def test_knobs_reach_prior_and_mcmc_config(self, tmp_path):
        from panelcast.models.bayes.fit import MCMCConfig
        from panelcast.models.bayes.priors import priors_for_transform

        config = PipelineConfig(
            sigma_artist_prior_type="lognormal",
            artist_effect_param="zerosum",
            init_strategy="feasible",
        )
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()

        priors = priors_for_transform(
            "identity",
            sigma_artist_prior_type=str(getattr(ctx, "sigma_artist_prior_type", "halfnormal")),
            artist_effect_param=str(getattr(ctx, "artist_effect_param", "noncentered")),
        )
        assert priors.sigma_artist_prior_type == "lognormal"
        assert priors.artist_effect_param == "zerosum"

        mcmc = MCMCConfig(init_strategy=str(getattr(ctx, "init_strategy", "uniform")))
        assert mcmc.init_strategy == "feasible"

    def test_defaults_are_byte_identical(self, tmp_path):
        from panelcast.models.bayes.fit import MCMCConfig
        from panelcast.models.bayes.priors import PriorConfig

        ctx = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)._create_stage_context()
        assert ctx.sigma_artist_prior_type == "halfnormal"
        assert ctx.artist_effect_param == "noncentered"
        assert ctx.init_strategy == "uniform"
        # Defaults match the untouched PriorConfig / MCMCConfig defaults.
        assert PriorConfig().sigma_artist_prior_type == "halfnormal"
        assert PriorConfig().artist_effect_param == "noncentered"
        assert MCMCConfig().init_strategy == "uniform"


class TestLognormalPriorParamPlumbing:
    """The sigma_rw / sigma_artist LogNormal(loc, sigma) params must reach the
    StageContext and, via priors_for_transform, the PriorConfig."""

    def test_context_carries_params(self, tmp_path):
        config = PipelineConfig(
            sigma_rw_lognormal_loc=-6.0,
            sigma_rw_lognormal_sigma=0.4,
            sigma_artist_lognormal_loc=-3.7,
            sigma_artist_lognormal_sigma=0.5,
        )
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()
        assert ctx.sigma_rw_lognormal_loc == -6.0
        assert ctx.sigma_rw_lognormal_sigma == 0.4
        assert ctx.sigma_artist_lognormal_loc == -3.7
        assert ctx.sigma_artist_lognormal_sigma == 0.5

    def test_params_reach_prior_config(self, tmp_path):
        from panelcast.models.bayes.priors import priors_for_transform

        config = PipelineConfig(
            sigma_rw_lognormal_loc=-6.0,
            sigma_rw_lognormal_sigma=0.4,
            sigma_artist_lognormal_loc=-3.7,
            sigma_artist_lognormal_sigma=0.5,
        )
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()

        priors = priors_for_transform(
            "identity",
            sigma_rw_lognormal_loc=float(getattr(ctx, "sigma_rw_lognormal_loc", -2.8)),
            sigma_rw_lognormal_sigma=float(getattr(ctx, "sigma_rw_lognormal_sigma", 0.6)),
            sigma_artist_lognormal_loc=float(getattr(ctx, "sigma_artist_lognormal_loc", -0.9)),
            sigma_artist_lognormal_sigma=float(
                getattr(ctx, "sigma_artist_lognormal_sigma", 0.6)
            ),
        )
        assert priors.sigma_rw_lognormal_loc == -6.0
        assert priors.sigma_rw_lognormal_sigma == 0.4
        assert priors.sigma_artist_lognormal_loc == -3.7
        assert priors.sigma_artist_lognormal_sigma == 0.5

    def test_defaults_are_byte_identical(self, tmp_path):
        from panelcast.models.bayes.priors import PriorConfig

        ctx = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)._create_stage_context()
        assert ctx.sigma_rw_lognormal_loc == -2.8
        assert ctx.sigma_rw_lognormal_sigma == 0.6
        assert ctx.sigma_artist_lognormal_loc == -0.9
        assert ctx.sigma_artist_lognormal_sigma == 0.6
        assert PriorConfig().sigma_rw_lognormal_loc == -2.8
        assert PriorConfig().sigma_rw_lognormal_sigma == 0.6
        assert PriorConfig().sigma_artist_lognormal_loc == -0.9
        assert PriorConfig().sigma_artist_lognormal_sigma == 0.6


class TestRhoPriorParamPlumbing:
    """The AR(1) rho_loc / rho_scale params must reach the StageContext and,
    via priors_for_transform, the PriorConfig."""

    def test_context_carries_params(self, tmp_path):
        config = PipelineConfig(rho_loc=0.2, rho_scale=0.02)
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()
        assert ctx.rho_loc == 0.2
        assert ctx.rho_scale == 0.02

    def test_params_reach_prior_config(self, tmp_path):
        from panelcast.models.bayes.priors import priors_for_transform

        config = PipelineConfig(rho_loc=0.2, rho_scale=0.02)
        ctx = PipelineOrchestrator(config, output_base=tmp_path)._create_stage_context()

        priors = priors_for_transform(
            "identity",
            rho_loc=float(getattr(ctx, "rho_loc", 0.0)),
            rho_scale=float(getattr(ctx, "rho_scale", 0.3)),
        )
        assert priors.rho_loc == 0.2
        assert priors.rho_scale == 0.02

    def test_defaults_are_byte_identical(self, tmp_path):
        from panelcast.models.bayes.priors import PriorConfig

        ctx = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)._create_stage_context()
        assert ctx.rho_loc == 0.0
        assert ctx.rho_scale == 0.3
        assert PriorConfig().rho_loc == 0.0
        assert PriorConfig().rho_scale == 0.3


class TestBetaBinomialGate:
    """The orchestrator rejects beta_binomial on a non-aggregation descriptor."""

    def _yaml(self, tmp_path, agg: bool) -> Path:
        p = tmp_path / "ds.yaml"
        p.write_text(
            f"name: ds\nn_obs_is_aggregation_count: {str(agg).lower()}\n", encoding="utf-8"
        )
        return p

    def test_gate_raises_for_non_aggregation_descriptor(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="beta_binomial",
            target_transform="identity",
            dataset=str(self._yaml(tmp_path, agg=False)),
        )
        with pytest.raises(ValueError, match="aggregation"):
            PipelineOrchestrator(config)

    def test_gate_allows_aggregation_descriptor(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="beta_binomial",
            target_transform="identity",
            dataset=str(self._yaml(tmp_path, agg=True)),
        )
        orch = PipelineOrchestrator(config)
        assert orch.descriptor.n_obs_is_aggregation_count is True

    def test_gate_ignores_other_families(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="studentt", dataset=str(self._yaml(tmp_path, agg=False))
        )
        orch = PipelineOrchestrator(config)
        assert orch.descriptor.n_obs_is_aggregation_count is False
