"""Descriptor-owned model facts: likelihood, transform, event cap (#268)."""

from __future__ import annotations

import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator


def _write_descriptor(tmp_path, body: str):
    path = tmp_path / "domain.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class TestDescriptorFields:
    def test_defaults_are_unset(self):
        d = DatasetDescriptor()
        assert d.likelihood_family is None
        assert d.target_transform is None
        assert d.max_events is None

    def test_hash_stable_when_facts_unset(self):
        # Pre-#268 descriptors must keep their recorded hashes.
        assert (
            DatasetDescriptor().descriptor_hash()
            == "a9e3e20540b1dcb5d6253bd342cff6fd73ed823597428f4e94abd51f8b67b8ec"
        )

    def test_hash_changes_when_facts_declared(self):
        base = DatasetDescriptor().descriptor_hash()
        assert DatasetDescriptor(likelihood_family="normal").descriptor_hash() != base
        assert DatasetDescriptor(target_transform="identity").descriptor_hash() != base
        assert DatasetDescriptor(max_events=60).descriptor_hash() != base


class TestConfigSentinels:
    def test_unset_config_carries_none(self):
        config = PipelineConfig()
        assert config.likelihood_family is None
        assert config.target_transform is None
        assert config.max_albums is None

    def test_explicit_values_validate_immediately(self):
        with pytest.raises(ValueError, match="likelihood_family"):
            PipelineConfig(likelihood_family="nope")
        with pytest.raises(ValueError, match="target_transform"):
            PipelineConfig(target_transform="nope")
        with pytest.raises(ValueError, match="max_albums"):
            PipelineConfig(max_albums=0)

    def test_family_without_transform_defers_coupling(self):
        # The identity requirement is checked after resolution, not on the
        # half-resolved sentinel state.
        config = PipelineConfig(likelihood_family="beta_binomial")
        assert config.target_transform is None


class TestOrchestratorResolution:
    def test_pipeline_defaults_when_descriptor_silent(self, tmp_path):
        config = PipelineConfig()
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.likelihood_family == "studentt"
        assert config.target_transform == "offset_logit"
        assert config.max_albums == 50

    def test_descriptor_owns_the_default_run(self, tmp_path):
        dataset = _write_descriptor(
            tmp_path,
            "name: facts\n"
            "likelihood_family: beta_binomial\n"
            "target_transform: identity\n"
            "max_events: 60\n",
        )
        config = PipelineConfig(dataset=dataset)
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.likelihood_family == "beta_binomial"
        assert config.target_transform == "identity"
        assert config.max_albums == 60

    def test_explicit_config_beats_the_descriptor(self, tmp_path):
        dataset = _write_descriptor(
            tmp_path,
            "name: facts\n"
            "likelihood_family: beta_binomial\n"
            "target_transform: identity\n"
            "max_events: 60\n",
        )
        config = PipelineConfig(
            dataset=dataset,
            likelihood_family="studentt",
            target_transform="offset_logit",
            max_albums=25,
        )
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.likelihood_family == "studentt"
        assert config.target_transform == "offset_logit"
        assert config.max_albums == 25

    def test_incoherent_descriptor_facts_fail_at_resolution(self, tmp_path):
        dataset = _write_descriptor(
            tmp_path,
            "name: facts\n"
            "likelihood_family: beta_binomial\n"
            "target_transform: offset_logit\n",
        )
        with pytest.raises(ValueError, match="identity"):
            PipelineOrchestrator(PipelineConfig(dataset=dataset), output_base=tmp_path)

    def test_unknown_descriptor_family_fails_at_resolution(self, tmp_path):
        dataset = _write_descriptor(tmp_path, "name: facts\nlikelihood_family: nope\n")
        with pytest.raises(ValueError, match="likelihood_family"):
            PipelineOrchestrator(PipelineConfig(dataset=dataset), output_base=tmp_path)


class TestSharedResolutionHelper:
    def test_resolves_and_is_idempotent(self, tmp_path):
        from panelcast.config.descriptor import load_descriptor
        from panelcast.pipelines.orchestrator import resolve_model_facts

        dataset = _write_descriptor(
            tmp_path,
            "name: facts\n"
            "likelihood_family: beta_binomial\n"
            "target_transform: identity\n"
            "max_events: 60\n",
        )
        config = PipelineConfig(dataset=dataset)
        descriptor = load_descriptor(dataset)
        resolve_model_facts(config, descriptor)
        resolved = (config.likelihood_family, config.target_transform, config.max_albums)
        assert resolved == ("beta_binomial", "identity", 60)
        resolve_model_facts(config, descriptor)
        assert (config.likelihood_family, config.target_transform, config.max_albums) == resolved

    def test_rejects_beta_binomial_without_aggregation_counts(self, tmp_path):
        from panelcast.config.descriptor import load_descriptor
        from panelcast.pipelines.orchestrator import resolve_model_facts

        dataset = _write_descriptor(
            tmp_path, "name: facts\nn_obs_is_aggregation_count: false\n"
        )
        config = PipelineConfig(
            dataset=dataset, likelihood_family="beta_binomial", target_transform="identity"
        )
        with pytest.raises(ValueError, match="aggregation"):
            resolve_model_facts(config, load_descriptor(dataset))

    def test_quick_preflight_resolves_descriptor_facts(self, tmp_path, monkeypatch):
        # Regression: the CLI preflights run before the orchestrator exists,
        # so they must resolve the sentinels themselves or max_seq sees None.
        from types import SimpleNamespace

        import panelcast.data.ingest as ingest_mod
        import panelcast.preflight as preflight_mod
        from panelcast.cli.run import _run_quick_preflight
        from panelcast.config.descriptor import DatasetDescriptor

        dataset = _write_descriptor(
            tmp_path,
            "name: facts\n"
            "likelihood_family: beta_binomial\n"
            "target_transform: identity\n"
            "max_events: 60\n",
        )
        captured: dict = {}

        def fake_check(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(status=preflight_mod.PreflightStatus.PASS, exit_code=0)

        monkeypatch.setattr(preflight_mod, "run_preflight_check", fake_check)
        monkeypatch.setattr(preflight_mod, "render_preflight_result", lambda *a, **k: None)
        monkeypatch.setattr(
            ingest_mod,
            "extract_data_dimensions",
            lambda **k: SimpleNamespace(n_observations=10, n_artists=3),
        )
        monkeypatch.setattr(
            DatasetDescriptor, "resolve_raw_path", lambda self: tmp_path / "raw.csv"
        )
        config = PipelineConfig(dataset=dataset)
        _run_quick_preflight(config, preflight_only=False, force_run=False)
        assert captured["max_seq"] == 60
        assert config.likelihood_family == "beta_binomial"
        assert config.target_transform == "identity"


class TestRescaleTargetToUnit:
    def test_normalizes_bounds_and_keeps_raw_readable(self):
        d = DatasetDescriptor(rescale_target_to_unit=True)
        assert tuple(d.target_bounds) == (0.0, 1.0)
        assert d.raw_target_bounds == (0.0, 100.0)

    def test_noop_when_bounds_already_unit(self):
        d = DatasetDescriptor(
            rescale_target_to_unit=True,
            target_bounds=(0.0, 1.0),
        )
        assert tuple(d.target_bounds) == (0.0, 1.0)
        assert d.raw_target_bounds == (0.0, 1.0)

    def test_requires_aggregation_count(self):
        with pytest.raises(ValueError, match="aggregation"):
            DatasetDescriptor(
                rescale_target_to_unit=True,
                n_obs_is_aggregation_count=False,
                secondary_target_col=None,
                secondary_prefix=None,
                secondary_n_obs_col=None,
            )

    def test_hash_stable_when_flag_off_and_changes_when_on(self):
        base = DatasetDescriptor().descriptor_hash()
        assert DatasetDescriptor(rescale_target_to_unit=False).descriptor_hash() == base
        assert DatasetDescriptor(rescale_target_to_unit=True).descriptor_hash() != base

    def test_hash_distinguishes_raw_spans_under_rescale(self):
        # Both normalize to (0, 1) but prepare different data; resume/skip
        # must never treat them as the same experiment.
        a = DatasetDescriptor(rescale_target_to_unit=True, target_bounds=(0.0, 100.0))
        b = DatasetDescriptor(rescale_target_to_unit=True, target_bounds=(20.0, 80.0))
        assert tuple(a.target_bounds) == tuple(b.target_bounds) == (0.0, 1.0)
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_prepare_rescales_target_but_not_counts(self, tmp_path, monkeypatch):
        import pandas as pd

        from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets

        csv = tmp_path / "raw.csv"
        pd.DataFrame(
            {
                "State": ["AK", "AK", "WY", "WY", "CA", "CA"],
                "Election_ID": [f"E{i}" for i in range(6)],
                "Year": [2018, 2020, 2018, 2020, 2018, 2020],
                "Election_Date": ["2018-11-06", "2020-11-03"] * 3,
                # Percent scale: a true proportion on a non-unit span.
                "Dem_Share_Pct": [24.2, 41.0, 62.2, 55.0, 70.1, 66.0],
                "Two_Party_Votes": [1000, 1200, 900, 950, 5000, 5200],
            }
        ).to_csv(csv, index=False)
        descriptor = DatasetDescriptor(
            name="pct",
            raw_path_env="PCT_DATASET_PATH",
            encoding="utf-8",
            raw_column_map={},
            required_raw_columns=[
                "State", "Election_ID", "Year", "Election_Date",
                "Dem_Share_Pct", "Two_Party_Votes",
            ],
            optional_raw_columns=[],
            entity_col="State",
            event_col="Election_ID",
            entity_group_col=None,
            date_col="Election_Date",
            parsed_date_col="Election_Date_Parsed",
            date_format="%Y-%m-%d",
            target_col="Dem_Share_Pct",
            target_bounds=(0.0, 100.0),
            model_prefix="dem",
            n_obs_col="Two_Party_Votes",
            n_obs_is_aggregation_count=True,
            rescale_target_to_unit=True,
            secondary_target_col=None,
            secondary_prefix=None,
            secondary_n_obs_col=None,
            multi_entity_col=None,
            unknown_entity_sentinel=None,
            min_year=1950,
            min_obs_thresholds=[1],
            primary_min_obs=1,
            processed_name_template="pct_{min_ratings}",
            feature_packs=[],
        )
        monkeypatch.chdir(tmp_path)
        result = prepare_datasets(
            PrepareConfig(
                raw_path=str(csv),
                output_dir="data/processed",
                audit_dir="data/audit",
                descriptor=descriptor,
            )
        )
        processed = pd.read_parquet(tmp_path / "data" / "processed" / "pct_1.parquet")
        assert processed["Dem_Share_Pct"].between(0.0, 1.0).all()
        assert processed["Dem_Share_Pct"].max() <= 0.71  # 70.1% -> 0.701
        assert processed["Two_Party_Votes"].max() == 5200  # counts untouched
        assert result is not None
