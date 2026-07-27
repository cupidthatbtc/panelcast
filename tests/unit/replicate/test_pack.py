"""Domain-pack contract (#276): manifest, build gate, scaffold, collection."""

from __future__ import annotations

import pytest

from panelcast.replicate.pack import ensure_panel, load_pack, scaffold_pack

_MINIMAL_MANIFEST = (
    "name: demo-pack\n"
    "paper:\n"
    '  citation: "Somebody (1999), JASA"\n'
    "data:\n"
    '  source: "a deposit"\n'
    '  license: "CC0"\n'
)


def _write_pack(tmp_path, manifest_extra: str = "", descriptor_body: str = "name: demo\n"):
    pack_dir = tmp_path / "demo-pack"
    pack_dir.mkdir()
    (pack_dir / "pack.yaml").write_text(_MINIMAL_MANIFEST + manifest_extra, encoding="utf-8")
    (pack_dir / "descriptor.yaml").write_text(descriptor_body, encoding="utf-8")
    return pack_dir


class TestManifest:
    def test_minimal_pack_loads(self, tmp_path):
        manifest, resolved = load_pack(_write_pack(tmp_path))
        assert manifest.name == "demo-pack"
        assert resolved.name == "demo-pack"

    def test_unknown_keys_fatal(self, tmp_path):
        pack_dir = _write_pack(tmp_path, "surprise: 1\n")
        with pytest.raises(ValueError):
            load_pack(pack_dir)

    def test_missing_manifest_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a domain pack"):
            load_pack(tmp_path)

    def test_missing_descriptor_actionable(self, tmp_path):
        pack_dir = _write_pack(tmp_path)
        (pack_dir / "descriptor.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="descriptor"):
            load_pack(pack_dir)

    def test_declared_files_must_exist(self, tmp_path):
        pack_dir = _write_pack(tmp_path, "claims: claims.yaml\n")
        with pytest.raises(FileNotFoundError, match="claims"):
            load_pack(pack_dir)

    def test_run_overrides_validated_against_config_fields(self, tmp_path):
        pack_dir = _write_pack(tmp_path, "run:\n  not_a_field: 1\n")
        with pytest.raises(ValueError, match="not_a_field"):
            load_pack(pack_dir)

    def test_valid_run_override_accepted(self, tmp_path):
        pack_dir = _write_pack(tmp_path, "run:\n  min_ratings: 1\n")
        manifest, _ = load_pack(pack_dir)
        assert manifest.run == {"min_ratings": 1}


class TestEnsurePanel:
    def _descriptor(self) -> str:
        # Panel path resolves relative to the descriptor's directory.
        return (
            "name: demo\n"
            "raw_path_env: DEMO_PACK_PATH\n"
            "raw_path_default: data/panel.csv\n"
        )

    def test_missing_panel_without_build_is_actionable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = _write_pack(tmp_path, descriptor_body=self._descriptor())
        manifest, resolved = load_pack(pack_dir)
        with pytest.raises(FileNotFoundError, match="no build step"):
            ensure_panel(manifest, resolved)

    def test_build_runs_and_expected_panel_gates(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = _write_pack(tmp_path, descriptor_body=self._descriptor())
        (pack_dir / "pack.yaml").write_text(
            "name: demo-pack\n"
            "paper:\n"
            '  citation: "Somebody (1999), JASA"\n'
            "data:\n"
            '  source: "a deposit"\n'
            '  license: "CC0"\n'
            "  expected_panel: {rows: 2, entities: 2}\n"
            "build: build.py\n",
            encoding="utf-8",
        )
        (pack_dir / "build.py").write_text(
            "from pathlib import Path\n"
            "out = Path(__file__).parent / 'data' / 'panel.csv'\n"
            "out.parent.mkdir(exist_ok=True)\n"
            "out.write_text('Artist,User_Score\\nA,70\\nB,80\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        # Deliberately NOT chdir'd into the pack: the primary path (--all, or
        # replicate from the repo root) runs with CWD elsewhere, and the
        # post-build resolution must still find the pack-local panel.
        monkeypatch.chdir(tmp_path)
        manifest, resolved = load_pack(pack_dir)
        ensure_panel(manifest, resolved)  # builds, then gates clean
        assert (pack_dir / "data" / "panel.csv").exists()

    def test_expected_panel_mismatch_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = _write_pack(tmp_path, descriptor_body=self._descriptor())
        (pack_dir / "pack.yaml").write_text(
            "name: demo-pack\n"
            "paper:\n"
            '  citation: "Somebody (1999), JASA"\n'
            "data:\n"
            '  source: "a deposit"\n'
            '  license: "CC0"\n'
            "  expected_panel: {rows: 99, entities: 99}\n",
            encoding="utf-8",
        )
        data_dir = pack_dir / "data"
        data_dir.mkdir()
        (data_dir / "panel.csv").write_text("Artist,User_Score\nA,70\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        manifest, resolved = load_pack(pack_dir)
        with pytest.raises(RuntimeError, match="expected 99 / 99"):
            ensure_panel(manifest, resolved)

    def test_missing_entity_column_is_actionable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = _write_pack(tmp_path, descriptor_body=self._descriptor())
        (pack_dir / "pack.yaml").write_text(
            "name: demo-pack\n"
            "paper:\n"
            '  citation: "Somebody (1999), JASA"\n'
            "data:\n"
            '  source: "a deposit"\n'
            '  license: "CC0"\n'
            "  expected_panel: {rows: 1, entities: 1}\n",
            encoding="utf-8",
        )
        data_dir = pack_dir / "data"
        data_dir.mkdir()
        (data_dir / "panel.csv").write_text("Wrong,Cols\n1,2\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        manifest, resolved = load_pack(pack_dir)
        with pytest.raises(RuntimeError, match="no entity column"):
            ensure_panel(manifest, resolved)


class TestScaffold:
    def test_scaffold_is_a_loadable_pack(self, tmp_path):
        created = scaffold_pack("fresh-pack", tmp_path)
        manifest, _ = load_pack(created)
        assert manifest.name == "fresh-pack"
        assert (created / ".gitignore").read_text(encoding="utf-8").startswith("data/")

    def test_scaffold_refuses_existing_dir(self, tmp_path):
        scaffold_pack("fresh-pack", tmp_path)
        with pytest.raises(FileExistsError):
            scaffold_pack("fresh-pack", tmp_path)

    def test_scaffold_rejects_names_the_manifest_would(self, tmp_path):
        with pytest.raises(ValueError, match="kebab/snake"):
            scaffold_pack("MyPack", tmp_path)


class TestCliModes:
    def test_pack_new_scaffolds(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        result = CliRunner().invoke(
            app, ["pack", "new", "cli-pack", "--parent", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cli-pack" / "pack.yaml").exists()

    def test_mode_exclusivity(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        pack_dir = _write_pack(tmp_path)
        result = CliRunner().invoke(
            app, ["replicate", str(pack_dir), "--dataset", "x.yaml"]
        )
        assert result.exit_code == 2
        assert "exactly one" in result.output

    def _runnable_pack(self, tmp_path, with_claims: bool):
        pack_dir = tmp_path / "demo-pack"
        pack_dir.mkdir()
        body = _MINIMAL_MANIFEST
        if with_claims:
            body += "claims: claims.yaml\n"
            (pack_dir / "claims.yaml").write_text(
                "claims:\n"
                "  - name: peak_age\n"
                "    quantity: covariate_vertex(age_c, age_sq)\n"
                "    expect: {in: [30, 40]}\n",
                encoding="utf-8",
            )
        (pack_dir / "pack.yaml").write_text(body, encoding="utf-8")
        (pack_dir / "descriptor.yaml").write_text(
            "name: demo\n"
            "raw_path_env: DEMO_PACK_PATH\n"
            "raw_path_default: data/panel.csv\n",
            encoding="utf-8",
        )
        data_dir = pack_dir / "data"
        data_dir.mkdir()
        (data_dir / "panel.csv").write_text("Artist,User_Score\nA,70\n", encoding="utf-8")
        return pack_dir

    def test_pack_run_grades_and_writes_notes(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app

        from .test_replicate import _write_models_dir

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=True)
        run_dir = pack_dir / "outputs" / "run"
        run_dir.mkdir(parents=True)
        models_dir = _write_models_dir(run_dir)
        monkeypatch.setattr(cmd, "_run_chain_for", lambda *a, **k: models_dir.resolve())
        result = CliRunner().invoke(app, ["replicate", str(pack_dir)])
        assert result.exit_code == 0, result.output
        notes = pack_dir / "notes" / "replicate_verdicts.json"
        assert notes.exists()
        assert "PASS" in result.output

    def test_claimless_pack_runs_and_says_so(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=False)
        monkeypatch.setattr(cmd, "_run_chain_for", lambda *a, **k: tmp_path)
        result = CliRunner().invoke(app, ["replicate", str(pack_dir)])
        assert result.exit_code == 0, result.output
        assert "nothing to grade" in result.output

    @pytest.mark.parametrize(
        ("fit_value", "manifest_value", "expected"),
        [(None, None, False), (True, None, True), (True, False, False)],
    )
    def test_pack_lock_policy(
        self, tmp_path, monkeypatch, fit_value, manifest_value, expected
    ):
        from pathlib import Path

        from rich.console import Console

        import panelcast.cli.replicate_cmd as cmd

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=False)
        additions = []
        if fit_value is not None:
            (pack_dir / "fit.yaml").write_text(
                f"enforce_lockfile: {str(fit_value).lower()}\n", encoding="utf-8"
            )
            additions.append("fit: fit.yaml")
        if manifest_value is not None:
            additions.extend(
                ["run:", f"  enforce_lockfile: {str(manifest_value).lower()}"]
            )
        if additions:
            with (pack_dir / "pack.yaml").open("a", encoding="utf-8") as manifest:
                manifest.write("\n".join(additions) + "\n")
        observed = []

        class FakeOrchestrator:
            def __init__(self, config) -> None:
                observed.append(config.enforce_lockfile)
                self.run_dir = Path("outputs/run")

            def run(self) -> int:
                (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
                return 0

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.PipelineOrchestrator", FakeOrchestrator
        )
        assert cmd._run_pack(pack_dir, Console()) == []
        assert observed == [expected]

    def test_pack_pipeline_is_rooted_in_pack(self, tmp_path, monkeypatch):
        from pathlib import Path

        from rich.console import Console

        import panelcast.cli.replicate_cmd as cmd

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=False)
        caller = tmp_path / "caller"
        caller.mkdir()
        monkeypatch.chdir(caller)
        original_cwd = Path.cwd()
        observed_cwd = []

        def fake_run_chain(*args, **kwargs):
            observed_cwd.append(Path.cwd())
            Path("data").mkdir(exist_ok=True)
            Path("data/runtime.marker").write_text("pack-local", encoding="utf-8")
            models = Path("outputs/demo/models")
            models.mkdir(parents=True)
            return models

        monkeypatch.setattr(cmd, "_run_chain_for", fake_run_chain)
        assert cmd._run_pack(pack_dir, Console()) == []
        assert observed_cwd == [pack_dir.resolve()]
        assert (pack_dir / "data" / "runtime.marker").exists()
        assert (pack_dir / "outputs" / "demo" / "models").is_dir()
        assert not (caller / "data").exists()
        assert not (caller / "outputs").exists()
        assert Path.cwd() == original_cwd

    def test_pack_panel_resolution_ignores_caller_data(self, tmp_path, monkeypatch):
        from pathlib import Path

        from rich.console import Console

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.config.descriptor import load_descriptor

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        collection = tmp_path / "collection"
        collection.mkdir()
        pack_dir = self._runnable_pack(collection, with_claims=False)
        with (pack_dir / "pack.yaml").open("a", encoding="utf-8") as manifest:
            manifest.write("  expected_panel: {rows: 1, entities: 1}\n")

        caller = tmp_path / "caller"
        caller_data = caller / "data"
        caller_data.mkdir(parents=True)
        (caller_data / "panel.csv").write_text(
            "Artist,User_Score\nwrong-a,10\nwrong-b,20\n", encoding="utf-8"
        )
        monkeypatch.chdir(caller)
        observed_panels = []

        def fake_run_chain(dataset, *args, **kwargs):
            observed_panels.append(load_descriptor(dataset).resolve_raw_path().resolve())
            models = Path("outputs/demo/models")
            models.mkdir(parents=True)
            return models

        monkeypatch.setattr(cmd, "_run_chain_for", fake_run_chain)
        assert cmd._run_pack(pack_dir, Console()) == []
        assert observed_panels == [(pack_dir / "data" / "panel.csv").resolve()]
        assert Path.cwd() == caller

    def test_pack_restores_cwd_when_pipeline_raises(self, tmp_path, monkeypatch):
        from pathlib import Path

        from rich.console import Console

        import panelcast.cli.replicate_cmd as cmd

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=False)
        original_cwd = Path.cwd()

        def fail(*args, **kwargs):
            assert Path.cwd() == pack_dir.resolve()
            raise RuntimeError("pipeline failed")

        monkeypatch.setattr(cmd, "_run_chain_for", fail)
        with pytest.raises(RuntimeError, match="pipeline failed"):
            cmd._run_pack(pack_dir, Console())
        assert Path.cwd() == original_cwd

    def test_chain_returns_and_reports_absolute_models_path(self, tmp_path, monkeypatch):
        from pathlib import Path

        from rich.console import Console

        import panelcast.cli.replicate_cmd as cmd

        descriptor = tmp_path / "descriptor.yaml"
        descriptor.write_text("name: demo\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        observed = []

        class FakeOrchestrator:
            def __init__(self, config) -> None:
                observed.append(config.enforce_lockfile)
                self.run_dir = Path("outputs/run")

            def run(self) -> int:
                (self.run_dir / "models").mkdir(parents=True)
                return 0

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.PipelineOrchestrator", FakeOrchestrator
        )
        console = Console(record=True, width=300)
        models = cmd._run_chain_for(str(descriptor), console)
        expected = (tmp_path / "outputs" / "run" / "models").resolve()
        assert models == expected
        assert str(expected) in console.export_text()
        assert observed == [True]

    def test_underscore_pack_still_runs_directly(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app

        monkeypatch.delenv("DEMO_PACK_PATH", raising=False)
        pack_dir = self._runnable_pack(tmp_path, with_claims=False)
        template = pack_dir.rename(tmp_path / "_template")
        monkeypatch.setattr(cmd, "_run_chain_for", lambda *a, **k: tmp_path)
        result = CliRunner().invoke(app, ["replicate", str(template)])
        assert result.exit_code == 0, result.output
        assert "nothing to grade" in result.output

    def test_collection_scoreboard(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app
        from panelcast.replicate.evaluate import ClaimVerdict

        _write_pack(tmp_path)  # demo-pack
        other = tmp_path / "other-pack"
        other.mkdir()
        (other / "pack.yaml").write_text(
            _MINIMAL_MANIFEST.replace("demo-pack", "other-pack"), encoding="utf-8"
        )
        (other / "descriptor.yaml").write_text("name: other\n", encoding="utf-8")
        template = tmp_path / "_template"
        template.mkdir()
        (template / "pack.yaml").write_text(
            _MINIMAL_MANIFEST.replace("demo-pack", "template"), encoding="utf-8"
        )
        (template / "descriptor.yaml").write_text("name: template\n", encoding="utf-8")

        def fake_run_pack(pack_dir, console):
            verdict = "PASS" if pack_dir.name == "demo-pack" else "DIVERGENCE"
            return [
                ClaimVerdict(
                    name="c",
                    quantity="q",
                    expected="e",
                    observed="o",
                    achieved="match" if verdict == "PASS" else "qualitative",
                    target="match",
                    verdict=verdict,
                    detail="",
                )
            ]

        monkeypatch.setattr(cmd, "_run_pack", fake_run_pack)
        result = CliRunner().invoke(app, ["replicate", "--all", str(tmp_path)])
        assert result.exit_code == 1  # worst pack: divergence
        assert "demo-pack" in result.output
        assert "other-pack" in result.output
        assert "_template" not in result.output

    def test_template_only_collection_is_actionable(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        pack_dir = _write_pack(tmp_path)
        pack_dir.rename(tmp_path / "_template")
        result = CliRunner().invoke(app, ["replicate", "--all", str(tmp_path)])
        assert result.exit_code == 2
        assert "only underscore-prefixed template packs" in result.output

    def test_pack_new_bad_name_is_a_clean_error(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        result = CliRunner().invoke(
            app, ["pack", "new", "MyPack", "--parent", str(tmp_path)]
        )
        assert result.exit_code == 2
        assert "kebab/snake" in result.output

    def test_claims_rejected_with_pack_modes(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        pack_dir = _write_pack(tmp_path)
        claims = tmp_path / "c.yaml"
        claims.write_text("claims: []\n", encoding="utf-8")
        result = CliRunner().invoke(
            app, ["replicate", str(pack_dir), "--claims", str(claims)]
        )
        assert result.exit_code == 2
        assert "declare their own claims" in result.output

    def test_dataset_can_allow_unlocked_env(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app

        from .test_replicate import _write_models_dir

        models = _write_models_dir(tmp_path)
        claims = tmp_path / "claims.yaml"
        claims.write_text(
            "claims:\n"
            "  - name: peak\n"
            "    quantity: covariate_vertex(age_c, age_sq)\n"
            "    expect: {in: [30, 40]}\n",
            encoding="utf-8",
        )
        observed = []

        def fake_run_chain(dataset, console, **kwargs):
            observed.append(kwargs["overrides"])
            return models

        monkeypatch.setattr(cmd, "_run_chain_for", fake_run_chain)
        result = CliRunner().invoke(
            app,
            [
                "replicate",
                "--dataset",
                "descriptor.yaml",
                "--claims",
                str(claims),
                "--allow-unlocked-env",
            ],
        )
        assert result.exit_code == 0, result.output
        assert observed == [{"enforce_lockfile": False}]

    def test_allow_unlocked_env_rejected_without_dataset(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        models = tmp_path / "models"
        models.mkdir()
        claims = tmp_path / "claims.yaml"
        claims.write_text("claims: []\n", encoding="utf-8")
        result = CliRunner().invoke(
            app,
            [
                "replicate",
                "--models",
                str(models),
                "--claims",
                str(claims),
                "--allow-unlocked-env",
            ],
        )
        assert result.exit_code == 2
        assert "only combines with --dataset" in result.output

    def test_all_rejects_json(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        _write_pack(tmp_path)
        result = CliRunner().invoke(
            app, ["replicate", "--all", str(tmp_path), "--json", str(tmp_path / "o.json")]
        )
        assert result.exit_code == 2
        assert "per-run" in result.output

    def test_collection_surfaces_pack_crashes(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import panelcast.cli.replicate_cmd as cmd
        from panelcast.cli import app

        _write_pack(tmp_path)

        def boom(pack_dir, console):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(cmd, "_run_pack", boom)
        result = CliRunner().invoke(app, ["replicate", "--all", str(tmp_path)])
        assert result.exit_code == 2
        assert "RuntimeError: kaboom" in result.output
