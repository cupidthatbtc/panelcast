import tempfile
from pathlib import Path

import pytest

from panelcast.config.loader import _deep_merge, _expand_env_vars, load_yaml_config


def _write_yaml(tmp_path, filename, content):
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _minimal_yaml(**overrides):
    """Return minimal valid YAML string with optional overrides."""
    base = {
        "dataset": {"raw_csv": "data.csv"},
        "splits": {"seed": 1},
        "model": {"tune": 100, "draws": 100, "chains": 2},
    }
    # Simple override for top-level keys
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v
    import yaml

    return yaml.dump(base)


# =============================================================================
# _deep_merge tests
# =============================================================================


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_replaces_non_dict(self):
        base = {"a": 1}
        override = {"a": {"nested": True}}
        result = _deep_merge(base, override)
        assert result == {"a": {"nested": True}}

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_does_not_modify_base(self):
        base = {"a": 1}
        _deep_merge(base, {"a": 2})
        assert base["a"] == 1

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}


# =============================================================================
# _expand_env_vars tests
# =============================================================================


class TestExpandEnvVars:
    """Tests for _expand_env_vars."""

    def test_expands_dollar_var(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        result = _expand_env_vars("$MY_VAR")
        assert result == "hello"

    def test_expands_braced_var(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "world")
        result = _expand_env_vars("${MY_VAR}")
        assert result == "world"

    def test_expands_in_dict(self, monkeypatch):
        monkeypatch.setenv("DB", "postgres")
        result = _expand_env_vars({"host": "$DB"})
        assert result == {"host": "postgres"}

    def test_expands_in_list(self, monkeypatch):
        monkeypatch.setenv("ITEM", "value")
        result = _expand_env_vars(["$ITEM", "literal"])
        assert result == ["value", "literal"]

    def test_leaves_non_string_unchanged(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_nested_dict_expansion(self, monkeypatch):
        monkeypatch.setenv("VAL", "test")
        result = _expand_env_vars({"a": {"b": "$VAL"}})
        assert result == {"a": {"b": "test"}}


# =============================================================================
# load_config tests
# =============================================================================


class TestLoadYamlConfig:
    """Tests for load_yaml_config (plain-dict YAML machinery)."""

    def test_merges_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AOTY_DATASET_PATH", "env.csv")
        base = _write_yaml(
            tmp_path,
            "base.yaml",
            """
dataset:
  raw_csv: "${AOTY_DATASET_PATH}"
splits:
  seed: 1
model:
  tune: 1000
  draws: 1000
  chains: 2
""",
        )
        override = _write_yaml(
            tmp_path,
            "override.yaml",
            """
model:
  tune: 2000
""",
        )
        cfg = load_yaml_config([base, override])
        assert cfg["dataset"]["raw_csv"] == "env.csv"
        assert cfg["model"]["tune"] == 2000
        assert cfg["model"]["draws"] == 1000

    def test_single_path_as_string(self, tmp_path):
        p = _write_yaml(tmp_path, "config.yaml", _minimal_yaml())
        cfg = load_yaml_config(str(p))
        assert isinstance(cfg, dict)

    def test_single_path_as_path(self, tmp_path):
        p = _write_yaml(tmp_path, "config.yaml", _minimal_yaml())
        cfg = load_yaml_config(p)
        assert isinstance(cfg, dict)

    def test_empty_yaml_file_treated_as_empty_dict(self, tmp_path):
        base = _write_yaml(tmp_path, "base.yaml", _minimal_yaml())
        empty = _write_yaml(tmp_path, "empty.yaml", "")
        cfg = load_yaml_config([base, empty])
        assert isinstance(cfg, dict)
        assert cfg["model"]["tune"] == 100

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = _write_yaml(tmp_path, "bad.yaml", "- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_yaml_config(p)

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_CSV", "expanded.csv")
        p = _write_yaml(
            tmp_path,
            "config.yaml",
            """
dataset:
  raw_csv: "${MY_CSV}"
splits:
  seed: 1
model:
  tune: 100
  draws: 100
  chains: 2
""",
        )
        cfg = load_yaml_config(p)
        assert cfg["dataset"]["raw_csv"] == "expanded.csv"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml_config(tmp_path / "nonexistent.yaml")

    def test_multiple_overrides_last_wins(self, tmp_path):
        base = _write_yaml(tmp_path, "base.yaml", _minimal_yaml())
        over1 = _write_yaml(tmp_path, "over1.yaml", "model:\n  tune: 200\n")
        over2 = _write_yaml(tmp_path, "over2.yaml", "model:\n  tune: 300\n")
        cfg = load_yaml_config([base, over1, over2])
        assert cfg["model"]["tune"] == 300
