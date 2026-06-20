"""YAML config loading machinery.

Provides deep-merge and environment-variable expansion used by the dataset
descriptor loader and the pipeline YAML config layer. Returns plain dicts;
typed validation happens at the consumer (DatasetDescriptor, PipelineConfig
mapping).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values.

    Walks through dicts and lists, expanding $VAR and ${VAR} patterns
    in string values using os.path.expandvars.

    Note:
        This function is called automatically during config loading.
        Be cautious when loading configs from untrusted sources.
    """
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml_config(paths: str | Path | list[str | Path]) -> dict[str, Any]:
    """Load one or more YAML files into a deep-merged, env-expanded dict.

    Later files override earlier ones key-by-key (nested dicts merge
    recursively). Environment variables in string values are expanded.
    """
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)

    data: dict = {}
    for path in path_list:
        with open(path, "r", encoding="utf-8") as f:
            next_data = yaml.safe_load(f)
        if next_data is None:
            next_data = {}
        elif not isinstance(next_data, dict):
            raise ValueError(
                f"Config file {path} must be a YAML mapping, got {type(next_data).__name__}"
            )
        data = _deep_merge(data, next_data)

    return _expand_env_vars(data)
