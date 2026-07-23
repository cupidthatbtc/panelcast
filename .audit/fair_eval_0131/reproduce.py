from __future__ import annotations

import argparse
import os
from dataclasses import fields
from pathlib import Path

import yaml

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.paths import ArtifactPaths
from panelcast.pipelines.evaluate import evaluate_models
from panelcast.pipelines.manifest import load_run_manifest
from panelcast.pipelines.stages import StageContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    args = parser.parse_args()

    source = args.source_run.resolve()
    output = args.output_run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PANELCAST_SAVE_LOG_LIKELIHOOD"] = "1"

    config = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(StageContext) if field.init}
    values = {key: value for key, value in config.items() if key in allowed}
    values.update(
        run_dir=output,
        seed=int(config["seed"]),
        strict=False,
        verbose=True,
        manifest=load_run_manifest(source / "manifest.json"),
        descriptor=DatasetDescriptor(),
        paths=ArtifactPaths(
            processed=Path("data/processed"),
            splits=Path("data/splits"),
            features=Path("data/features"),
            models=source / "models",
            evaluation=output / "evaluation",
            predictions=output / "predictions",
            reports=output / "reports",
        ),
    )
    values["calibration_intervals"] = tuple(values["calibration_intervals"])
    evaluate_models(StageContext(**values))


if __name__ == "__main__":
    main()
