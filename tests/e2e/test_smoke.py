"""Fast end-to-end smoke check for the CI PR smoke job.

Runs the full pipeline (data → … → report) on the synthetic aerospace domain at
1 chain × 100 draws in an isolated temp dir — the same shape as `panelcast
demo` — and asserts it completes and writes a model card. Marked ``smoke`` (run
by the PR smoke job via ``-m smoke``) and ``e2e`` (excluded from the fast
suite's ``not slow and not e2e`` selection).
"""

from __future__ import annotations

import pytest

from tests.e2e.test_domain_portability import (
    AERO_DESCRIPTOR,
    _run_pipeline_in,
    _write_aero_raw,
)


@pytest.mark.smoke
@pytest.mark.e2e
def test_demo_pipeline_smoke(tmp_path):
    _write_aero_raw(tmp_path)
    exit_code, orchestrator = _run_pipeline_in(
        tmp_path,
        tmp_path / "outputs",
        stages=["data", "splits", "features", "train", "evaluate", "predict", "report"],
        dataset=str(AERO_DESCRIPTOR),
        min_ratings=5,
        num_chains=1,
        num_samples=100,
        num_warmup=100,
        max_albums=10,
        min_albums_filter=2,
        allow_divergences=True,
        rhat_threshold=1.1,
        ess_threshold=100,
    )
    assert exit_code == 0, "demo smoke pipeline failed"
    run_dir = orchestrator.run_dir
    assert (run_dir / "reports" / "MODEL_CARD.md").exists(), "no model card produced"
    assert (run_dir / "evaluation" / "metrics.json").exists()
