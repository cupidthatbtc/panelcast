import re
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"


def test_publish_job_is_the_only_oidc_boundary() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    oidc_jobs = {
        name
        for name, job in jobs.items()
        if (job.get("permissions") or {}).get("id-token") == "write"
    }
    publish = jobs["publish"]
    steps = publish["steps"]

    assert oidc_jobs == {"publish"}
    assert all("run" not in step for step in steps)
    assert all("actions/checkout" not in str(step.get("uses", "")) for step in steps)
    assert any("pypa/gh-action-pypi-publish" in str(step.get("uses", "")) for step in steps)


def test_release_fails_closed_on_tag_version_mismatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "cache: pip" not in text
    assert 'tag_version="${GITHUB_REF_NAME#v}"' in text
    assert 'importlib.metadata.version("panelcast")' in text
    assert '[[ "$tag_version" != "$wheel_version" ]]' in text


def test_release_actions_are_commit_pinned() -> None:
    actions = re.findall(r"uses: [^@\n]+@([^\s]+)", WORKFLOW.read_text(encoding="utf-8"))

    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in actions)
