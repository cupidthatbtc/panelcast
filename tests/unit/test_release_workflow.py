import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"


def test_publish_job_is_the_only_oidc_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    build, publish = text.split("  publish:\n", maxsplit=1)

    assert "id-token: write" not in build
    assert "id-token: write" in publish
    assert "run:" not in publish
    assert "actions/checkout" not in publish
    assert "pypa/gh-action-pypi-publish" in publish


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
