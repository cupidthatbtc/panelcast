"""Isolation policy for the credential-bearing Claude review job (#371).

The reviewer runs with a long-lived Claude token, a minted GitHub App token, and
OIDC. A collaborator branch controls everything in the checkout, so the policy
below is what keeps that branch from getting its own code executed next to those
credentials. The mutation tests exist because a rule nobody can fail is not a
rule.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
SYNTHETIC_ARTIFACT = "claude-safe-workspace-${{ github.run_id }}"
SYNTHETIC_BUILDER = "Build a constant synthetic action workspace"
SYNTHETIC_UPLOAD = "Upload the synthetic action workspace"
SYNTHETIC_DOWNLOAD = "Download the synthetic action workspace"

# The CI MCP server is action-owned and receives a separate read-only workflow
# token. The credentialed reviewer gets no filesystem, shell, or network tool.
READ_ONLY_TOOLS = frozenset(
    {
        "mcp__review_diff__get_diff",
        "mcp__github_ci__get_ci_status",
        "mcp__github_ci__get_workflow_run_details",
    }
)

# Denials that have to survive the union with the tools the action injects for
# its own tag mode, which include workspace writes and git add/commit/rm.
REQUIRED_DENIALS = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "Write",
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "Bash",
        "WebFetch",
        "WebSearch",
    }
)

WRITE_SCOPES_ALLOWED_WITH_CREDENTIALS = frozenset({"pull-requests", "id-token"})

# Triggers that would run this workflow from the base branch, or on a payload a
# fork can write, while the review credentials are in scope.
UNSAFE_TRIGGERS = frozenset(
    {"pull_request_target", "workflow_run", "issue_comment", "pull_request_review_comment"}
)

TOOLCHAIN_ACTIONS = ("prefix-dev/setup-pixi", "actions/setup-python", "actions/setup-node")

REVIEW_ACTION = "anthropics/claude-code-action"

PINNED = re.compile(r"[^@\s]+@[0-9a-f]{40}")


def load_workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow.get("jobs", {})


def triggers(workflow: dict[str, Any]) -> Any:
    # PyYAML resolves a bare `on:` key to the boolean True.
    return workflow.get("on", workflow.get(True))


def _text(node: Any) -> str:
    return yaml.safe_dump(node, sort_keys=True, default_flow_style=False)


def _effective_permissions(
    workflow: dict[str, Any], job: dict[str, Any]
) -> dict[str, str] | None:
    raw = job["permissions"] if "permissions" in job else workflow.get("permissions")
    return raw if isinstance(raw, dict) else None


def holds_credentials(job: dict[str, Any], workflow: dict[str, Any] | None = None) -> bool:
    workflow = workflow or {}
    permissions = _effective_permissions(workflow, job) or {}
    inherited = {"env": workflow.get("env"), "job": job}
    return "secrets." in _text(inherited) or permissions.get("id-token") == "write"


def _executes_pull_request_code(job: dict[str, Any]) -> bool:
    if "uses" in job:
        return True
    return any(
        "run" in step or str(step.get("uses", "")).startswith("./")
        for step in _steps(job)
    )


def _workflow_paths() -> list[Path]:
    return sorted({*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")})


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps") or []


def _review_action_inputs(job: dict[str, Any]) -> dict[str, Any]:
    for step in _steps(job):
        if str(step.get("uses", "")).startswith(REVIEW_ACTION):
            return step.get("with") or {}
    return {}


def _tools(claude_args: str, flag: str) -> list[str]:
    # The action accumulates repeated tool flags, so every occurrence counts.
    values = re.findall(rf'{re.escape(flag)}\s+"([^"]*)"', claude_args)
    return [tool.strip() for value in values for tool in value.split(",") if tool.strip()]


def _setting_sources(claude_args: str) -> list[set[str]]:
    return [
        {source.strip() for source in value.split(",")}
        for value in re.findall(r"--setting-sources\s+(\S+)", claude_args)
    ]


def _job_violations(job: dict[str, Any]) -> Iterator[str]:
    if any("run" in step for step in _steps(job)):
        yield "runs a shell step alongside the review credentials"

    downloads = [
        step
        for step in _steps(job)
        if str(step.get("uses", "")).startswith("actions/download-artifact")
    ]
    if len(downloads) != 1:
        yield "does not have exactly one synthetic artifact download"
    else:
        download = downloads[0]
        if download.get("name") != SYNTHETIC_DOWNLOAD:
            yield "renames the synthetic artifact download step"
        if (download.get("with") or {}).get("name") != SYNTHETIC_ARTIFACT:
            yield "downloads an artifact other than the synthetic workspace"

    if "head.repo.full_name == github.repository" not in str(job.get("if", "")):
        yield "does not restrict itself to same-repository pull requests"

    for step in _steps(job):
        uses = str(step.get("uses", ""))
        if not uses:
            continue
        if not PINNED.fullmatch(uses):
            yield f"uses an action that is not commit-pinned: {uses}"
        if uses.startswith("actions/checkout"):
            yield "checks out pull-request code alongside review credentials"
        if uses.startswith(TOOLCHAIN_ACTIONS):
            yield f"installs a project toolchain: {uses}"

    for scope, level in (job.get("permissions") or {}).items():
        if level == "write" and scope not in WRITE_SCOPES_ALLOWED_WITH_CREDENTIALS:
            yield f"requests {scope}: write"

    if (job.get("env") or {}).get("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB") != "1":
        yield "does not enable subprocess environment scrubbing"

    inputs = _review_action_inputs(job)
    claude_args = str(inputs.get("claude_args", ""))

    sources = _setting_sources(claude_args)
    if not sources or any(source != {"user"} for source in sources):
        yield "loads in-repo Claude settings from the pull request checkout"

    allowed = set(_tools(claude_args, "--allowedTools"))
    if not allowed:
        yield "omits the allowlist, so the pinned action enables its default tools"
    for tool in sorted(allowed - READ_ONLY_TOOLS):
        yield f"allows a tool that is not read-only: {tool}"
    for tool in sorted(READ_ONLY_TOOLS - allowed):
        yield f"omits required review tool: {tool}"

    for tool in sorted(REQUIRED_DENIALS - set(_tools(claude_args, "--disallowedTools"))):
        yield f"does not deny {tool}"

    if inputs.get("plugins") or inputs.get("plugin_marketplaces"):
        yield "installs mutable plugin code alongside the review credentials"

    if inputs.get("allowed_non_write_users") != "${{ github.repository_owner }}":
        yield "allows sandbox-bootstrap actors beyond the repository owner"

    if "actions: read" not in str(inputs.get("additional_permissions", "")):
        yield "does not enable the read-only GitHub CI MCP server"


def _synthetic_workspace_violations(workflow: dict[str, Any]) -> list[str]:
    input_steps = _steps(jobs(workflow).get("review-input") or {})
    review_steps = _steps(jobs(workflow).get("review") or {})
    uploads = [
        step
        for step in input_steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    downloads = [
        step
        for step in review_steps
        if str(step.get("uses", "")).startswith("actions/download-artifact")
    ]
    builders = [step for step in input_steps if step.get("name") == SYNTHETIC_BUILDER]
    findings: list[str] = []
    if len(uploads) != 1 or len(downloads) != 1 or len(builders) != 1:
        return ["synthetic workspace requires exactly one builder, upload, and download"]

    upload, download, builder = uploads[0], downloads[0], builders[0]
    if upload.get("name") != SYNTHETIC_UPLOAD:
        findings.append("synthetic workspace upload step has the wrong name")
    if download.get("name") != SYNTHETIC_DOWNLOAD:
        findings.append("synthetic workspace download step has the wrong name")
    upload_with = upload.get("with") or {}
    download_with = download.get("with") or {}
    if upload_with.get("name") != SYNTHETIC_ARTIFACT:
        findings.append("synthetic workspace upload uses the wrong artifact name")
    if download_with.get("name") != SYNTHETIC_ARTIFACT:
        findings.append("synthetic workspace download uses the wrong artifact name")
    if upload_with.get("path") != "${{ runner.temp }}/claude-safe-workspace/":
        findings.append("synthetic workspace is not isolated under runner temp")
    if upload_with.get("include-hidden-files") is not True:
        findings.append("synthetic workspace omits its Git metadata")
    if download_with.get("path") != "${{ github.workspace }}":
        findings.append("synthetic workspace is not restored at the action workspace")
    if input_steps.index(upload) != input_steps.index(builder) + 1:
        findings.append("synthetic workspace is not uploaded immediately after its builder")

    builder_run = str(builder.get("run", ""))
    required_lines = (
        'safe="$RUNNER_TEMP/claude-safe-workspace"',
        'export HOME="$RUNNER_TEMP/claude-safe-home"',
        "GIT_CONFIG_NOSYSTEM=1",
        'rm -rf "$safe" "$HOME"',
        "git init --bare .review-origin",
        "Synthetic workspace: no pull-request files.",
        "cat > inline-review-mcp.cjs <<'NODE'",
        "PANELCAST_REVIEW_DIFF_B64",
        'message.method === "tools/list"',
        'message.params?.name === "get_diff"',
        "git add README.md inline-review-mcp.cjs",
    )
    findings.extend(
        f"synthetic workspace builder omits {required}"
        for required in required_lines
        if required not in builder_run
    )
    return findings


def _inline_diff_violations(workflow: dict[str, Any]) -> list[str]:
    review_input = jobs(workflow).get("review-input") or {}
    render_steps = [
        step
        for step in _steps(review_input)
        if step.get("name") == "Render the pull request diff for review"
    ]
    if len(render_steps) != 1:
        return ["inline diff requires exactly one named render step"]
    render = render_steps[0]
    run = str(render.get("run", ""))
    findings: list[str] = []
    if render.get("id") != "render":
        findings.append("inline diff render step has the wrong id")
    if (review_input.get("outputs") or {}).get("review_diff_b64") != (
        "${{ steps.render.outputs.review_diff_b64 }}"
    ):
        findings.append("encoded diff job output is not wired to the render step")
    for required in (
        'if [ "$size" -gt 200000 ]',
        "review_diff_b64=%s",
        'base64 -w0 "$diff_file"',
        '>> "$GITHUB_OUTPUT"',
    ):
        if required not in run:
            findings.append(f"encoded diff renderer omits {required}")

    inputs = _review_action_inputs(jobs(workflow).get("review") or {})
    prompt = str(inputs.get("prompt", ""))
    if "call mcp__review_diff__get_diff exactly once" not in prompt:
        findings.append("credentialed review prompt does not require the diff tool")
    if "untrusted data" not in prompt:
        findings.append("credentialed review prompt does not label the diff as untrusted data")
    try:
        settings = json.loads(str(inputs.get("settings", "")))
        servers = settings["mcpServers"]
        decoder = servers["review_diff"]
    except (KeyError, TypeError, json.JSONDecodeError):
        findings.append("credentialed review settings omit the diff decoder")
    else:
        if set(servers) != {"review_diff"}:
            findings.append("credentialed review settings enable extra MCP servers")
        if decoder.get("command") != "node" or decoder.get("args") != [
            "inline-review-mcp.cjs"
        ]:
            findings.append("diff decoder does not run the synthetic server")
        expected = "${{ needs.review-input.outputs.review_diff_b64 }}"
        if (decoder.get("env") or {}).get("PANELCAST_REVIEW_DIFF_B64") != expected:
            findings.append("diff decoder is not wired to the encoded job output")
    return findings


def credential_isolation_violations(workflow: dict[str, Any]) -> list[str]:
    credentialed = {name: job for name, job in jobs(workflow).items() if holds_credentials(job)}
    if not credentialed:
        return ["no credential-bearing job found, so the policy proves nothing"]

    findings = [
        f"workflow: reacts to {event}, which hands the credentials a base-branch"
        " context a pull request can aim"
        for event in sorted(set(triggers(workflow) or {}) & UNSAFE_TRIGGERS)
    ]
    findings += [
        f"{name}: {violation}"
        for name, job in credentialed.items()
        for violation in _job_violations(job)
    ]
    findings += _synthetic_workspace_violations(workflow)
    findings += _inline_diff_violations(workflow)
    return findings


@pytest.fixture
def review_workflow() -> dict[str, Any]:
    return load_workflow("claude-review.yml")


@pytest.fixture
def credentialed_job(review_workflow: dict[str, Any]) -> dict[str, Any]:
    credentialed = [job for job in jobs(review_workflow).values() if holds_credentials(job)]
    assert len(credentialed) == 1
    return credentialed[0]


def test_review_workflow_satisfies_the_isolation_policy(review_workflow: dict[str, Any]) -> None:
    assert credential_isolation_violations(review_workflow) == []


def test_policy_actually_applies_to_a_credential_bearing_job(
    review_workflow: dict[str, Any],
) -> None:
    assert any(holds_credentials(job) for job in jobs(review_workflow).values())


def test_review_never_runs_on_a_fork_or_base_branch_trigger(
    review_workflow: dict[str, Any], credentialed_job: dict[str, Any]
) -> None:
    events = triggers(review_workflow)

    assert set(events) == {"pull_request"}
    assert events["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    ]
    assert "head.repo.full_name == github.repository" in credentialed_job["if"]


def _pull_request_workflow_violations(path: Path, workflow: dict[str, Any]) -> list[str]:
    events = set(triggers(workflow) or {})
    findings = [
        f"{path.name}: reacts to unsafe trigger {event}"
        for event in sorted(events & UNSAFE_TRIGGERS)
    ]
    if "pull_request" not in events:
        return findings
    for name, job in jobs(workflow).items():
        if not _executes_pull_request_code(job):
            continue
        label = f"{path.name}:{name}"
        if holds_credentials(job, workflow):
            findings.append(f"{label}: runs pull-request code with credentials")
        permissions = _effective_permissions(workflow, job)
        if permissions is None:
            findings.append(f"{label}: has no explicit read-only permissions mapping")
        elif set(permissions.values()) - {"read"}:
            findings.append(f"{label}: permissions are not read-only")
    return findings


def test_pull_request_code_only_executes_where_there_are_no_credentials() -> None:
    workflows = [(path, load_workflow(path.name)) for path in _workflow_paths()]
    violations = [
        finding
        for path, workflow in workflows
        for finding in _pull_request_workflow_violations(path, workflow)
    ]
    executing = [
        job
        for _, workflow in workflows
        if "pull_request" in set(triggers(workflow) or {})
        for job in jobs(workflow).values()
        if _executes_pull_request_code(job)
    ]

    assert executing, "secretless CI must run the pull request's own suite"
    assert any("pixi run pytest" in _text(job) for job in executing)
    assert violations == []


@pytest.mark.parametrize(
    ("workflow", "expected"),
    [
        (
            {"on": {"pull_request_target": {}}, "jobs": {}},
            "reacts to unsafe trigger pull_request_target",
        ),
        (
            {
                "on": {"pull_request": {}},
                "permissions": {"contents": "read"},
                "env": {"TOKEN": "${{ secrets.TOKEN }}"},
                "jobs": {"test": {"steps": [{"run": "pytest"}]}},
            },
            "runs pull-request code with credentials",
        ),
        (
            {
                "on": {"pull_request": {}},
                "permissions": "write-all",
                "jobs": {"test": {"steps": [{"run": "pytest"}]}},
            },
            "has no explicit read-only permissions mapping",
        ),
        (
            {
                "on": {"pull_request": {}},
                "permissions": {"contents": "read"},
                "env": {"TOKEN": "${{ secrets.TOKEN }}"},
                "jobs": {"test": {"steps": [{"uses": "./.github/actions/test"}]}},
            },
            "runs pull-request code with credentials",
        ),
        (
            {
                "on": {"pull_request": {}},
                "permissions": {"contents": "read"},
                "env": {"TOKEN": "${{ secrets.TOKEN }}"},
                "jobs": {"test": {"uses": "./.github/workflows/test.yml"}},
            },
            "runs pull-request code with credentials",
        ),
    ],
)
def test_repository_sweep_rejects_policy_bypasses(workflow, expected) -> None:
    violations = _pull_request_workflow_violations(Path("evil.yaml"), workflow)
    assert any(expected in violation for violation in violations), violations


def test_workflow_paths_include_yml_and_yaml(tmp_path, monkeypatch) -> None:
    (tmp_path / "one.yml").write_text("name: one", encoding="utf-8")
    (tmp_path / "two.yaml").write_text("name: two", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", tmp_path)
    assert {path.name for path in _workflow_paths()} == {"one.yml", "two.yaml"}


def test_secretless_diff_log_is_complete_and_unforgeably_delimited(
    review_workflow: dict[str, Any],
) -> None:
    run = "\n".join(
        str(step.get("run", "")) for step in _steps(jobs(review_workflow)["review-input"])
    )

    assert "--no-ext-diff --no-textconv" in run
    assert "START $stop" in run and "END $stop" in run
    assert "200000" in run
    assert "300000" not in run and "2000000" not in run
    assert "diff is too large" in run
    assert 'base64 -w0 "$diff_file"' in run
    assert "review_diff_b64=%s" in run
    assert '>> "$GITHUB_OUTPUT"' in run

    review_input = jobs(review_workflow)["review-input"]
    assert review_input["outputs"]["review_diff_b64"] == (
        "${{ steps.render.outputs.review_diff_b64 }}"
    )
    inputs = _review_action_inputs(jobs(review_workflow)["review"])
    assert "mcp__review_diff__get_diff exactly once" in inputs["prompt"]
    assert "untrusted data" in inputs["prompt"]
    assert "${{ needs.review-input.outputs.review_diff_b64 }}" in inputs["settings"]


def test_reviewer_can_still_read_ci_results_instead_of_running_them(
    credentialed_job: dict[str, Any],
) -> None:
    assert credentialed_job["permissions"]["actions"] == "read"

    inputs = _review_action_inputs(credentialed_job)
    claude_args = inputs["claude_args"]
    allowed = set(_tools(claude_args, "--allowedTools"))
    ci_tools = {tool for tool in allowed if tool.startswith("mcp__github_ci__")}

    assert ci_tools == {
        "mcp__github_ci__get_ci_status",
        "mcp__github_ci__get_workflow_run_details",
    }
    assert "actions: read" in inputs["additional_permissions"]

    briefing = re.search(r'--append-system-prompt\s+"([^"]*)"', claude_args)
    assert briefing is not None
    assert "CI" in briefing.group(1)
    assert "never execute" in briefing.group(1).lower()


def test_visible_review_output_is_preserved(credentialed_job: dict[str, Any]) -> None:
    inputs = _review_action_inputs(credentialed_job)

    assert inputs["display_report"] == "true"
    assert inputs["track_progress"] == "true"
    assert inputs["prompt"].startswith("Review ")
    assert not inputs.get("plugins")
    assert not inputs.get("plugin_marketplaces")


def test_long_lived_token_is_withheld_when_workload_identity_is_configured(
    credentialed_job: dict[str, Any],
) -> None:
    inputs = _review_action_inputs(credentialed_job)

    # The action ignores federation whenever a static token is also present.
    assert inputs["claude_code_oauth_token"] == (
        "${{ vars.ANTHROPIC_ORGANIZATION_ID == '' && secrets.CLAUDE_CODE_OAUTH_TOKEN || '' }}"
    )
    assert inputs["anthropic_federation_rule_id"] == "${{ vars.ANTHROPIC_FEDERATION_RULE_ID }}"
    assert inputs["anthropic_organization_id"] == "${{ vars.ANTHROPIC_ORGANIZATION_ID }}"


def _only_credentialed_job(workflow: dict[str, Any]) -> dict[str, Any]:
    return next(job for job in jobs(workflow).values() if holds_credentials(job))


def _edit_claude_args(workflow: dict[str, Any], old: str, new: str) -> None:
    inputs = _review_action_inputs(_only_credentialed_job(workflow))
    assert old in inputs["claude_args"]
    inputs["claude_args"] = inputs["claude_args"].replace(old, new)


def _allow_tool(workflow: dict[str, Any], tool: str) -> None:
    _edit_claude_args(
        workflow,
        '--allowedTools "',
        f'--allowedTools "{tool},',
    )


def _reallow_test_commands(workflow: dict[str, Any]) -> None:
    _allow_tool(workflow, "Bash(pixi run *)")


def _allow_git_output(workflow: dict[str, Any]) -> None:
    _allow_tool(workflow, "Bash(git show --output=*)")


def _allow_git_external_diff(workflow: dict[str, Any]) -> None:
    _allow_tool(workflow, "Bash(git show --ext-diff:*)")


def _allow_shell_redirection(workflow: dict[str, Any]) -> None:
    _allow_tool(workflow, "Bash(git show *>*)")


def _allow_command_composition(workflow: dict[str, Any]) -> None:
    _allow_tool(workflow, "Bash(git show:*;*)")


def _drop_allowlist(workflow: dict[str, Any]) -> None:
    inputs = _review_action_inputs(_only_credentialed_job(workflow))
    inputs["claude_args"] = re.sub(
        r'^\s*--allowedTools "[^"]*"\s*$', "", inputs["claude_args"], flags=re.MULTILINE
    )


def _install_mutable_plugin(workflow: dict[str, Any]) -> None:
    inputs = _review_action_inputs(_only_credentialed_job(workflow))
    inputs["plugin_marketplaces"] = "https://github.com/example/plugins.git"
    inputs["plugins"] = "review@example"


def _add_test_step(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["steps"].append({"name": "verify", "run": "pixi run pytest"})


def _install_toolchain(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["steps"].insert(
        1, {"uses": "prefix-dev/setup-pixi@" + "b" * 40}
    )


def _checkout_pr_code(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["steps"].insert(
        0, {"uses": "actions/checkout@" + "a" * 40}
    )


def _unpin_action(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["steps"][0]["uses"] = "actions/checkout@v4"


def _load_repo_settings(workflow: dict[str, Any]) -> None:
    _edit_claude_args(workflow, "--setting-sources user", "--setting-sources user,project,local")


def _drop_a_denial(workflow: dict[str, Any]) -> None:
    _edit_claude_args(workflow, "Bash,", "")


def _drop_env_scrub(workflow: dict[str, Any]) -> None:
    del _only_credentialed_job(workflow)["env"]["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"]


def _widen_permissions(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["permissions"]["contents"] = "write"


def _widen_app_token(workflow: dict[str, Any]) -> None:
    del _review_action_inputs(_only_credentialed_job(workflow))["additional_permissions"]


def _allow_every_non_write_user(workflow: dict[str, Any]) -> None:
    _review_action_inputs(_only_credentialed_job(workflow))["allowed_non_write_users"] = "*"


def _accept_fork_pull_requests(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["if"] = "github.event.pull_request.draft == false"


def _switch_to_pull_request_target(workflow: dict[str, Any]) -> None:
    events = triggers(workflow)
    events["pull_request_target"] = events.pop("pull_request")


def _strip_credentials(workflow: dict[str, Any]) -> None:
    job = _only_credentialed_job(workflow)
    del job["permissions"]["id-token"]
    _review_action_inputs(job)["claude_code_oauth_token"] = "${{ vars.SOMETHING_ELSE }}"


def _upload_pr_workspace(workflow: dict[str, Any]) -> None:
    upload = next(
        step
        for step in _steps(jobs(workflow)["review-input"])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    upload["with"]["path"] = "${{ github.workspace }}"


def _omit_synthetic_git_metadata(workflow: dict[str, Any]) -> None:
    upload = next(
        step
        for step in _steps(jobs(workflow)["review-input"])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    upload["with"]["include-hidden-files"] = False


def _add_decoy_artifact_transfer(workflow: dict[str, Any]) -> None:
    jobs(workflow)["review-input"]["steps"].insert(
        0,
        {
            "name": "Decoy upload",
            "uses": "actions/upload-artifact@" + "a" * 40,
            "with": {"name": SYNTHETIC_ARTIFACT, "path": "${{ github.workspace }}"},
        },
    )
    _only_credentialed_job(workflow)["steps"].insert(
        0,
        {
            "name": "Decoy download",
            "uses": "actions/download-artifact@" + "a" * 40,
            "with": {"name": SYNTHETIC_ARTIFACT, "path": "${{ github.workspace }}"},
        },
    )


def _move_builder_contract_to_a_decoy(workflow: dict[str, Any]) -> None:
    steps = jobs(workflow)["review-input"]["steps"]
    builder = next(step for step in steps if step.get("name") == SYNTHETIC_BUILDER)
    original = str(builder["run"])
    builder["run"] = "printf '%s\\n' unsafe-builder"
    steps.insert(0, {"name": "Decoy safe text", "run": original})


def _render_step(workflow: dict[str, Any]) -> dict[str, Any]:
    return next(
        step
        for step in _steps(jobs(workflow)["review-input"])
        if step.get("name") == "Render the pull request diff for review"
    )


def _drop_base64_encoding(workflow: dict[str, Any]) -> None:
    render = _render_step(workflow)
    render["run"] = str(render["run"]).replace(
        'base64 -w0 "$diff_file"', 'cat "$diff_file"'
    )


def _drop_inline_diff_wiring(workflow: dict[str, Any]) -> None:
    del jobs(workflow)["review-input"]["outputs"]["review_diff_b64"]


def _widen_inline_diff_cap(workflow: dict[str, Any]) -> None:
    render = _render_step(workflow)
    render["run"] = str(render["run"]).replace("-gt 200000", "-gt 2000000")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_reallow_test_commands, "allows a tool that is not read-only: Bash(pixi run *)"),
        (_allow_git_output, "allows a tool that is not read-only: Bash(git show --output=*)"),
        (
            _allow_git_external_diff,
            "allows a tool that is not read-only: Bash(git show --ext-diff:*)",
        ),
        (_allow_shell_redirection, "allows a tool that is not read-only: Bash(git show *>*)"),
        (_allow_command_composition, "allows a tool that is not read-only: Bash(git show:*;*)"),
        (_drop_allowlist, "omits the allowlist"),
        (_install_mutable_plugin, "installs mutable plugin code"),
        (_add_test_step, "runs a shell step alongside the review credentials"),
        (_install_toolchain, "installs a project toolchain: prefix-dev/setup-pixi"),
        (_checkout_pr_code, "checks out pull-request code alongside review credentials"),
        (_unpin_action, "not commit-pinned: actions/checkout@v4"),
        (_load_repo_settings, "loads in-repo Claude settings"),
        (_drop_a_denial, "does not deny Bash"),
        (_drop_env_scrub, "does not enable subprocess environment scrubbing"),
        (_widen_permissions, "requests contents: write"),
        (_widen_app_token, "does not enable the read-only GitHub CI MCP server"),
        (_allow_every_non_write_user, "beyond the repository owner"),
        (_accept_fork_pull_requests, "does not restrict itself to same-repository"),
        (_switch_to_pull_request_target, "reacts to pull_request_target"),
        (_strip_credentials, "no credential-bearing job found"),
        (_upload_pr_workspace, "not isolated under runner temp"),
        (_omit_synthetic_git_metadata, "omits its Git metadata"),
        (_add_decoy_artifact_transfer, "exactly one builder, upload, and download"),
        (_move_builder_contract_to_a_decoy, "builder omits"),
        (_drop_base64_encoding, "renderer omits"),
        (_drop_inline_diff_wiring, "job output is not wired"),
        (_widen_inline_diff_cap, "renderer omits"),
    ],
)
def test_policy_rejects_a_reviewer_that_can_execute_pull_request_code(
    review_workflow: dict[str, Any], mutate: Any, expected: str
) -> None:
    tampered = copy.deepcopy(review_workflow)
    mutate(tampered)

    violations = credential_isolation_violations(tampered)

    assert any(expected in violation for violation in violations), violations
