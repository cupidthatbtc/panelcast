"""Isolation policy for the credential-bearing Claude review job (#371).

The reviewer runs with a long-lived Claude token, a minted GitHub App token, and
OIDC. A collaborator branch controls everything in the checkout, so the policy
below is what keeps that branch from getting its own code executed next to those
credentials. The mutation tests exist because a rule nobody can fail is not a
rule.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

# The CI MCP server is action-owned and receives a separate read-only workflow
# token. The credentialed reviewer gets no filesystem, shell, or network tool.
READ_ONLY_TOOLS = frozenset(
    {
        "mcp__github_ci__get_ci_status",
        "mcp__github_ci__get_workflow_run_details",
        "mcp__github_ci__download_job_log",
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


def holds_credentials(job: dict[str, Any]) -> bool:
    permissions = job.get("permissions") or {}
    return "secrets." in _text(job) or permissions.get("id-token") == "write"


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

    if "actions: read" not in str(inputs.get("additional_permissions", "")):
        yield "does not enable the read-only GitHub CI MCP server"


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
    assert events["pull_request"]["types"] == ["opened", "reopened", "ready_for_review"]
    assert "head.repo.full_name == github.repository" in credentialed_job["if"]


def test_pull_request_code_only_executes_where_there_are_no_credentials() -> None:
    executing: dict[str, dict[str, Any]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = load_workflow(path.name)
        if "pull_request" not in set(triggers(workflow) or {}):
            continue
        for name, job in jobs(workflow).items():
            if any("run" in step for step in _steps(job)):
                executing[f"{path.name}:{name}"] = job

    assert executing, "secretless CI must run the pull request's own suite"
    assert any("pixi run pytest" in _text(job) for job in executing.values())
    for name, job in executing.items():
        assert not holds_credentials(job), f"{name} runs PR code with credentials"
        assert set((job.get("permissions") or {}).values()) <= {"read"}, name


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
        "mcp__github_ci__download_job_log",
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
        '--allowedTools "mcp__github_ci__',
        f'--allowedTools "{tool},mcp__github_ci__',
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


def _accept_fork_pull_requests(workflow: dict[str, Any]) -> None:
    _only_credentialed_job(workflow)["if"] = "github.event.pull_request.draft == false"


def _switch_to_pull_request_target(workflow: dict[str, Any]) -> None:
    events = triggers(workflow)
    events["pull_request_target"] = events.pop("pull_request")


def _strip_credentials(workflow: dict[str, Any]) -> None:
    job = _only_credentialed_job(workflow)
    del job["permissions"]["id-token"]
    _review_action_inputs(job)["claude_code_oauth_token"] = "${{ vars.SOMETHING_ELSE }}"


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
        (_accept_fork_pull_requests, "does not restrict itself to same-repository"),
        (_switch_to_pull_request_target, "reacts to pull_request_target"),
        (_strip_credentials, "no credential-bearing job found"),
    ],
)
def test_policy_rejects_a_reviewer_that_can_execute_pull_request_code(
    review_workflow: dict[str, Any], mutate: Any, expected: str
) -> None:
    tampered = copy.deepcopy(review_workflow)
    mutate(tampered)

    violations = credential_isolation_violations(tampered)

    assert any(expected in violation for violation in violations), violations
