from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load("audit_capability_barriers", "scripts/audit_capability_barriers.py")
REGISTER = _load("register_released_baseline_scopes", "scripts/register_released_baseline_scopes.py")
ACTIVE_REGISTER = _load("register_active_milestone_scopes", "scripts/register_active_milestone_scopes.py")


def _registry(tmp_path: Path, payload: dict | None = None) -> Path:
    policies = tmp_path / "policies"
    policies.mkdir(exist_ok=True)
    path = policies / "phase_scope_registry.json"
    path.write_text(
        json.dumps(payload or {"entries": {}, "patterns": []}),
        encoding="utf-8",
    )
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_git(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "alice-tests@example.invalid")
    _git(repo, "config", "user.name", "A.L.I.C.E. Tests")


def test_unscoped_permanent_barrier_is_reported(tmp_path: Path) -> None:
    _registry(tmp_path)
    (tmp_path / "policy.py").write_text(
        "default_deny: true\nweb_access_allowed = False\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path)
    assert any(item.disposition == "unresolved_active_barrier" for item in findings)


def test_registered_compatibility_barrier_is_not_active(tmp_path: Path) -> None:
    _registry(
        tmp_path,
        {
            "entries": {
                "policy.py": {
                    "scope_kind": "phase_local_compatibility",
                    "capability_ceiling": False,
                }
            },
            "patterns": [],
        },
    )
    (tmp_path / "policy.py").write_text("web_access_allowed = False\n", encoding="utf-8")
    findings = MODULE.audit(tmp_path)
    assert findings
    assert all(item.disposition != "unresolved_active_barrier" for item in findings)


def test_registered_pattern_scopes_compatibility_files(tmp_path: Path) -> None:
    _registry(
        tmp_path,
        {
            "entries": {},
            "patterns": [
                {
                    "glob": "tests/phase3/**",
                    "scope_kind": "compatibility_test",
                    "capability_ceiling": False,
                }
            ],
        },
    )
    target = tmp_path / "tests" / "phase3" / "test_legacy.py"
    target.parent.mkdir(parents=True)
    target.write_text("web_access_allowed = False\n", encoding="utf-8")
    findings = MODULE.audit(tmp_path)
    assert findings
    assert all(item.disposition != "unresolved_active_barrier" for item in findings)


def test_external_registry_can_audit_another_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "policy.py").write_text("web_access_allowed = False\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "entries": {
                    "policy.py": {
                        "scope_kind": "phase_local_compatibility",
                        "capability_ceiling": False,
                    }
                },
                "patterns": [],
            }
        ),
        encoding="utf-8",
    )
    findings = MODULE.audit(repo, registry)
    assert findings
    assert all(item.disposition != "unresolved_active_barrier" for item in findings)


def test_component_local_json_policy_flags_are_not_global_barriers(tmp_path: Path) -> None:
    _registry(tmp_path)
    (tmp_path / "policies" / "claim_entailment_policy.json").write_text(
        json.dumps(
            {
                "claim_entailment_policy_schema_version": 1,
                "policy_id": "claim-entailment-v1",
                "maximum_output_tokens": 512,
                "memory_write_allowed": False,
                "tool_calling_allowed": False,
                "web_access_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path)
    assert findings
    assert all(item.disposition != "unresolved_active_barrier" for item in findings)
    assert any(item.disposition == "component_local_policy_declaration" for item in findings)


def test_destination_policy_false_capability_remains_a_barrier(tmp_path: Path) -> None:
    _registry(tmp_path)
    (tmp_path / "policies" / "global_governance_policy.json").write_text(
        json.dumps(
            {
                "policy_id": "global-governance",
                "scope": "global",
                "web_access_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path)
    assert any(item.disposition == "unresolved_active_barrier" for item in findings)


def test_maximum_parameter_is_not_a_fixed_universal_limit(tmp_path: Path) -> None:
    _registry(tmp_path)
    (tmp_path / "transport.py").write_text(
        "def read_exact(length: int, *, maximum: int) -> bytes:\n    return b''\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path)
    assert not any(item.code == "FIXED_UNIVERSAL_LIMIT" for item in findings)


def test_numeric_hard_limit_is_still_reported(tmp_path: Path) -> None:
    _registry(tmp_path)
    (tmp_path / "global_policy.py").write_text(
        "# hard limit of 2 retries for every future runtime\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path)
    assert any(item.code == "FIXED_UNIVERSAL_LIMIT" for item in findings)


def test_released_baseline_registrar_creates_exact_entries_only(tmp_path: Path) -> None:
    _init_git(tmp_path)
    target = tmp_path / "src" / "alice_conversation" / "policy.py"
    target.parent.mkdir(parents=True)
    target.write_text("web_access_allowed = False\n", encoding="utf-8")
    registry = _registry(tmp_path)
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "baseline")

    added = REGISTER.register(tmp_path, registry, "HEAD")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert added == 1
    assert "src/alice_conversation/policy.py" in payload["entries"]
    assert not any(
        item.get("glob") == "src/alice_conversation/**"
        for item in payload.get("patterns", [])
    )


def test_changed_only_audit_ignores_unchanged_legacy_and_flags_new_line(tmp_path: Path) -> None:
    _init_git(tmp_path)
    _registry(tmp_path)
    target = tmp_path / "src" / "alice_conversation" / "policy.py"
    target.parent.mkdir(parents=True)
    target.write_text("web_access_allowed = False\n", encoding="utf-8")
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "baseline")

    # Unchanged legacy restriction is not part of the changed-line audit.
    target.write_text(
        "web_access_allowed = False\n# ordinary Phase 4.5 implementation change\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path, changed_only_against="HEAD")
    assert not any(item.disposition == "unresolved_active_barrier" for item in findings)

    # A newly added global ceiling is reviewed strictly even in a legacy path.
    target.write_text(
        "web_access_allowed = False\n# ordinary Phase 4.5 implementation change\ndefault_deny: true\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path, changed_only_against="HEAD")
    assert any(item.code == "DEFAULT_DENY_GLOBAL" for item in findings)
    assert any(item.disposition == "unresolved_active_barrier" for item in findings)


def test_exact_active_milestone_scope_can_cover_listed_changed_line_rule(tmp_path: Path) -> None:
    _init_git(tmp_path)
    target = tmp_path / "src" / "alice_information" / "grounding_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text("# baseline\n", encoding="utf-8")
    registry = _registry(
        tmp_path,
        {
            "entries": {
                "src/alice_information/grounding_policy.py": {
                    "scope_kind": "active_milestone_guard",
                    "capability_ceiling": False,
                    "profile": "information.phase4.grounding",
                    "milestone": "P4.5a",
                    "applies_to_changed_lines": True,
                    "changed_line_rule_codes": ["MUST_REMAIN_DISABLED"],
                    "sunset_condition": "Migrate to capability profiles.",
                }
            },
            "patterns": [],
        },
    )
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "baseline")
    target.write_text(
        "# P4.5a\nraise ValueError('External actions must remain disabled.')\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path, registry, changed_only_against="HEAD")
    assert findings
    assert all(item.disposition != "unresolved_active_barrier" for item in findings)
    assert any(item.disposition == "registered_active_milestone_scope" for item in findings)


def test_active_milestone_scope_does_not_waive_unlisted_rule(tmp_path: Path) -> None:
    _init_git(tmp_path)
    target = tmp_path / "src" / "alice_information" / "grounding_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text("# baseline\n", encoding="utf-8")
    registry = _registry(
        tmp_path,
        {
            "entries": {
                "src/alice_information/grounding_policy.py": {
                    "scope_kind": "active_milestone_guard",
                    "capability_ceiling": False,
                    "profile": "information.phase4.grounding",
                    "milestone": "P4.5a",
                    "applies_to_changed_lines": True,
                    "changed_line_rule_codes": ["MUST_REMAIN_DISABLED"],
                    "sunset_condition": "Migrate to capability profiles.",
                }
            },
            "patterns": [],
        },
    )
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "baseline")
    target.write_text("# P4.5a\ndefault_deny: true\n", encoding="utf-8")
    findings = MODULE.audit(tmp_path, registry, changed_only_against="HEAD")
    assert any(item.code == "DEFAULT_DENY_GLOBAL" for item in findings)
    assert any(item.disposition == "unresolved_active_barrier" for item in findings)


def test_active_milestone_scope_requires_matching_milestone(tmp_path: Path) -> None:
    _init_git(tmp_path)
    target = tmp_path / "src" / "alice_information" / "grounding_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text("# baseline\n", encoding="utf-8")
    registry = _registry(
        tmp_path,
        {
            "entries": {
                "src/alice_information/grounding_policy.py": {
                    "scope_kind": "active_milestone_guard",
                    "capability_ceiling": False,
                    "profile": "information.phase4.grounding",
                    "milestone": "P4.5a",
                    "applies_to_changed_lines": True,
                    "changed_line_rule_codes": ["MUST_REMAIN_DISABLED"],
                    "sunset_condition": "Migrate to capability profiles.",
                }
            },
            "patterns": [],
        },
    )
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "baseline")
    target.write_text(
        "# Different milestone\nraise ValueError('Memory writes must remain disabled.')\n",
        encoding="utf-8",
    )
    findings = MODULE.audit(tmp_path, registry, changed_only_against="HEAD")
    assert any(item.disposition == "unresolved_active_barrier" for item in findings)


def test_active_milestone_registrar_adds_exact_grounding_scope(tmp_path: Path) -> None:
    active = tmp_path / "active"
    active.mkdir()
    target = active / "src" / "alice_information" / "grounding_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "# P4.5a\n"
        "raise ValueError('Prohibited P4.5a capabilities must remain disabled.')\n"
        "raise ValueError('External actions must remain disabled.')\n"
        "raise ValueError('Memory writes must remain disabled.')\n",
        encoding="utf-8",
    )
    registry = _registry(tmp_path)
    assert ACTIVE_REGISTER.register(active, registry) is True
    payload = json.loads(registry.read_text(encoding="utf-8"))
    entry = payload["entries"]["src/alice_information/grounding_policy.py"]
    assert entry["scope_kind"] == "active_milestone_guard"
    assert entry["changed_line_rule_codes"] == ["MUST_REMAIN_DISABLED"]
