from __future__ import annotations

from pathlib import Path

import yaml

from alice_conversation.constitutional_policy import (
    load_constitutional_dialogue_policy,
)
from alice_conversation.constitutional_prompt import (
    compile_constitutional_system_contract,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_sensitive_external_transmission_preserves_phase2_compatibility() -> None:
    root = _root()
    registry = yaml.safe_load(
        (root / "policies" / "permissions.yaml").read_text(encoding="utf-8")
    )
    permissions = {item["id"]: item for item in registry["permissions"]}
    external = permissions["highly_sensitive.transmit_external"]

    assert external["level"] == "P4"
    assert external["confirmation"] == "strong"
    assert external["standing_authorization_allowed"] is False
    assert external["ratified_mission_authorization_allowed"] is True
    assert external["allowed_data_classes"] == ["HIGHLY_SENSITIVE"]


def test_phase3_compiler_resolves_ratified_governance_migration() -> None:
    root = _root()
    policy = load_constitutional_dialogue_policy(
        root / "policies" / "conversation_constitutional_policy.json"
    )
    contract = compile_constitutional_system_contract(
        policy=policy,
        repository_root=root,
    )

    assert contract.version == "alice-constitutional-dialogue-1.0.0"
    assert contract.constitution_version == "1.1.0"
    assert {source.path: source.version for source in contract.sources} == {
        "docs/ALICE_CONSTITUTION.md": "1.1.0",
        "docs/EVALUATION_CHARTER.md": "2.0.0",
        "docs/PERMISSION_MODEL.md": "3.0.0",
        "docs/THREAT_MODEL.md": "2.0.0",
    }
    assert "profile-local maturity boundary" in contract.content
    assert "intended future directions" in contract.content
