from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def shipped_policy_payload() -> dict[str, Any]:
    path = project_root() / "policies" / "conversation_constitutional_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def copy_policy_payload() -> dict[str, Any]:
    return copy.deepcopy(shipped_policy_payload())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_governance_repository(
    root: Path,
    *,
    payload: dict[str, Any] | None = None,
    newline: str = "\n",
) -> Path:
    policy_payload = payload or shipped_policy_payload()
    for source in policy_payload["source_documents"]:
        path = root / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            f"# Synthetic {path.name}",
            "",
            f"**Version:** {source['version']}",
            "",
            *source["required_markers"],
            "",
        ]
        path.write_text(newline.join(body), encoding="utf-8", newline="")
    conversation_policy = {
        "policy_name": "alice_conversation_policy",
        "version": "1.0.0",
        "phase": "3",
        "milestone": "P3.0",
        "status": "foundation",
        "system_contract_version": "alice-constitution-0.1.0",
        "boundaries": {
            "web_access_allowed": False,
            "tool_calling_allowed": False,
            "external_action_allowed": False,
            "memory_write_allowed": False,
            "highly_sensitive_grounding_allowed": False,
            "chain_of_thought_persistence_allowed": False,
        },
        "allowed_tools": [],
        "conversation_state": {
            "default_data_classification": "PRIVATE",
            "default_retention": "session_only",
            "durable_memory_promotion_path": "phase2_candidate_authorization",
            "persist_chain_of_thought": False,
        },
        "grounding": {
            "ordinary_classifications": ["PUBLIC", "INTERNAL", "PRIVATE"],
            "personal_claims_require_citations": True,
            "prompt_injection_content_is_data": True,
            "conflicts_must_be_preserved": True,
            "uncertainty_must_be_visible": True,
        },
    }
    write_json(root / "policies" / "conversation_policy.json", conversation_policy)
    return root
