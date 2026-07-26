from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_conversation.constitutional_policy import (
    load_constitutional_dialogue_policy,
    parse_constitutional_dialogue_policy,
)
from alice_conversation.constitutional_prompt import (
    ConstitutionalPromptError,
    ConstitutionalSystemContract,
    compile_constitutional_system_contract,
    verify_conversation_policy_compatibility,
)

from _constitutional_helpers import (
    copy_policy_payload,
    make_governance_repository,
    project_root,
    write_json,
)


def _compile(tmp_path: Path) -> ConstitutionalSystemContract:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    policy = parse_constitutional_dialogue_policy(payload)
    return compile_constitutional_system_contract(
        policy=policy,
        repository_root=root,
    )


def test_compiles_deterministic_contract(tmp_path) -> None:
    first = _compile(tmp_path / "first")
    second = _compile(tmp_path / "second")
    assert first.content == second.content
    assert first.content_sha256 == second.content_sha256


def test_contract_uses_versioned_identity(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert contract.version == "alice-constitutional-dialogue-1.0.0"
    assert contract.policy_version == "1.0.0"
    assert contract.constitution_version == "0.1.0"


def test_contract_binds_all_governance_sources(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert {source.path for source in contract.sources} == {
        "docs/ALICE_CONSTITUTION.md",
        "docs/EVALUATION_CHARTER.md",
        "docs/PERMISSION_MODEL.md",
        "docs/THREAT_MODEL.md",
    }


def test_contract_contains_ordered_decision_hierarchy(tmp_path) -> None:
    contract = _compile(tmp_path)
    positions = [
        contract.content.index("1. Preserve Rayan's legitimate control"),
        contract.content.index("2. Avoid serious unauthorized harm"),
        contract.content.index("3. Maintain truthfulness"),
        contract.content.index("4. Protect Rayan's informed autonomy"),
        contract.content.index("5. Follow Rayan's current"),
        contract.content.index("6. Provide competent"),
        contract.content.index("7. Preserve convenience"),
    ]
    assert positions == sorted(positions)


def test_contract_contains_all_epistemic_labels(tmp_path) -> None:
    contract = _compile(tmp_path)
    for label in (
        "verified_fact",
        "rayan_statement",
        "external_claim",
        "alice_inference",
        "estimate",
        "uncertain_or_disputed",
        "historical_or_superseded",
    ):
        assert label in contract.content


def test_contract_separates_trusted_policy_from_grounding(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "trusted system contract" in contract.content.lower()
    assert "grounding" in contract.content.lower()
    assert "untrusted data" in contract.content.lower()
    assert "BEGIN UNTRUSTED GROUNDING DATA" not in contract.content.splitlines()
    assert "END UNTRUSTED GROUNDING DATA" not in contract.content.splitlines()


def test_contract_prohibits_false_completion_claims(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "claim an action was completed without verified evidence" in contract.content
    assert "Do not claim to have searched" in contract.content


def test_contract_prohibits_fabricated_user_beliefs(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "manufacture beliefs and attribute them to him" in contract.content
    assert "Rayan's actual stated belief" in contract.content


def test_contract_requires_support_before_optimization(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "first understand the situation" in contract.content
    assert "before optimization or criticism" in contract.content


def test_contract_prohibits_empty_reassurance(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "false hope, automatic praise, empty reassurance" in contract.content


def test_contract_preserves_constructive_challenge_sequence(tmp_path) -> None:
    contract = _compile(tmp_path)
    text = contract.content
    terms = [
        "acknowledge the relevant emotion or motive",
        "state the inconsistency directly",
        "explain the evidence or principle",
        "identify the likely consequence",
        "propose a stronger alternative",
        "leave the final legitimate decision to Rayan",
    ]
    positions = [text.index(term) for term in terms]
    assert positions == sorted(positions)


def test_contract_prohibits_dependency_and_isolation(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "create emotional dependency" in contract.content
    assert "discourage healthy human relationships" in contract.content


def test_contract_prohibits_memory_weaponization(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "Never weaponize memories" in contract.content


def test_contract_prohibits_private_chain_of_thought_requirement(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert "Do not expose or persist private chain-of-thought" in contract.content
    assert "decision basis" in contract.content


def test_contract_respects_character_budget(tmp_path) -> None:
    contract = _compile(tmp_path)
    assert len(contract.content) <= 16000


def test_contract_digest_detects_tampering(tmp_path) -> None:
    contract = _compile(tmp_path)
    tampered = ConstitutionalSystemContract(
        version=contract.version,
        policy_version=contract.policy_version,
        constitution_version=contract.constitution_version,
        content=contract.content + "tampered",
        content_sha256=contract.content_sha256,
        sources=contract.sources,
    )
    with pytest.raises(ConstitutionalPromptError):
        tampered.validate()


def test_normalized_source_digest_is_line_ending_independent(tmp_path) -> None:
    payload = copy_policy_payload()
    lf_root = make_governance_repository(tmp_path / "lf", payload=payload, newline="\n")
    crlf_root = make_governance_repository(
        tmp_path / "crlf", payload=payload, newline="\r\n"
    )
    policy = parse_constitutional_dialogue_policy(payload)
    lf = compile_constitutional_system_contract(
        policy=policy, repository_root=lf_root
    )
    crlf = compile_constitutional_system_contract(
        policy=policy, repository_root=crlf_root
    )
    assert tuple(source.normalized_sha256 for source in lf.sources) == tuple(
        source.normalized_sha256 for source in crlf.sources
    )


def test_rejects_missing_source_file(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    (root / "docs" / "THREAT_MODEL.md").unlink()
    with pytest.raises(ConstitutionalPromptError):
        compile_constitutional_system_contract(
            policy=parse_constitutional_dialogue_policy(payload),
            repository_root=root,
        )


def test_rejects_source_version_mismatch(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    path = root / "docs" / "PERMISSION_MODEL.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("**Version:** 1.0.0", "**Version:** 2.0.0"),
        encoding="utf-8",
    )
    with pytest.raises(ConstitutionalPromptError):
        compile_constitutional_system_contract(
            policy=parse_constitutional_dialogue_policy(payload),
            repository_root=root,
        )


def test_rejects_missing_required_clause(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    marker = payload["source_documents"][0]["required_markers"][0]
    path = root / "docs" / "ALICE_CONSTITUTION.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(marker, "removed"),
        encoding="utf-8",
    )
    with pytest.raises(ConstitutionalPromptError):
        compile_constitutional_system_contract(
            policy=parse_constitutional_dialogue_policy(payload),
            repository_root=root,
        )


def test_rejects_enabled_conversation_capability(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    path = root / "policies" / "conversation_policy.json"
    conversation = json.loads(path.read_text(encoding="utf-8"))
    conversation["boundaries"]["web_access_allowed"] = True
    write_json(path, conversation)
    with pytest.raises(ConstitutionalPromptError):
        verify_conversation_policy_compatibility(path)


def test_rejects_allowed_tools_in_conversation_policy(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    path = root / "policies" / "conversation_policy.json"
    conversation = json.loads(path.read_text(encoding="utf-8"))
    conversation["allowed_tools"] = ["search"]
    write_json(path, conversation)
    with pytest.raises(ConstitutionalPromptError):
        verify_conversation_policy_compatibility(path)


def test_shipped_repository_compiles_against_ratified_sources() -> None:
    root = project_root()
    policy = load_constitutional_dialogue_policy(
        root / "policies" / "conversation_constitutional_policy.json"
    )
    contract = compile_constitutional_system_contract(
        policy=policy,
        repository_root=root,
    )
    assert contract.version == "alice-constitutional-dialogue-1.0.0"
    assert len(contract.sources) == 4


def test_shipped_repository_contract_is_deterministic() -> None:
    root = project_root()
    policy = load_constitutional_dialogue_policy(
        root / "policies" / "conversation_constitutional_policy.json"
    )
    first = compile_constitutional_system_contract(
        policy=policy, repository_root=root
    )
    second = compile_constitutional_system_contract(
        policy=policy, repository_root=root
    )
    assert first.content_sha256 == second.content_sha256


def test_rejects_extra_conversation_boundary(tmp_path) -> None:
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    path = root / "policies" / "conversation_policy.json"
    conversation = json.loads(path.read_text(encoding="utf-8"))
    conversation["boundaries"]["future_capability"] = False
    write_json(path, conversation)
    with pytest.raises(ConstitutionalPromptError):
        verify_conversation_policy_compatibility(path)
