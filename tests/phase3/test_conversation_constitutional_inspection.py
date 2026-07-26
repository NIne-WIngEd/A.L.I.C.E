from __future__ import annotations

from alice_conversation.constitutional_inspection import (
    inspect_constitutional_system_contract,
    render_constitutional_contract_inspection,
)
from alice_conversation.constitutional_policy import (
    parse_constitutional_dialogue_policy,
)
from alice_conversation.constitutional_prompt import (
    compile_constitutional_system_contract,
)

from _constitutional_helpers import copy_policy_payload, make_governance_repository


def _inspection(tmp_path):
    payload = copy_policy_payload()
    root = make_governance_repository(tmp_path, payload=payload)
    contract = compile_constitutional_system_contract(
        policy=parse_constitutional_dialogue_policy(payload),
        repository_root=root,
    )
    return contract, inspect_constitutional_system_contract(contract)


def test_inspection_reports_contract_metadata(tmp_path) -> None:
    contract, inspection = _inspection(tmp_path)
    assert inspection.version == contract.version
    assert inspection.policy_version == contract.policy_version
    assert inspection.constitution_version == contract.constitution_version
    assert inspection.content_sha256 == contract.content_sha256
    assert inspection.content_characters == len(contract.content)


def test_inspection_reports_all_source_bindings(tmp_path) -> None:
    _, inspection = _inspection(tmp_path)
    assert inspection.source_count == 4
    assert len(inspection.sources) == 4
    assert all(len(source.normalized_sha256) == 64 for source in inspection.sources)


def test_inspection_confirms_no_embedded_grounding(tmp_path) -> None:
    _, inspection = _inspection(tmp_path)
    assert inspection.contains_untrusted_grounding is False


def test_inspection_confirms_no_source_text_export(tmp_path) -> None:
    _, inspection = _inspection(tmp_path)
    assert inspection.contains_source_text is False


def test_rendered_inspection_is_deterministic(tmp_path) -> None:
    _, inspection = _inspection(tmp_path)
    assert render_constitutional_contract_inspection(inspection) == (
        render_constitutional_contract_inspection(inspection)
    )


def test_rendered_inspection_excludes_contract_body(tmp_path) -> None:
    contract, inspection = _inspection(tmp_path)
    rendered = render_constitutional_contract_inspection(inspection)
    assert contract.content not in rendered
    assert "AUTHORITY AND IDENTITY" not in rendered
    assert "content_sha256=" in rendered


def test_rendered_inspection_lists_only_source_metadata(tmp_path) -> None:
    _, inspection = _inspection(tmp_path)
    rendered = render_constitutional_contract_inspection(inspection)
    assert rendered.count("source=") == 4
    assert "required_markers" not in rendered
