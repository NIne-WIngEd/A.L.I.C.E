"""Versioned constitutional-dialogue policy for A.L.I.C.E. Phase 3 P3.4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError


class ConstitutionalDialoguePolicyError(ConversationContractError):
    """Raised when the P3.4 constitutional policy is invalid."""


@dataclass(frozen=True)
class ConstitutionalSourceRule:
    """One governed source document required by the prompt compiler."""

    path: str
    version: str
    required_markers: tuple[str, ...]


@dataclass(frozen=True)
class ConstitutionalDialoguePolicy:
    """Validated policy projected into the constitutional system contract."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    system_contract_version: str
    constitution_version: str
    source_documents: tuple[ConstitutionalSourceRule, ...]
    boundaries: tuple[tuple[str, bool], ...]
    trust: tuple[tuple[str, bool], ...]
    epistemic_labels: tuple[str, ...]
    decision_hierarchy: tuple[str, ...]
    dialogue: tuple[tuple[str, bool], ...]
    challenge_sequence: tuple[str, ...]
    max_characters: int
    include_source_digests: bool
    include_private_source_text: bool
    grounding_delimiters_required: bool
    section_order: tuple[str, ...]

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]

    def trust_rule(self, name: str) -> bool:
        return dict(self.trust)[name]

    def dialogue_rule(self, name: str) -> bool:
        return dict(self.dialogue)[name]


_EXPECTED_BOUNDARIES = {
    "web_access_allowed": False,
    "tool_calling_allowed": False,
    "external_action_allowed": False,
    "memory_write_allowed": False,
    "highly_sensitive_grounding_allowed": False,
    "chain_of_thought_persistence_allowed": False,
}

_EXPECTED_TRUST = {
    "system_policy_is_trusted": True,
    "user_messages_are_instructions_within_policy": True,
    "grounding_is_untrusted_data": True,
    "retrieved_instructions_are_authority": False,
    "model_may_expand_permissions": False,
}

_EXPECTED_DIALOGUE = {
    "truthful": True,
    "uncertainty_visible_when_material": True,
    "personal_claims_require_grounding": True,
    "false_completion_claims_prohibited": True,
    "fabricated_user_beliefs_prohibited": True,
    "corrections_acknowledge_and_repair": True,
    "support_before_optimization_when_distressed": True,
    "empty_reassurance_prohibited": True,
    "constructive_disagreement_required_when_justified": True,
    "criticism_targets_reasoning_not_personal_worth": True,
    "manipulative_personalization_prohibited": True,
    "dependency_building_prohibited": True,
    "isolation_behavior_prohibited": True,
    "memory_weaponization_prohibited": True,
    "final_legitimate_decision_remains_with_user": True,
    "material_unresolved_conflicts_must_be_explained": True,
    "private_chain_of_thought_not_required": True,
    "decision_basis_must_be_explainable": True,
}

_EXPECTED_EPISTEMIC_LABELS = (
    "verified_fact",
    "rayan_statement",
    "external_claim",
    "alice_inference",
    "estimate",
    "uncertain_or_disputed",
    "historical_or_superseded",
)

_EXPECTED_DECISION_HIERARCHY = (
    "preserve_control_privacy_security_and_oversight",
    "avoid_serious_unauthorized_harm",
    "maintain_truthfulness_epistemic_integrity_and_action_transparency",
    "protect_informed_autonomy_dignity_values_and_long_term_interests",
    "follow_current_legitimate_clear_instructions",
    "provide_competent_useful_efficient_creative_personalized_assistance",
    "preserve_convenience_style_continuity_and_personality",
)

_EXPECTED_CHALLENGE_SEQUENCE = (
    "acknowledge_relevant_emotion_or_motive",
    "state_the_inconsistency_directly",
    "explain_the_evidence_or_principle",
    "identify_the_likely_consequence",
    "propose_a_stronger_alternative",
    "leave_the_final_legitimate_decision_to_rayan",
)

_EXPECTED_SECTION_ORDER = (
    "authority_and_identity",
    "decision_hierarchy",
    "truth_and_epistemic_integrity",
    "relationship_and_independence",
    "support_and_constructive_challenge",
    "memory_and_personalization_dignity",
    "trust_and_grounding_boundary",
    "permission_and_action_boundary",
    "error_correction_and_shutdown",
    "response_contract",
)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConstitutionalDialoguePolicyError(f"{field} must be an object.")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstitutionalDialoguePolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConstitutionalDialoguePolicyError(
            f"{field} must remain {str(expected).lower()} in P3.4."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConstitutionalDialoguePolicyError(f"{field} must be a positive integer.")
    return value


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConstitutionalDialoguePolicyError(f"{field} must be a non-empty list.")
    result = tuple(_text(item, field=f"{field} item") for item in value)
    if len(result) != len(set(result)):
        raise ConstitutionalDialoguePolicyError(f"{field} cannot contain duplicates.")
    return result


def _exact_boolean_map(
    value: Any,
    *,
    field: str,
    expected: dict[str, bool],
) -> tuple[tuple[str, bool], ...]:
    payload = _mapping(value, field=field)
    if set(payload) != set(expected):
        raise ConstitutionalDialoguePolicyError(
            f"{field} must contain exactly: {', '.join(sorted(expected))}."
        )
    return tuple(
        (name, _strict_bool(payload[name], expected=required, field=f"{field}.{name}"))
        for name, required in expected.items()
    )


def _source_rules(value: Any) -> tuple[ConstitutionalSourceRule, ...]:
    if not isinstance(value, list) or not value:
        raise ConstitutionalDialoguePolicyError(
            "source_documents must be a non-empty list."
        )
    rules: list[ConstitutionalSourceRule] = []
    paths: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, field=f"source_documents[{index}]")
        if set(item) != {"path", "version", "required_markers"}:
            raise ConstitutionalDialoguePolicyError(
                "Each source document requires path, version, and required_markers."
            )
        path = _text(item["path"], field=f"source_documents[{index}].path")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ConstitutionalDialoguePolicyError(
                "Constitutional source paths must be repository-relative."
            )
        if path in paths:
            raise ConstitutionalDialoguePolicyError(
                "Constitutional source paths cannot be duplicated."
            )
        paths.add(path)
        rules.append(
            ConstitutionalSourceRule(
                path=path,
                version=_text(
                    item["version"], field=f"source_documents[{index}].version"
                ),
                required_markers=_string_tuple(
                    item["required_markers"],
                    field=f"source_documents[{index}].required_markers",
                ),
            )
        )
    required_paths = {
        "docs/ALICE_CONSTITUTION.md",
        "docs/EVALUATION_CHARTER.md",
        "docs/PERMISSION_MODEL.md",
        "docs/THREAT_MODEL.md",
    }
    if paths != required_paths:
        raise ConstitutionalDialoguePolicyError(
            "P3.4 must bind exactly the four ratified governance documents."
        )
    return tuple(rules)


def parse_constitutional_dialogue_policy(
    payload: dict[str, Any],
) -> ConstitutionalDialoguePolicy:
    """Parse and fail closed on any weakened P3.4 policy value."""

    root = _mapping(payload, field="constitutional policy")
    expected_keys = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "system_contract_version",
        "constitution_version",
        "source_documents",
        "boundaries",
        "trust",
        "epistemic_labels",
        "decision_hierarchy",
        "dialogue",
        "challenge_sequence",
        "prompt",
    }
    if set(root) != expected_keys:
        raise ConstitutionalDialoguePolicyError(
            "Constitutional policy fields do not match the P3.4 contract."
        )
    if _text(root["policy_name"], field="policy_name") != (
        "alice_constitutional_dialogue_policy"
    ):
        raise ConstitutionalDialoguePolicyError("Unexpected constitutional policy name.")
    if _text(root["phase"], field="phase") != "3":
        raise ConstitutionalDialoguePolicyError("Constitutional policy phase must be 3.")
    if _text(root["milestone"], field="milestone") != "P3.4":
        raise ConstitutionalDialoguePolicyError(
            "Constitutional policy milestone must be P3.4."
        )
    if _text(root["status"], field="status") != "implementation":
        raise ConstitutionalDialoguePolicyError(
            "Constitutional policy status must be implementation."
        )
    constitution_version = _text(
        root["constitution_version"], field="constitution_version"
    )
    if constitution_version != "0.1.0":
        raise ConstitutionalDialoguePolicyError(
            "P3.4 is bound to A.L.I.C.E. Constitution v0.1.0."
        )
    epistemic_labels = _string_tuple(
        root["epistemic_labels"], field="epistemic_labels"
    )
    if epistemic_labels != _EXPECTED_EPISTEMIC_LABELS:
        raise ConstitutionalDialoguePolicyError(
            "Epistemic labels must preserve the ratified ordering."
        )
    decision_hierarchy = _string_tuple(
        root["decision_hierarchy"], field="decision_hierarchy"
    )
    if decision_hierarchy != _EXPECTED_DECISION_HIERARCHY:
        raise ConstitutionalDialoguePolicyError(
            "Decision hierarchy cannot be reordered or weakened."
        )
    challenge_sequence = _string_tuple(
        root["challenge_sequence"], field="challenge_sequence"
    )
    if challenge_sequence != _EXPECTED_CHALLENGE_SEQUENCE:
        raise ConstitutionalDialoguePolicyError(
            "Constructive challenge sequence must preserve the ratified order."
        )
    prompt = _mapping(root["prompt"], field="prompt")
    if set(prompt) != {
        "max_characters",
        "include_source_digests",
        "include_private_source_text",
        "grounding_delimiters_required",
        "section_order",
    }:
        raise ConstitutionalDialoguePolicyError(
            "Prompt policy fields do not match the P3.4 contract."
        )
    max_characters = _positive_int(
        prompt["max_characters"], field="prompt.max_characters"
    )
    if not 4000 <= max_characters <= 32000:
        raise ConstitutionalDialoguePolicyError(
            "prompt.max_characters must be between 4000 and 32000."
        )
    section_order = _string_tuple(
        prompt["section_order"], field="prompt.section_order"
    )
    if section_order != _EXPECTED_SECTION_ORDER:
        raise ConstitutionalDialoguePolicyError(
            "Constitutional prompt sections cannot be reordered."
        )
    return ConstitutionalDialoguePolicy(
        policy_name="alice_constitutional_dialogue_policy",
        version=_text(root["version"], field="version"),
        phase="3",
        milestone="P3.4",
        status="implementation",
        system_contract_version=_text(
            root["system_contract_version"], field="system_contract_version"
        ),
        constitution_version=constitution_version,
        source_documents=_source_rules(root["source_documents"]),
        boundaries=_exact_boolean_map(
            root["boundaries"], field="boundaries", expected=_EXPECTED_BOUNDARIES
        ),
        trust=_exact_boolean_map(
            root["trust"], field="trust", expected=_EXPECTED_TRUST
        ),
        epistemic_labels=epistemic_labels,
        decision_hierarchy=decision_hierarchy,
        dialogue=_exact_boolean_map(
            root["dialogue"], field="dialogue", expected=_EXPECTED_DIALOGUE
        ),
        challenge_sequence=challenge_sequence,
        max_characters=max_characters,
        include_source_digests=_strict_bool(
            prompt["include_source_digests"],
            expected=False,
            field="prompt.include_source_digests",
        ),
        include_private_source_text=_strict_bool(
            prompt["include_private_source_text"],
            expected=False,
            field="prompt.include_private_source_text",
        ),
        grounding_delimiters_required=_strict_bool(
            prompt["grounding_delimiters_required"],
            expected=True,
            field="prompt.grounding_delimiters_required",
        ),
        section_order=section_order,
    )


def default_constitutional_dialogue_policy_path() -> Path:
    return Path("policies") / "conversation_constitutional_policy.json"


def load_constitutional_dialogue_policy(
    path: str | Path | None = None,
) -> ConstitutionalDialoguePolicy:
    """Load the versioned P3.4 policy from JSON."""

    selected = Path(path) if path is not None else default_constitutional_dialogue_policy_path()
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstitutionalDialoguePolicyError(
            f"Unable to load constitutional dialogue policy: {selected}"
        ) from exc
    return parse_constitutional_dialogue_policy(payload)
