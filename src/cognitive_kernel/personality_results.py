"""Link an explicitly reported result to an existing personality decision.

The result owner supplies its source-event association and exact persisted
capsule bytes. This adapter records that report, not proof of execution, causal
credit, accepted memory or N3 teaching eligibility. Payload custody, current
lifecycle authority and runtime activation remain the caller's responsibility.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .canonical import (
    CognitiveKernelContractError, canonical_sha256, normalize_timestamp,
    require_identifier, require_sha256,
)
from .contracts import ProvenanceReference
from .experience import ExperienceEvent
from .ledger import ExperienceLedgerTransactionReceipt
from .ledger_store import ExperienceLedgerStore
from .results import ResultCapsule


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CognitiveKernelContractError(reason)


def record_personality_result(
    ledger: ExperienceLedgerStore,
    *,
    decision_event_id: str,
    expected_decision_event_sha256: str,
    capsule: ResultCapsule,
    payload: bytes,
    expected_payload_sha256: str,
    payload_reference: str,
    committed_at: str,
) -> ExperienceLedgerTransactionReceipt:
    """Append a result association without rewriting the original decision.

    Persist the capsule metadata payload first, then provide its independent
    byte digest and opaque reference. JSON formatting may vary; its complete
    semantics must match the validated capsule. Repeat the same inputs on a
    retry: the existing ledger rejects the identical event. This is not a new
    cross-payload semantic-deduplication or execution-authorization service.
    """
    _require(isinstance(ledger, ExperienceLedgerStore), 'existing Experience ledger required')
    _require(isinstance(capsule, ResultCapsule), 'existing Result Capsule required')
    capsule.validate()
    require_sha256(expected_decision_event_sha256, 'expected_decision_event_sha256')
    require_sha256(expected_payload_sha256, 'expected_payload_sha256')
    _require(type(payload) is bytes and sha256(payload).hexdigest() == expected_payload_sha256,
             'result payload differs from its external digest')
    _require(isinstance(payload_reference, str) and bool(payload_reference.strip()),
             'persisted result payload reference required')
    try:
        saved = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CognitiveKernelContractError('result payload is not valid JSON') from error
    _require(type(saved) is dict and canonical_sha256(saved) == canonical_sha256(capsule.metadata_record()),
             'complete saved result metadata differs from capsule')

    decision = ledger.load_event(decision_event_id)
    _require(decision.event_sha256 == expected_decision_event_sha256,
             'stored personality decision differs from its external digest')
    _require(decision.event_type == 'personality.development_decision',
             'result must reference an existing personality decision')
    _require(capsule.scope.storage_scope() == decision.scope.storage_scope() == ledger.scope.storage_scope(),
             'result and decision product-host encryption scope differs')
    _require(decision.event_id in capsule.source_event_ids,
             'result owner must explicitly reference the personality decision')
    result_time = normalize_timestamp(capsule.produced_at, 'produced_at')
    _require(result_time >= decision.occurred_at,
             'reported result precedes its personality decision')
    for item in (decision, capsule):
        _require(require_identifier(item.storage_tier, 'storage_tier') not in ('deleted', 'quarantine')
                 and require_identifier(item.retention_class, 'retention_class') != 'quarantine',
                 'inactive decision or result cannot enter the active result link')

    references = tuple(dict.fromkeys((
        'personality-decision:' + decision.event_sha256,
        'result-capsule:' + capsule.capsule_sha256,
        *capsule.source_event_ids, *capsule.output_reference_ids, *capsule.evidence_reference_ids,
    )))
    provenance = ProvenanceReference.create(
        provenance_type='derived_inference', source_reference_ids=references,
        derivation_activity_id='personality-result-link:' + canonical_sha256(
            [decision.event_sha256, capsule.capsule_sha256]),
        responsible_component='alice-personality-result-link', model_id=None, confidence=None,
    )
    event = ExperienceEvent.create(
        event_type='personality.reported_result', scope=capsule.scope,
        occurred_at=result_time, content_digest=expected_payload_sha256,
        provenance=provenance, retention_class=capsule.retention_class, storage_tier='ledger',
        parent_event_ids=(decision.event_id,), outcome_reference_ids=(capsule.capsule_id,),
        deletion_lineage=tuple(dict.fromkeys((*decision.deletion_lineage, *capsule.deletion_lineage))),
        policy_bindings=tuple(dict.fromkeys((*decision.policy_bindings, *capsule.policy_bindings))),
        payload_reference=payload_reference,
    )
    return ledger.append_event(event, committed_at=committed_at)
