"""Record native developmental personality decisions in existing Experience.

This consumes an externally pinned, already persisted governed-response payload.
The caller owns payload custody/resolution and the source's product/host scope.
An event records decision construction, not performed action, observed success,
historical truth, owner acceptance, or N3 training eligibility.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .canonical import CognitiveKernelContractError, canonical_sha256, require_sha256
from .contracts import ProductHostScope, ProvenanceReference
from .experience import ExperienceEvent
from .ledger import ExperienceLedgerTransactionReceipt
from .ledger_store import ExperienceLedgerStore


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CognitiveKernelContractError(reason)


def _sealed(value: object, schema: str) -> dict:
    _require(type(value) is dict and value.get('schema') == schema,
             'personality response schema differs')
    _require(value.get('receipt_sha256') == canonical_sha256(
        {key: item for key, item in value.items() if key != 'receipt_sha256'}),
        'personality response seal differs')
    _require(value.get('qualification') == 'UNQUALIFIED'
             and all(value.get(key) is False for key in (
                 'acceptance_authority', 'owner_fidelity_approved', 'spoken_fidelity_verified')),
             'developmental personality receipt cannot grant authority')
    return value


def record_personality_decision(
    ledger: ExperienceLedgerStore,
    *,
    payload: bytes,
    expected_payload_sha256: str,
    payload_reference: str,
    scope: ProductHostScope,
    invocation_id: str,
    occurred_at: str,
    committed_at: str,
    policy_bindings: tuple[str, ...],
    parent_event_ids: tuple[str, ...] = (),
    deletion_lineage: tuple[str, ...] = (),
) -> ExperienceLedgerTransactionReceipt:
    """Append one complete decision's metadata through the existing writer.

    Persist the exact payload first and supply its independent digest and opaque
    reference. This function verifies those bytes, not the reference's backing
    store or the source owner's authorization. It never re-runs the model or
    changes its state. Retain invocation/time on retries: duplicate events are
    rejected by the existing ledger, not rewritten as new experiences.
    """
    _require(isinstance(ledger, ExperienceLedgerStore), 'existing Experience ledger required')
    _require(isinstance(scope, ProductHostScope), 'explicit source scope required')
    scope.validate()
    _require(type(payload) is bytes, 'personality payload must be exact saved bytes')
    require_sha256(expected_payload_sha256, 'expected_payload_sha256')
    _require(sha256(payload).hexdigest() == expected_payload_sha256,
             'personality payload differs from its external digest')
    _require(isinstance(payload_reference, str) and bool(payload_reference.strip()),
             'persisted personality payload reference required')
    try:
        response = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CognitiveKernelContractError('personality payload is not valid JSON') from error
    response = _sealed(response, 'alice-personality-governed-response-v1')
    selected = _sealed(response.get('preference_selection'), 'alice-personality-response-construction-v1')
    _require(response.get('packet_sha256') == selected.get('packet_sha256'),
             'governed and native packet identity differs')
    _require(selected.get('training_state') == 'OPTIMIZED_UNQUALIFIED'
             and type(response.get('decision')) is dict and type(response.get('output_scope')) is dict,
             'complete native developmental decision and teaching scope required')

    references = ['personality-response:' + response['receipt_sha256'],
                  'personality-preference-selection:' + selected['receipt_sha256']]
    for field in ('document_sha256', 'frame_sha256', 'feature_binding_sha256',
                  'closed_session_sha256', 'packet_sha256'):
        digest = require_sha256(selected.get(field), field)
        references.append('personality-' + field.removesuffix('_sha256').replace('_', '-') + ':' + digest)
    model_sha = require_sha256(selected.get('model_state_sha256'), 'model_state_sha256')
    checkpoints = selected.get('checkpoint_manifest_sha256')
    _require(type(checkpoints) is list and bool(checkpoints), 'actual native checkpoint lineage required')
    for digest in checkpoints:
        references.append('personality-checkpoint:' + require_sha256(digest, 'checkpoint_manifest_sha256'))
    provenance = ProvenanceReference.create(
        provenance_type='derived_inference',
        source_reference_ids=tuple(dict.fromkeys(references)),
        derivation_activity_id=invocation_id,
        responsible_component='alice-personality',
        model_id='personality-model:' + model_sha,
        confidence=None,
    )
    event = ExperienceEvent.create(
        event_type='personality.development_decision', scope=scope,
        occurred_at=occurred_at, content_digest=expected_payload_sha256,
        provenance=provenance, retention_class='active_project', storage_tier='ledger',
        parent_event_ids=parent_event_ids, deletion_lineage=deletion_lineage,
        outcome_reference_ids=(), policy_bindings=policy_bindings,
        payload_reference=payload_reference,
    )
    # The existing writer owns storage isolation: product, host and encryption
    # domain. A record schema version is not a different storage identity.
    return ledger.append_event(event, committed_at=committed_at)
