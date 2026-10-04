"""Explicit fictional prior-memory inputs, authored independently of target reads.

These are training-world snapshots, not claims about a real person or a live
memory service. Their IDs and alternatives are source input. Stored statements
remain stored claims; a current independent gate decides changes. No expected
action, target label, teacher answer or write permission enters this snapshot.
"""
from dataclasses import replace
from hashlib import sha256

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError
from cognitive_kernel.formation_contracts import FormationEvidenceRef
from cognitive_kernel.formation_semantics_v16 import (
    RegisteredSourceSensitivity, bundle_v16_from_output,
)

SCHEMA = 'mfm-readonly-prior-memory-snapshot-v1'
# Explicit fictional authoring declaration, not inferred from invisibility.
PUBLIC_NEW_UNCERTAINTY = {'v16-seed-18': ('v16-seed-18-interview-date-unresolved',)}


def local_uncertainty_allocations(example):
    return PUBLIC_NEW_UNCERTAINTY.get(example.case_id, ()) if any(
        p.base.uncertainty_ref in PUBLIC_NEW_UNCERTAINTY.get(example.case_id, ())
        for p in example.target.proposals) else ()

# Independently authored inventory for the three incomplete public seed inputs.
# Multiple candidate identities avoid making registration a sole-answer cue.
PUBLIC_SEED_PRIOR_MEMORY = {
    'v16-seed-06': (
        ('calendar-meeting-claim', 'A stored calendar claim says the owner met Sol at 14:00 on February 21.'),
        ('calendar-location-claim', 'A stored calendar record names the north room as the planned meeting location.'),
    ),
    'v16-seed-11': (
        ('current-study-preference', 'An unscoped stored claim says the owner always studies with music, including January.'),
        ('quiet-study-preference', 'An unscoped stored claim says the owner always prefers quiet tables, including March.'),
        ('study-window-preference', 'A separate preference record concerns studying in the afternoon.'),
    ),
    'v16-seed-18': (
        ('archive-interview-date', 'A stored archive record assigns a date to the interview; its reliability is unresolved.'),
        ('archive-visit-date', 'A separate stored archive record assigns a date to a visit, rather than the interview.'),
    ),
}


def attach_prior_memory_snapshot(example, entries):
    """Supply an already-authorized read snapshot as registered structured input.

    A production caller must obtain entries from its governed read service.
    This function grants no read/write authorization and resolves no policies.
    The snapshot is context evidence, never added to new experience references.
    """
    example.validate()
    if not entries or any(type(entry) not in (tuple, list) or len(entry) != 2 or any(
            type(value) is not str or not value.strip() for value in entry) for entry in entries) or len({entry[0] for entry in entries}) != len(entries):
        raise CognitiveKernelContractError('prior-memory entries are empty or duplicated')
    context = example.context
    base = context.base
    ref_id = base.scope.host_instance_id + '-prior-memory-snapshot'
    if ref_id in {ref.ref_id for ref in base.evidence}:
        raise CognitiveKernelContractError('prior-memory source already exists')
    payload = canonical_json_bytes({
        'schema': SCHEMA, 'scope': base.scope.metadata_record(),
        'authority_namespace_id': base.authority_namespace_id,
        'snapshot_only': True, 'entries': [
            {'target_ref': name, 'kind': 'claim', 'stored_value_text': text}
            for name, text in entries],
    })
    observed = max((ref.recorded_at or ref.observed_at for ref in base.evidence
                    if ref.recorded_at or ref.observed_at), default=None)
    source = FormationEvidenceRef(
        ref_id=ref_id, scope=base.scope, authority_namespace_id=base.authority_namespace_id,
        content_digest=sha256(payload).hexdigest(), role='tool_or_action_observation',
        modality='structured', speaker_ref='registered-memory-read-service',
        observed_at=observed, recorded_at=observed)
    context = replace(context, base=replace(base, evidence=(*base.evidence, source)),
        source_sensitivities=(*context.source_sensitivities, RegisteredSourceSensitivity(ref_id, 'private')))
    target = bundle_v16_from_output(context, example.target.output_record(),
        artifact_sha256=example.target.base.model_artifact_digest,
        inference_run_id=example.target.base.inference_run_id)
    result = replace(example, context=context, opened_sources=(*example.opened_sources, (ref_id, payload)), target=target)
    result.validate()
    return result


def augment_public_seed_example(example):
    entries = PUBLIC_SEED_PRIOR_MEMORY.get(example.case_id)
    return attach_prior_memory_snapshot(example, entries) if entries is not None else example
