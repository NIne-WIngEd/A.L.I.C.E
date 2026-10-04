"""Learn memory decisions; allocate artifact-local bookkeeping outside the model.

This separately versioned output does not alter the historical v1.6 codec.
Evidence, existing-memory references, subjects, times and all other semantic
fields remain learned. Compilation assigns only new local proposal/value/
uncertainty identifiers; it neither repairs content nor authorizes a write.
Existing uncertainty identities stay literal. Only caller-declared new local
uncertainty identities are allocated, with sharing preserved across references.
"""
from copy import deepcopy

from .canonical import CognitiveKernelContractError, canonical_sha256
from .formation_learning_v16 import supervised_output_record_v16
from .formation_semantics_v16 import (
    OUTPUT_SCHEMA_V16, bundle_v16_from_output, validate_formation_grounding_v16,
)

SCHEMA = 'mfm-formation-decision-v1.6-local-ids-v1'
OBJECTIVE = 'mfm-grounded-full-role-decisions-v1.6-local-ids-v1'
LOCAL_FIELDS = ('proposal_id', 'value_ref')


def _refs(values, names):
    return [deepcopy(names.get(value, value)) for value in values]


def decision_from_bundle(bundle, *, local_uncertainty_refs=()):
    output = deepcopy(bundle.output_record())
    names = {}
    for index, item in enumerate(output['proposals'], 1):
        for field, tag in (('proposal_id', 'local_proposal'), ('value_ref', 'local_value')):
            name = item['proposal'][field]
            if name in names:
                raise CognitiveKernelContractError('ambiguous local decision reference')
            names[name] = {tag: index}
    declared = set(local_uncertainty_refs)
    if len(declared) != len(local_uncertainty_refs) or any(type(v) is not str or not v for v in declared):
        raise CognitiveKernelContractError('local uncertainty allocation declarations differ')
    allocated = {}
    for item in output['proposals']:
        value = item['proposal']['uncertainty_ref']
        if value in declared and value not in allocated:
            if value in names:
                raise CognitiveKernelContractError('ambiguous local uncertainty reference')
            allocated[value] = {'local_uncertainty': len(allocated) + 1}
    if set(allocated) != declared:
        raise CognitiveKernelContractError('declared new uncertainty is not proposed')
    names.update(allocated)
    for item in output['proposals']:
        proposal = item['proposal']
        proposal['uncertainty_ref'] = deepcopy(names.get(proposal['uncertainty_ref'], proposal['uncertainty_ref']))
        for field in LOCAL_FIELDS:
            proposal.pop(field)
        for field in ('contradicts', 'target_refs'):
            proposal[field] = _refs(proposal[field], names)
    for item in output['dispositions']:
        item['target_refs'] = _refs(item['target_refs'], names)
    output['schema'] = SCHEMA
    return output


def decision_record(example, *, local_uncertainty_refs=()):
    # Retain the existing complete-adjudication check, including unknown labels.
    supervised_output_record_v16(example)
    return decision_from_bundle(example.target, local_uncertainty_refs=local_uncertainty_refs)


def compile_decision(context, decision, *, artifact_sha256, inference_run_id):
    if not isinstance(decision, dict) or set(decision) != {'schema', 'proposals', 'dispositions'} or decision['schema'] != SCHEMA:
        raise CognitiveKernelContractError('unsupported learned decision output')
    if type(decision['proposals']) is not list or type(decision['dispositions']) is not list:
        raise CognitiveKernelContractError('decision proposals/dispositions must be arrays')
    output = deepcopy(decision)
    prefix = 'mfm-' + canonical_sha256({'context': context.content_digest(), 'run': inference_run_id})[:24]
    count = len(output['proposals'])
    uncertainty_slots = set()
    for item in output['proposals']:
        if not isinstance(item, dict) or not isinstance(item.get('proposal'), dict):
            raise CognitiveKernelContractError('decision proposal must be an object')
        ref = item['proposal'].get('uncertainty_ref')
        if isinstance(ref, dict):
            if set(ref) != {'local_uncertainty'} or type(ref['local_uncertainty']) is not int or ref['local_uncertainty'] < 1:
                raise CognitiveKernelContractError('invalid local uncertainty reference')
            uncertainty_slots.add(ref['local_uncertainty'])
        elif ref is not None and (type(ref) is not str or not ref):
            raise CognitiveKernelContractError('invalid existing uncertainty identity')
    if uncertainty_slots != set(range(1, len(uncertainty_slots) + 1)):
        raise CognitiveKernelContractError('new uncertainty slots must be contiguous')

    def restore_refs(values):
        if type(values) is not list:
            raise CognitiveKernelContractError('decision references must be arrays')
        restored = []
        for value in values:
            if isinstance(value, dict):
                if len(value) != 1 or next(iter(value)) not in {'local_proposal', 'local_value', 'local_uncertainty'}:
                    raise CognitiveKernelContractError('invalid local decision reference')
                kind, index = next(iter(value.items()))
                limit = len(uncertainty_slots) if kind == 'local_uncertainty' else count
                if type(index) is not int or not 1 <= index <= limit:
                    raise CognitiveKernelContractError('local decision reference is out of range')
                value = f'{prefix}-' + {'local_proposal': 'p', 'local_value': 'v', 'local_uncertainty': 'u'}[kind] + f'-{index}'
            restored.append(value)
        return restored

    for index, item in enumerate(output['proposals'], 1):
        proposal = item['proposal']
        if not isinstance(proposal, dict) or any(field in proposal for field in LOCAL_FIELDS):
            raise CognitiveKernelContractError('model cannot supply local bookkeeping IDs')
        proposal['proposal_id'] = f'{prefix}-p-{index}'
        proposal['value_ref'] = f'{prefix}-v-{index}'
        # Preserve literal existing pointers; resolve only explicitly typed locals.
        if 'uncertainty_ref' not in proposal:
            raise CognitiveKernelContractError('decision lacks explicit uncertainty reference')
        proposal['uncertainty_ref'] = restore_refs([proposal['uncertainty_ref']])[0]
        for field in ('contradicts', 'target_refs'):
            proposal[field] = restore_refs(proposal.get(field))
    for item in output['dispositions']:
        if not isinstance(item, dict):
            raise CognitiveKernelContractError('decision disposition must be an object')
        item['target_refs'] = restore_refs(item.get('target_refs'))
    output['schema'] = OUTPUT_SCHEMA_V16
    # The original strict parser rejects extra/missing fields and invalid values.
    return bundle_v16_from_output(context, output, artifact_sha256=artifact_sha256,
                                  inference_run_id=inference_run_id)
