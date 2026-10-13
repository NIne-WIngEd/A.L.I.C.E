"""Owned MFM reference scoring after a complete semantic decision.

The inherited lexical decoder is unchanged. Reference keys use source states,
not labels or ordinal embeddings. Mean pooling can dilute long records; masks
are type constraints, not evidence of learned choice or grounding. The target
adapter supplies whole-record geometry and verifies actual generated token IDs;
the consumer verifies the complete marker set and performs real grounding.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from .canonical import canonical_sha256, require_identifier, require_sha256
from .formation_decision_specialist_v16 import FormationDecisionSpecialist
from .formation_v1_specialist import SpecialistConfig


TASK = 'mfm-native-reference-producer-v1'
CONDITIONING = 'completed-decision-conditioned'
_BINDING_FIELDS = {'caller_sha256', 'candidate_inventory_sha256',
                   'processed_view_sha256', 'provenance_sha256'}
_CANDIDATE_FIELDS = {'slot', 'ref_id', 'kind', 'origin', 'namespace', 'eligible_paths',
                     'meaning_token_indices', 'index_token_indices', 'independent_observation'}
_REQUEST_FIELDS = {'pointer', 'candidate_path', 'eligible_candidate_indices',
                   'query_index', 'decoder_input_index'}
_GOVERNANCE = ('proposals[].proposal.contradicts', 'proposals[].proposal.target_refs',
               'dispositions[].target_refs')
_CITATIONS = ('proposals[].proposal.evidence_refs', 'proposals[].proposal.anchors[].ref_id',
             'dispositions[].evidence_refs')
_TARGET_PATHS = {
    'entity': ('proposals[].proposal.subject_ref', 'proposals[].episode.participant_refs',
               'proposals[].relationship_counterpart_ref'),
    'scene': ('proposals[].episode.scene_refs',),
    'mission': ('proposals[].mission_target_refs',),
    'workspace': ('proposals[].workspace_target_refs',),
}
_KINDS = ('evidence', 'source_subject', 'entity', 'scene', 'mission', 'workspace',
          'claim', 'episode', 'uncertainty')
_ORIGINS = ('opened_source', 'source_subject', 'context_target', 'readonly_registry')
_PATHS = tuple(dict.fromkeys((*_GOVERNANCE, *_CITATIONS,
    *(path for paths in _TARGET_PATHS.values() for path in paths),
    'proposals[].proposal.uncertainty_ref')))
_PATH_INDEX = {path: i for i, path in enumerate(_PATHS)}
_SINGULAR_PATHS = {'proposals[].proposal.subject_ref', 'proposals[].proposal.uncertainty_ref',
                   'proposals[].relationship_counterpart_ref', 'proposals[].proposal.anchors[].ref_id'}
_ARRAY_INDEX = r'/(?:0|[1-9][0-9]*)'
_POINTER_PATTERNS = {
    path: re.compile('/' + re.escape(path).replace(r'\[\]', _ARRAY_INDEX).replace(r'\.', '/') +
                     ('' if path in _SINGULAR_PATHS else _ARRAY_INDEX))
    for path in _PATHS
}


def _binding(value):
    if type(value) is not dict or set(value) != _BINDING_FIELDS:
        raise ValueError('reference binding requires exact caller/inventory/view/provenance digests')
    for name, digest in value.items():
        if require_sha256(digest, name) != digest:
            raise ValueError('reference digest must be canonical')
    return dict(value)


def _paths(kind, origin):
    # Mirrors the existing caller role contract; custody is verified by the
    # caller/J5 binder, not reconstructed from this shortened key metadata.
    if (kind, origin) == ('evidence', 'opened_source'):
        return (*_CITATIONS, *_GOVERNANCE)
    if (kind, origin) == ('source_subject', 'source_subject'):
        return ('proposals[].proposal.subject_ref',)
    if origin == 'context_target' and kind in _TARGET_PATHS:
        return (*_TARGET_PATHS[kind], *_GOVERNANCE)
    if origin == 'readonly_registry' and kind in ('claim', 'episode', 'uncertainty', 'evidence'):
        extra = ('proposals[].proposal.uncertainty_ref',) if kind == 'uncertainty' else ()
        return (*extra, *_GOVERNANCE)
    raise ValueError('unsupported reference kind/origin role')


def _pointer_path(pointer):
    if type(pointer) is not str:
        raise ValueError('reference pointer must be concrete JSONPointer text')
    # Match actual leaves, never a template or an array container. These closed
    # schema keys need no escaped JSONPointer tokens; indices are canonical.
    for path, pattern in _POINTER_PATTERNS.items():
        if pattern.fullmatch(pointer):
            return path
    raise ValueError('unsupported concrete reference operand pointer')


def _file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class FormationReferenceSpecialist(FormationDecisionSpecialist):
    """One-case reference head on the actual full-vocabulary MFM decoder.

    Lexical ``forward`` and ``next_token_logits`` remain inherited and support
    their original batches. Reference packets bind one complete caller case.
    ``config_sha256`` pins canonical UTF-8 ``SpecialistConfig.record()`` bytes,
    not a training-receipt/config-file digest. Actual safetensors bytes are
    separately pinned by ``parent_sha256``.
    """

    def __init__(self, config: SpecialistConfig):
        super().__init__(config)
        self._inherited_keys = tuple(self.state_dict())
        width = config.width
        self.reference_meaning_key = nn.Linear(width, width, bias=False)
        self.reference_index_key = nn.Linear(width, width, bias=False)
        self.reference_kind = nn.Embedding(len(_KINDS), width)
        self.reference_origin = nn.Embedding(len(_ORIGINS), width)
        self.reference_path = nn.Embedding(len(_PATHS), width)
        self.reference_query = nn.Linear(2 * width, width)
        self.parent_binding = None

    @classmethod
    def from_parent_safetensors(cls, config, path, *, parent_sha256, config_sha256):
        model = cls(config)
        model.load_parent_safetensors(path, parent_sha256=parent_sha256,
                                      config_sha256=config_sha256)
        return model

    def load_parent_safetensors(self, path, *, parent_sha256, config_sha256):
        """Validate every inherited tensor before changing any model state.

        Existing reference tensors are retained, with zero migration updates.
        This is a parent migration, never a permissive new-task reload.
        """
        for value, name in ((parent_sha256, 'parent_sha256'), (config_sha256, 'config_sha256')):
            if require_sha256(value, name) != value:
                raise ValueError('parent/config pins must be canonical')
        if canonical_sha256(self.config.record()) != config_sha256:
            raise ValueError('parent config canonical-byte binding mismatch')
        if _file_sha256(path) != parent_sha256:
            raise ValueError('parent safetensors FILE binding mismatch')
        try:
            parent = load_file(str(path), device='cpu')
        except Exception as exc:
            raise ValueError('parent safetensors cannot be read') from exc
        current = self.state_dict()
        if set(parent) != set(self._inherited_keys):
            raise ValueError('parent inherited tensor names differ')
        for name, tensor in parent.items():
            if tensor.shape != current[name].shape or tensor.dtype != torch.float32:
                raise ValueError(f'parent tensor shape/FP32 contract differs: {name}')
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f'parent tensor is non-finite: {name}')
        if _file_sha256(path) != parent_sha256:
            raise ValueError('parent safetensors changed during validation')
        self.load_state_dict({**current, **parent}, strict=True)
        self.parent_binding = {
            'parent_weight_file_sha256': parent_sha256,
            'parent_config_canonical_sha256': config_sha256,
            'new_task': TASK,
            'reference_initialization': 'fresh-native-typed-reference-head-v1',
            'reference_updates_performed_by_migration': 0,
        }
        return dict(self.parent_binding)

    def _source(self, base_states, source_mask):
        if (not isinstance(base_states, torch.Tensor) or base_states.ndim != 3 or
                base_states.shape[0] != 1 or base_states.shape[2] != self.config.base_hidden_size or
                not base_states.shape[1] or not base_states.is_floating_point()):
            raise ValueError('reference source requires one complete floating-point case')
        if (not isinstance(source_mask, torch.Tensor) or
                source_mask.shape != base_states.shape[:2] or
                not torch.all((source_mask == 0) | (source_mask == 1)).item() or
                not source_mask.any().item() or not torch.isfinite(base_states).all().item()):
            raise ValueError('reference source mask/states are invalid')
        if base_states.device != self.source_projection.weight.device or source_mask.device != base_states.device:
            raise ValueError('reference source and model must share a device')
        # Only owned source_projection is trainable, never the supplied backbone.
        return base_states.detach()

    def decision_states(self, *, base_states, source_mask, input_ids, binding, mode):
        """Decode [BOS, *complete body IDs] with no labels or authored choices.

        Final input position predicts EOS and supplies complete-record context.
        A generated packet requires geometry from the actual generated IDs;
        teacher-forced packets are explicitly separate and may form ordinal loss.
        """
        bound = _binding(binding)
        if mode not in ('generated', 'teacher_forced'):
            raise ValueError('reference states require an explicit generated/teacher_forced mode')
        source = self._source(base_states, source_mask)
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or
                input_ids.shape[0] != 1 or input_ids.shape[1] < 2 or
                input_ids.dtype != torch.long or input_ids.device != source.device or
                not torch.all((input_ids >= 0) & (input_ids < self.config.vocabulary_size)).item() or
                input_ids[0, 0].item() != self.config.start_token_id or
                input_ids.eq(self.config.pad_token_id).any().item()):
            raise ValueError('reference decoder inputs require complete unpadded whole-record IDs')
        hidden = self._decode(base_states=source, source_mask=source_mask, input_ids=input_ids)
        return {'binding': bound, 'hidden': hidden, 'input_ids': input_ids.detach().clone(),
                'mode': mode, 'completion_index': input_ids.shape[1] - 1,
                'conditioning': CONDITIONING}

    def candidate_keys(self, *, base_states, source_mask, candidates, binding):
        """Pool unique meaning/index positions independently, retaining global slots."""
        bound = _binding(binding)
        source = self._source(base_states, source_mask)
        if type(candidates) is not list:
            raise ValueError('reference candidates must be an explicit ordered array')
        rows, indices, roles, slots = [], [], set(), set()
        for row in candidates:
            if type(row) is not dict or set(row) != _CANDIDATE_FIELDS:
                raise ValueError('reference candidate differs from the J5 join contract')
            if type(row['slot']) is not int or row['slot'] < 0 or row['slot'] in slots:
                raise ValueError('reference candidate global slot is invalid or duplicated')
            slots.add(row['slot'])
            for field in ('ref_id', 'namespace'):
                if require_identifier(row[field], field) != row[field]:
                    raise ValueError('reference candidate identity must be canonical')
            if type(row['kind']) is not str or type(row['origin']) is not str:
                raise ValueError('reference kind/origin must be explicit typed strings')
            expected_paths = list(_paths(row['kind'], row['origin']))
            if row['eligible_paths'] != expected_paths or row['independent_observation'] is not False:
                raise ValueError('reference candidate role/path contract differs')
            role = (row['ref_id'], row['kind'], row['origin'], row['namespace'])
            if role in roles:
                raise ValueError('duplicate reference identity within the same typed role')
            roles.add(role)
            pooled_indices = []
            for field in ('meaning_token_indices', 'index_token_indices'):
                values = row[field]
                if (type(values) is not list or not values or
                        any(type(i) is not int or not 0 <= i < source.shape[1] for i in values)):
                    raise ValueError('reference key needs complete valid source indices')
                unique = sorted(set(values))
                if not source_mask[0, unique].bool().all().item():
                    raise ValueError('reference key index points at a masked source token')
                pooled_indices.append(unique)
            indices.append(pooled_indices)
            rows.append(deepcopy(row))
        if slots != set(range(len(rows))):
            raise ValueError('reference keys must preserve every caller inventory slot')
        projected = self.source_projection(source.to(self.source_projection.weight.dtype))[0]
        keys = []
        for row, (meaning, index) in zip(rows, indices):
            path_ids = torch.tensor([_PATH_INDEX[path] for path in row['eligible_paths']], device=source.device)
            key = (self.reference_meaning_key(projected[meaning].mean(0)) +
                   self.reference_index_key(projected[index].mean(0)) +
                   self.reference_kind.weight[_KINDS.index(row['kind'])] +
                   self.reference_origin.weight[_ORIGINS.index(row['origin'])] +
                   self.reference_path(path_ids).mean(0))
            keys.append(key)
        tensor = torch.stack(keys) if keys else projected.new_empty((0, self.config.width))
        return {'binding': bound, 'keys': tensor, 'slots': [row['slot'] for row in rows], 'candidates': rows}

    def reference_logits(self, hidden, requests, keys, eligibility):
        """Score physical key columns; masks and global ordinals must agree exactly."""
        if _binding(hidden['binding']) != _binding(keys['binding']):
            raise ValueError('reference states and keys have stale or foreign bindings')
        states, candidate_keys = hidden['hidden'], keys['keys']
        if (hidden['mode'] not in ('generated', 'teacher_forced') or
                hidden['conditioning'] != CONDITIONING or states.ndim != 3 or
                states.shape[0] != 1 or states.shape[2] != self.config.width or
                hidden['completion_index'] != states.shape[1] - 1 or
                candidate_keys.shape != (len(keys['candidates']), self.config.width) or
                keys['slots'] != [row['slot'] for row in keys['candidates']]):
            raise ValueError('invalid reference state/key packet geometry')
        if (type(requests) is not list or not isinstance(eligibility, torch.Tensor) or
                eligibility.dtype != torch.bool or
                eligibility.shape != (len(requests), len(keys['slots'])) or
                eligibility.device != candidate_keys.device or states.device != candidate_keys.device):
            raise ValueError('reference eligibility must match every request and physical key column')
        seen, queries = set(), []
        for i, request in enumerate(requests):
            if type(request) is not dict or set(request) != _REQUEST_FIELDS:
                raise ValueError('reference request has missing or unsupported fields')
            pointer, path = request['pointer'], request['candidate_path']
            if _pointer_path(pointer) != path or pointer in seen:
                raise ValueError('reference request pointer/path is wrong or duplicated')
            seen.add(pointer)
            expected = sorted(row['slot'] for row in keys['candidates'] if path in row['eligible_paths'])
            declared = request['eligible_candidate_indices']
            if (type(declared) is not list or any(type(slot) is not int for slot in declared) or
                    not expected or declared != expected):
                raise ValueError('mandatory reference request needs the complete typed candidate set')
            mask = torch.tensor([slot in expected for slot in keys['slots']], device=eligibility.device)
            if not torch.equal(mask, eligibility[i]):
                raise ValueError('reference mask differs from typed global candidate eligibility')
            j, operand = request['query_index'], request['decoder_input_index']
            if (type(j) is not int or type(operand) is not int or j < 0 or
                    operand != j + 1 or operand >= states.shape[1]):
                raise ValueError('reference query must predict operand j+1 at predictor j')
            completed = states[0, hidden['completion_index']]
            query = self.reference_query(torch.cat((states[0, j], completed)))
            queries.append(query + self.reference_path.weight[_PATH_INDEX[path]])
        if not queries:
            return candidate_keys.new_empty((0, len(keys['slots'])))
        logits = torch.stack(queries) @ candidate_keys.T / self.config.width ** .5
        if not torch.isfinite(logits).all().item():
            raise ValueError('reference scoring produced non-finite logits')
        return logits.masked_fill(~eligibility, -torch.inf)

    def predict_references(self, hidden, requests, keys, eligibility):
        """Return only bound concrete pointers and predicted global ordinals."""
        if hidden['mode'] != 'generated':
            raise ValueError('reference predictions require actual generated-decision states')
        logits = self.reference_logits(hidden, requests, keys, eligibility)
        # Exact ties resolve by global ordinal, independent of key row order.
        order = sorted(range(len(keys['slots'])), key=lambda column: keys['slots'][column])
        columns = [order[i] for i in logits[:, order].argmax(1).tolist()] if requests else []
        return {'binding': dict(hidden['binding']), 'rows': [
            {'pointer': request['pointer'], 'candidate_ordinal': keys['slots'][column]}
            for request, column in zip(requests, columns)]}

    def selection_loss(self, hidden, requests, keys, eligibility, *, authored_ordinals):
        """Mean CE over explicitly authored integer choices, with no inferred negatives."""
        if hidden['mode'] != 'teacher_forced':
            raise ValueError('ordinal supervision requires a separate teacher-forced decision')
        logits = self.reference_logits(hidden, requests, keys, eligibility)
        if (not requests or type(authored_ordinals) is not list or
                len(authored_ordinals) != len(requests)):
            raise ValueError('selection loss requires an explicit choice for every request')
        authored = {}
        for row in authored_ordinals:
            if (type(row) is not dict or set(row) != {'pointer', 'candidate_ordinal'} or
                    type(row['pointer']) is not str or row['pointer'] in authored or
                    type(row['candidate_ordinal']) is not int):
                raise ValueError('authored selection must be an unambiguous integer ordinal')
            authored[row['pointer']] = row['candidate_ordinal']
        if set(authored) != {request['pointer'] for request in requests}:
            raise ValueError('authored selection pointer set differs from reference requests')
        targets = []
        for request in requests:
            ordinal = authored[request['pointer']]
            if ordinal not in request['eligible_candidate_indices']:
                raise ValueError('authored ordinal is not a typed eligible caller candidate')
            targets.append(keys['slots'].index(ordinal))
        return F.cross_entropy(logits.float(), torch.tensor(targets, device=logits.device))

    def generate_reference_decision(self, *, processor, base_states, source_mask, caller,
            caller_sha256, joint_alignment, joint_alignment_sha256, expected_binding,
            artifact_sha256, inference_run_id, max_new_tokens):
        """Own lexical generation, exact generated geometry, reference head and grounding.

        ``max_new_tokens`` counts choices including EOS, within the unchanged
        full target budget. Source-feature admission is the original reader's
        responsibility. This full-reference FP32 route makes no cache/speed or
        learned-behavior claim. Failed outputs remain verbatim, never repaired.
        """
        from . import formation_reference_decision as consumer
        from scripts.mfm import formation_reference_target_alignment as alignment
        from scripts.mfm import formation_source_alignment as source_api

        result = {'generated_token_ids': [], 'raw_output_text': '', 'eos_observed': False,
                  'validation_status': 'invalid', 'validation_error': None,
                  'failure_stage': None, 'reference_predictions': None, 'bundle': None}
        stage, raw_decoded = 'preflight', False
        modes = [(module, module.training) for module in self.modules()]
        device = self.source_projection.weight.device
        devices = [device.index] if device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), \
                    torch.autocast(device.type, enabled=False):
                if (device.type not in ('cpu', 'cuda') or
                        any(parameter.dtype != torch.float32 or parameter.device != device
                            for parameter in self.parameters())):
                    raise ValueError('native reference inference requires one CPU/CUDA device and FP32 owned weights')
                self.config.validate()
                if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.config.max_target_tokens:
                    raise ValueError('native reference inference budget exceeds full target geometry')
                for value, name in ((artifact_sha256, 'artifact_sha256'), (caller_sha256, 'caller_sha256'),
                                    (joint_alignment_sha256, 'joint_alignment_sha256')):
                    if require_sha256(value, name) != value:
                        raise ValueError('native inference pins must be canonical')
                if require_identifier(inference_run_id, 'inference_run_id') != inference_run_id:
                    raise ValueError('native inference run identity must be canonical')
                inventory = consumer.bind_reference_inventory(caller, joint_alignment,
                    caller_sha256=caller_sha256, joint_alignment_sha256=joint_alignment_sha256,
                    expected_binding=expected_binding)
                if source_api.processor_binding(processor) != inventory['processor_binding']:
                    raise ValueError('native inference processor binding differs')
                tokenizer = processor.tokenizer
                if (len(tokenizer) != self.config.vocabulary_size or
                        tokenizer.bos_token_id != self.config.start_token_id or
                        tokenizer.eos_token_id != self.config.end_token_id or
                        tokenizer.pad_token_id != self.config.pad_token_id):
                    raise ValueError('native inference processor/config geometry differs')
                source = self._source(base_states, source_mask)
                if (source.shape[1] != len(joint_alignment['input_ids']) or
                        source_mask[0].tolist() != joint_alignment['attention_mask']):
                    raise ValueError('native inference requires the complete bound source length/mask')
                context, opened, _ = consumer.caller_api.caller_input_from_record(caller)
                self.eval()
                stage = 'generation'
                prefix = [self.config.start_token_id]
                for _ in range(max_new_tokens):
                    inputs = torch.tensor([prefix], dtype=torch.long, device=device)
                    logits = self.next_token_logits(base_states=source, source_mask=source_mask, input_ids=inputs)
                    if (logits.shape != (1, self.config.vocabulary_size) or
                            not torch.isfinite(logits).all().item()):
                        raise ValueError('native lexical logits have invalid geometry or nonfinite values')
                    token = int(logits[0].argmax().item())
                    if token == self.config.end_token_id:
                        result['eos_observed'] = True
                        break
                    result['generated_token_ids'].append(token)
                    prefix.append(token)
                # Decode even a capped prefix; preserve the first failure stage.
                if not result['eos_observed']:
                    raise ValueError('no EOS inside the declared native inference budget')
                stage = 'decode'
                result['raw_output_text'] = tokenizer.decode(result['generated_token_ids'],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
                raw_decoded = True
                stage = 'parse'
                def unique(pairs):
                    record = {}
                    for key, value in pairs:
                        if key in record:
                            raise ValueError('duplicate generated JSON key')
                        record[key] = value
                    return record
                def nonfinite(value):
                    raise ValueError('nonfinite generated JSON constant: ' + value)
                decision = json.loads(result['raw_output_text'], object_pairs_hook=unique, parse_constant=nonfinite)
                stage = 'alignment'
                geometry = alignment.align_generated_reference_decision(processor, decision,
                    result['generated_token_ids'], inventory, max_target_tokens=self.config.max_target_tokens,
                    eos_observed=True)
                stage = 'reference'
                hidden = self.decision_states(base_states=source, source_mask=source_mask,
                    input_ids=torch.tensor([geometry['decoder_inputs']], dtype=torch.long, device=device),
                    binding=inventory['binding'], mode='generated')
                keys = self.candidate_keys(base_states=source, source_mask=source_mask,
                    candidates=inventory['candidates'], binding=inventory['binding'])
                requests = geometry['requests']
                eligibility = torch.tensor([[slot in request['eligible_candidate_indices'] for slot in keys['slots']]
                    for request in requests], dtype=torch.bool, device=device).reshape(len(requests), len(keys['slots']))
                result['reference_predictions'] = self.predict_references(hidden, requests, keys, eligibility)
                stage = 'compile_ground'
                result['bundle'] = consumer.compile_reference_decision(context, opened, decision,
                    result['reference_predictions'], caller=caller, caller_sha256=caller_sha256,
                    joint_alignment=joint_alignment, joint_alignment_sha256=joint_alignment_sha256,
                    expected_binding=expected_binding, artifact_sha256=artifact_sha256,
                    inference_run_id=inference_run_id)
                result['validation_status'] = 'grounded_proposal_only'
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError) as exc:
            result['failure_stage'] = stage
            result['validation_error'] = f'{type(exc).__name__}: {exc}'
            if stage == 'decode':
                # This operation already failed; IDs/EOS remain available.
                result['raw_output_text'] = None
            elif not raw_decoded and stage == 'generation':
                try:
                    # Failure evidence is inference work too. The main scope
                    # has restored caller RNG; retain that state during decode.
                    with torch.random.fork_rng(devices=devices), torch.inference_mode(), \
                            torch.autocast(device.type, enabled=False):
                        result['raw_output_text'] = processor.tokenizer.decode(result['generated_token_ids'],
                            skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except (ValueError, TypeError, AttributeError, RuntimeError) as decode_error:
                    result['raw_output_text'] = None
                    result['raw_decode_error'] = f'{type(decode_error).__name__}: {decode_error}'
        finally:
            for module, training in modes:
                module.training = training
        return result
