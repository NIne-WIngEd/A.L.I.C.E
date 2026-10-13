"""One-packet native lexical/reference/original-copy learning mechanics.

Caller admission and source/target rights belong to the original material
reader. This numerical component neither admits a corpus nor qualifies a fit.
"""
from __future__ import annotations

from copy import deepcopy
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file

from .canonical import canonical_json_bytes, canonical_sha256, require_identifier, require_sha256
from . import formation_original_copy_decision as copies
from . import formation_original_copy_specialist as copy_model
from . import formation_reference_specialist as reference_model
from . import formation_v1_specialist as lexical_model
from .formation_original_copy_specialist import FormationOriginalCopySpecialist, _file_sha256
from scripts.mfm import formation_original_copy_target_alignment as target
from scripts.mfm import formation_source_alignment as source_alignment


TASK = 'mfm-full-formation-lexical-reference-original-copy-v1'
COMPONENT_SCHEMA = 'mfm-native-joint-component-v1'


def _read_json(path):
    def unique(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError('duplicate component JSON key')
            record[key] = value
        return record
    def reject_constant(value):
        raise ValueError('nonfinite component JSON value: ' + value)
    return json.loads(Path(path).read_bytes().decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=reject_constant)


def _interface_pins():
    """Hash the local Python import closure, including executed package inits.

This is a source/interface closure, not a dependency-version or data-rights
claim. External package versions belong to the separately pinned runtime.
    """
    root = Path(__file__).resolve().parents[2]
    roots = (root / 'src', root)

    def module_file(name):
        parts = name.split('.')
        if not parts or any(not part.isidentifier() for part in parts):
            return None
        for base in roots:
            path = base.joinpath(*parts)
            for candidate in (path.with_suffix('.py'), path / '__init__.py'):
                if candidate.is_file():
                    return candidate
        return None

    pending = ['cognitive_kernel.formation_joint_learning']
    observed = {}
    while pending:
        name = pending.pop()
        path = module_file(name)
        if path is None or name in observed:
            continue
        observed[name] = path
        for length in range(1, len(name.split('.'))):
            parent = '.'.join(name.split('.')[:length])
            parent_path = module_file(parent)
            if parent_path is not None and parent_path.name == '__init__.py':
                pending.append(parent)
        package = name.split('.') if path.name == '__init__.py' else name.split('.')[:-1]
        tree = ast.parse(path.read_bytes().decode('utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        continue
                    parts = package[:len(package)-node.level+1]
                    if node.module:
                        parts += node.module.split('.')
                    base_name = '.'.join(parts)
                else:
                    base_name = node.module or ''
                if base_name:
                    pending.append(base_name)
                    pending.extend(base_name + '.' + alias.name for alias in node.names if alias.name != '*')
    return {path.relative_to(root).as_posix(): _file_sha256(path)
            for path in sorted(set(observed.values()))}


class FormationJointLearning(FormationOriginalCopySpecialist):
    """Exact complete-wire operation on the existing own decoder and heads."""

    def joint_loss(self, *, processor, base_states, source_mask, inventory, decision,
                   alignment, alignment_sha256, input_ids, labels, authored_ordinals,
                   authored_copy_tuples, target_view, expected_view_sha256):
        copies._inventory(inventory)
        tokenizer = processor.tokenizer
        if (len(tokenizer) != self.config.vocabulary_size or
                tokenizer.bos_token_id != self.config.start_token_id or
                tokenizer.eos_token_id != self.config.end_token_id or
                tokenizer.pad_token_id != self.config.pad_token_id):
            raise ValueError('joint training tokenizer/model geometry differs from serving')
        require_sha256(expected_view_sha256, 'expected_view_sha256')
        target._view(target_view, expected_view_sha256)
        if (canonical_sha256(target_view['decision']) != canonical_sha256(decision) or
                canonical_sha256(target_view['binding']) != canonical_sha256(inventory['reference_inventory']['binding']) or
                canonical_sha256(target_view['copy_binding']) != canonical_sha256(copies.copy_binding(inventory)) or
                canonical_sha256(target_view['processor_binding']) != canonical_sha256(inventory['reference_inventory']['processor_binding']) or
                target_view['source_profile']['joint_schema'] != inventory['joint_schema'] or
                canonical_sha256(target_view['authored_ordinals']) != canonical_sha256(authored_ordinals) or
                canonical_sha256(target_view['authored_copy_tuples']) != canonical_sha256(authored_copy_tuples)):
            raise ValueError('externally pinned complete target/view/choices differ')
        require_sha256(alignment_sha256, 'alignment_sha256')
        if (type(alignment) is not dict or alignment.get('record_sha256') != alignment_sha256 or
                canonical_sha256({k:v for k,v in alignment.items() if k != 'record_sha256'}) != alignment_sha256):
            raise ValueError('complete mixed-wire alignment seal differs')
        ref_inventory = inventory['reference_inventory']
        if canonical_sha256(source_alignment.processor_binding(processor)) != canonical_sha256(ref_inventory['processor_binding']):
            raise ValueError('mixed-wire processor/input binding differs')
        ref_leaves = copies.reference_requests(decision, inventory)
        copy_leaves = copies.copy_requests(decision, inventory)
        expected = target.align_original_copy_decision_target(processor, target_view,
            view_sha256=expected_view_sha256, max_target_tokens=self.config.max_target_tokens)
        if (alignment_sha256 != expected['record_sha256'] or
                canonical_sha256(source_alignment.processor_binding(processor)) != canonical_sha256(ref_inventory['processor_binding'])):
            raise ValueError('mixed-wire geometry or complete marker roster differs')
        if (not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor) or
                input_ids.dtype != torch.long or labels.dtype != torch.long or
                input_ids.shape != (1, len(expected['decoder_inputs'])) or
                labels.shape != input_ids.shape or
                input_ids.tolist() != [expected['decoder_inputs']] or labels.tolist() != [expected['labels']] or
                input_ids.device != base_states.device or labels.device != base_states.device):
            raise ValueError('whole mixed-wire BOS/body/EOS shift or labels differ')
        if (type(authored_ordinals) is not list or
                [r.get('pointer') for r in authored_ordinals if type(r) is dict] !=
                [r['pointer'] for r in ref_leaves] or len(authored_ordinals) != len(ref_leaves)):
            raise ValueError('complete authored reference marker set differs')
        if (type(authored_copy_tuples) is not dict or
                authored_copy_tuples.get('binding') != copies.copy_binding(inventory) or
                type(authored_copy_tuples.get('rows')) is not list or
                [r.get('pointer') for r in authored_copy_tuples['rows'] if type(r) is dict] !=
                [r['pointer'] for r in copy_leaves] or
                len(authored_copy_tuples['rows']) != len(copy_leaves)):
            raise ValueError('complete authored copy marker set differs')
        source = self._source(base_states, source_mask)
        geometry = inventory['source_geometry']
        if (source.shape[1] != geometry['token_count'] or
                source_mask[0].tolist() != geometry['attention_mask']):
            raise ValueError('complete frozen source geometry differs')
        hidden = self.decision_states(base_states=source, source_mask=source_mask,
            input_ids=input_ids, binding=ref_inventory['binding'], mode='teacher_forced')
        decoded = hidden['hidden']
        # The original FormationSpecialist.forward calculation, on this already
        # decoded packet: tied full-vocabulary projection, chunked checkpoint,
        # summed token CE divided by all mixed-body and EOS labels.
        def chunk_loss(states, targets):
            logits = F.linear(states, self.token_embedding.weight, self.output_bias)
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=-100, reduction='sum')
        lexical = decoded.new_zeros((), dtype=torch.float32)
        for start in range(0, decoded.shape[1], self.config.logit_chunk_tokens):
            end = min(start + self.config.logit_chunk_tokens, decoded.shape[1])
            lexical = lexical + checkpoint(chunk_loss, decoded[:, start:end], labels[:, start:end],
                                           use_reentrant=False)
        lexical = lexical / labels.numel()
        reference_loss = None
        if ref_leaves:
            keys = self.candidate_keys(base_states=source, source_mask=source_mask,
                candidates=ref_inventory['candidates'], binding=ref_inventory['binding'])
            eligibility = torch.tensor([[slot in request['eligible_candidate_indices'] for slot in keys['slots']]
                for request in expected['requests']], dtype=torch.bool, device=source.device)
            reference_loss = self.selection_loss(hidden, expected['requests'], keys, eligibility,
                authored_ordinals=authored_ordinals)
        copy_loss = None
        if copy_leaves:
            keys = self.copy_keys(base_states=source, source_mask=source_mask, inventory=inventory)
            copy_loss = self.copy_loss(hidden, expected['copy_requests'], keys,
                                      authored_copy_tuples=authored_copy_tuples)
        total = lexical
        if reference_loss is not None:
            total = total + reference_loss
        if copy_loss is not None:
            total = total + copy_loss
        return dict(total_loss=total, lexical_loss=lexical, reference_loss=reference_loss,
            copy_loss=copy_loss, token_count=labels.numel(), reference_request_count=len(ref_leaves),
            copy_request_count=len(copy_leaves), objective_recipe=TASK,
            alignment_sha256=alignment_sha256, inventory_sha256=inventory['record_sha256'], hidden=hidden)

    def export_component(self, weight_path, config_path, manifest_path, *, component_id,
                         provenance, provenance_sha256, source_pins, source_pins_sha256,
                         evidence_scope):
        paths = [Path(p) for p in (weight_path, config_path, manifest_path)]
        if len({p.resolve() for p in paths}) != 3 or any(p.exists() for p in paths):
            raise FileExistsError('component output paths must be distinct and absent')
        if require_identifier(component_id, 'component_id') != component_id:
            raise ValueError('component_id must be canonical')
        if (type(provenance) is not dict or not provenance or
                canonical_sha256(provenance) != require_sha256(provenance_sha256, 'provenance_sha256') or
                type(source_pins) is not dict or not source_pins or
                canonical_sha256(source_pins) != require_sha256(source_pins_sha256, 'source_pins_sha256') or
                evidence_scope not in ('untrained_mechanics', 'unqualified_training')):
            raise ValueError('component requires externally pinned provenance/source and explicit evidence scope')
        state = {k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('complete component export requires finite FP32 tensors')
        inventory = {k:{'shape':list(v.shape),'dtype':'F32'} for k,v in sorted(state.items())}
        staged = []
        created = []
        try:
            for path in paths:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.joint-', delete=False) as stream:
                    staged.append(Path(stream.name))
            save_file(state, str(staged[0]))
            staged[1].write_bytes(canonical_json_bytes(self.config.record()))
            manifest = {'schema':COMPONENT_SCHEMA, 'component_id':component_id, 'task':TASK,
                'kind':'unqualified_component', 'evidence_scope':evidence_scope,
                'weight_file_sha256':_file_sha256(staged[0]), 'weight_bytes':staged[0].stat().st_size,
                'config_file_sha256':_file_sha256(staged[1]),
                'config_canonical_sha256':canonical_sha256(self.config.record()),
                'tensor_inventory':inventory, 'source_pins':deepcopy(source_pins),
                'source_pins_sha256':source_pins_sha256, 'provenance':deepcopy(provenance),
                'provenance_sha256':provenance_sha256, 'interface_pins':_interface_pins(),
                'authority_granted':False}
            staged[2].write_bytes(canonical_json_bytes(manifest))
            if any(p.exists() for p in paths):
                raise FileExistsError('component output appeared during export')
            for temporary, destination in zip(staged, paths):
                os.link(temporary, destination)
                created.append(destination)
            return deepcopy(manifest)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        finally:
            for path in staged:
                path.unlink(missing_ok=True)

    def load_component(self, weight_path, config_path, manifest_path, *, weight_sha256,
                       config_sha256, manifest_sha256, expected_provenance_sha256,
                       expected_source_pins, expected_evidence_scope):
        for name,value in (('weight_sha256',weight_sha256), ('config_sha256',config_sha256),
                           ('manifest_sha256',manifest_sha256),
                           ('expected_provenance_sha256',expected_provenance_sha256)):
            require_sha256(value,name)
        if any(_file_sha256(path) != pin for path,pin in
               ((weight_path,weight_sha256),(config_path,config_sha256),(manifest_path,manifest_sha256))):
            raise ValueError('complete component FILE binding differs')
        manifest = _read_json(manifest_path)
        config = _read_json(config_path)
        state = load_file(str(weight_path),device='cpu')
        current = self.state_dict()
        if (type(manifest) is not dict or
                require_identifier(manifest.get('component_id'), 'component_id') != manifest.get('component_id') or
                type(manifest.get('weight_bytes')) is not int or manifest['weight_bytes'] <= 0 or
                type(manifest.get('source_pins')) is not dict or not manifest['source_pins'] or
                type(expected_source_pins) is not dict or not expected_source_pins or
                type(manifest.get('provenance')) is not dict or not manifest['provenance']):
            raise ValueError('complete component manifest identity/schema differs')
        if (type(manifest) is not dict or set(manifest) != {
                'schema','component_id','task','kind','evidence_scope','weight_file_sha256','weight_bytes',
                'config_file_sha256','config_canonical_sha256','tensor_inventory','source_pins',
                'source_pins_sha256','provenance','provenance_sha256','interface_pins','authority_granted'} or
                manifest['schema'] != COMPONENT_SCHEMA or manifest['task'] != TASK or
                manifest['kind'] != 'unqualified_component' or manifest['authority_granted'] is not False or
                manifest['evidence_scope'] != expected_evidence_scope or
                manifest['evidence_scope'] not in ('untrained_mechanics','unqualified_training') or
                manifest['weight_file_sha256'] != weight_sha256 or
                manifest['weight_bytes'] != Path(weight_path).stat().st_size or
                manifest['config_file_sha256'] != config_sha256 or
                manifest['config_canonical_sha256'] != canonical_sha256(self.config.record()) or
                canonical_sha256(config) != manifest['config_canonical_sha256'] or
                config != self.config.record() or
                manifest['provenance_sha256'] != expected_provenance_sha256 or
                canonical_sha256(manifest['provenance']) != expected_provenance_sha256 or
                canonical_sha256(manifest['source_pins']) != manifest['source_pins_sha256'] or
                canonical_sha256(manifest['source_pins']) != canonical_sha256(expected_source_pins) or
                canonical_sha256(manifest['interface_pins']) != canonical_sha256(_interface_pins())):
            raise ValueError('complete component config/source/interface/provenance differs')
        expected_inventory = {k:{'shape':list(v.shape),'dtype':'F32'} for k,v in sorted(current.items())}
        if (canonical_sha256(manifest['tensor_inventory']) != canonical_sha256(expected_inventory) or
                set(state) != set(current)):
            raise ValueError('complete component tensor inventory differs')
        for name,value in state.items():
            if (value.shape != current[name].shape or value.dtype != torch.float32 or
                    current[name].dtype != torch.float32 or not torch.isfinite(value).all().item()):
                raise ValueError('complete component tensor shape/finite FP32 differs: '+name)
        if any(_file_sha256(path) != pin for path,pin in
               ((weight_path,weight_sha256),(config_path,config_sha256),(manifest_path,manifest_sha256))) or \
                canonical_sha256(manifest['interface_pins']) != canonical_sha256(_interface_pins()):
            raise ValueError('complete component or source changed during validation')
        self.load_state_dict(state,strict=True)
        self.component_binding = deepcopy(manifest)
        return deepcopy(manifest)
