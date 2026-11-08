"""Bounded compatible governor successors, without changing scientific sources.

Load exact optimized N1/N2[/N3] parent pins; supply a NEW TeachingAdmission,
explicit valid/superseded prior batch scope and reviewed replay. run_phase
warm-starts the complete model and uses the original IdentityTrainer with a
fresh AdamW. This is not exact optimizer/RNG/cursor resume. Its untouched native
checkpoint remains resumable by the original trainer under the SAME admission.

N1 continuation requires an explicit trained N1 auxiliary-objective pin.
N2-only continuation preserves N1 and N3 tensors. Governor changes make
historical N3 calibration inapplicable; original inference still numerically
uses those retained N3 tensors, and the response explicitly reports this.
Fresh outcome calibration and tensor/source-kind migrations are deferred.
Previously superseded batches cannot reenter through this compatible route;
explicit rollback/readmission requires a separate authorized migration.

Call respond_with_governance here for cumulative valid teaching scope. The
original consumer response, feature custody, full packet and renderer checks
are reused. Valid scope is a caller declaration, never qualification evidence.
This helper does not author targets, authorize private work or submit jobs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from src.alice_personality.identity.codec import canonical, fingerprint, label_semantics_fingerprint
from src.alice_personality.identity.consumer import CheckpointPin, DecisionConsumer
from src.alice_personality.identity.contracts import IdentityError
from src.alice_personality.teaching.contracts import TeachingError, TrainingRecipe
from src.alice_personality.teaching.objectives import IdentityObjectives
from src.alice_personality.teaching.trainer import (
    ArtifactPin, IdentityTrainer, TeachingAdmission, _digest, _hash, _json, _path,
    model_fingerprint,
)
from scripts.eipm.gemma_n0.governed_response import respond_with_governance as _governed_response

LINEAGE_SCHEMA = 'alice-personality-compatible-cumulative-successor-v1'


@dataclass(frozen=True)
class SuccessorPin:
    path: str
    file_sha256: str


def _pin_mapping(pin):
    return {'path': str(pin.path), 'manifest_sha256': pin.manifest_sha256}


def _strict_tensors(state, expected, label):
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise TeachingError(f'{label} must contain complete exact named tensors')
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise TeachingError(f'{label} tensor geometry/dtype or finite state differs')


def _checkpoint(config, pin, *, device='cpu', deep_custody=True):
    """Read full exact state, then run native evidence and admission validation."""
    if type(deep_custody) is not bool:
        raise TeachingError('checkpoint custody depth requires an actual Boolean')
    if type(pin) is not CheckpointPin:
        raise TeachingError('checkpoint needs its exact external typed pin')
    path = _path(pin.path, directory=True)
    if {item.name for item in path.iterdir()} != {'manifest.json', 'resume.pt'}:
        raise TeachingError('checkpoint payload membership changed')
    manifest = _json(_path(path / 'manifest.json'))
    if (manifest.get('manifest_sha256') != _digest(pin.manifest_sha256, 'checkpoint manifest')
            or sha256(canonical({k: v for k, v in manifest.items() if k != 'manifest_sha256'})).hexdigest()
            != pin.manifest_sha256):
        raise TeachingError('checkpoint manifest differs from its external pin')
    payload_path = _path(path / 'resume.pt')
    if _hash(payload_path) != manifest.get('resume_sha256'):
        raise TeachingError('checkpoint payload differs from its exact manifest')
    payload = torch.load(payload_path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
            'model', 'objectives', 'optimizer', 'torch_rng', 'cuda_rng', 'python_rng',
            'numpy_rng', 'steps', 'epoch', 'batch_index'}:
        raise TeachingError('checkpoint complete resumable payload schema differs')
    # Evidence verification must not consume the active training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = IdentityModel(config)
        objectives = IdentityObjectives(config.learned_width)
    _strict_tensors(payload['model'], model.state_dict(), 'checkpoint model')
    _strict_tensors(payload['objectives'], objectives.state_dict(), 'checkpoint objective')
    model.load_state_dict(payload['model'], strict=True)
    model.to(torch.device(device))
    model.set_phase('inference')
    try:
        DecisionConsumer(model, (pin,))
        data = dict(manifest['recipe'])
        data['betas'] = tuple(data['betas'])
        recipe = TrainingRecipe(**data)
        binding = manifest['binding']
        admission = TeachingAdmission({name: ArtifactPin(value['path'], value['file_sha256'])
            for name, value in binding['artifacts'].items()}, binding['model_config_sha256'],
            binding['model_initial_state_sha256'], binding['recipe_sha256'], binding['code_sha256'])
        reviewed, _, authorization = admission.verify(model, recipe, deep_custody=deep_custody)
    except (IdentityError, KeyError, TypeError) as error:
        raise TeachingError('checkpoint native evidence/admission differs') from error
    if canonical(admission.binding()) != canonical(binding) or recipe.phase != manifest['phase']:
        raise TeachingError('checkpoint admission runtime/recipe/phase binding differs')
    order = reviewed['batch_order'].get('CALIBRATION_TRAIN' if recipe.phase == 'calibration' else 'TRAIN')
    if not isinstance(order, list) or not order or len(set(order)) != len(order):
        raise TeachingError('checkpoint explicit reviewed replay order differs')
    for key, manifest_key in (('steps', 'step'), ('epoch', 'epoch'), ('batch_index', 'batch_index')):
        if type(payload[key]) is not int or payload[key] < 0 or payload[key] != manifest.get(manifest_key):
            raise TeachingError('checkpoint replay cursor differs from actual update evidence')
    if payload['batch_index'] >= len(order) or payload['steps'] != payload['epoch'] * len(order) + payload['batch_index']:
        raise TeachingError('checkpoint cursor differs from exact replay order')
    prefixes = ('n1.',) if recipe.phase == 'n1' else ('n2.', 'n1.') if recipe.adapt_n1 else \
        ('n2.',) if recipe.phase == 'n2' else ('n3.',)
    selected = [('model.' + name, value) for name, value in model.named_parameters() if name.startswith(prefixes)]
    if recipe.phase == 'n1':
        selected += [('teaching.' + name, value) for name, value in objectives.named_parameters()]
    if manifest.get('optimizer_parameters') != [name for name, _ in selected]:
        raise TeachingError('checkpoint optimizer parameter scope differs')
    optimizer = torch.optim.AdamW([value for _, value in selected], lr=recipe.learning_rate,
        betas=recipe.betas, eps=recipe.eps, weight_decay=recipe.weight_decay)
    try:
        optimizer.load_state_dict(payload['optimizer'])
    except (ValueError, KeyError, TypeError) as error:
        raise TeachingError('checkpoint optimizer state differs') from error
    for group in optimizer.param_groups:
        if (group['lr'] != recipe.learning_rate or tuple(group['betas']) != recipe.betas
                or group['eps'] != recipe.eps or group['weight_decay'] != recipe.weight_decay
                or group.get('amsgrad') is not False or group.get('maximize') is not False):
            raise TeachingError('checkpoint optimizer hyperparameters differ')
    for parameter, state in optimizer.state.items():
        if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                or any(not isinstance(state[key], torch.Tensor) or state[key].shape != parameter.shape
                    or state[key].dtype != parameter.dtype or not bool(torch.isfinite(state[key]).all())
                    for key in ('exp_avg', 'exp_avg_sq'))
                or bool((state['exp_avg_sq'] < 0).any())
                or not isinstance(state['step'], torch.Tensor) or state['step'].numel() != 1
                or not bool(torch.isfinite(state['step']).all()) or float(state['step']) < 0
                or not float(state['step']).is_integer()):
            raise TeachingError('checkpoint AdamW moments/step differ')
    for index, event in enumerate(manifest['events']):
        digest = order[index % len(order)]
        if (event.get('teaching_batch_sha256') != digest or digest not in reviewed['reviewed_batches']
                or event.get('source_class') != reviewed['source_class']
                or event.get('split') != reviewed['reviewed_batches'][digest]['split']
                or event.get('epoch') != (index + 1) // len(order)
                or event.get('behavior_qualification') is not None
                or event.get('private_gradient_authorized') is not authorization['private_gradient_authorized']):
            raise TeachingError('checkpoint events differ from actual admitted replay')
    if (_hash(payload_path) != manifest['resume_sha256']
            or canonical(_json(path / 'manifest.json')) != canonical(manifest)):
        raise TeachingError('checkpoint custody changed during reconstruction')
    return model, manifest, payload['objectives']


def _batches(manifests):
    return {event['teaching_batch_sha256'] for manifest in manifests if manifest['phase'] != 'calibration'
        for event in manifest['events']}


def _scope(retained, superseded, previous, order, phase, manifests, historical_superseded):
    if (type(retained) is not tuple or len(set(retained)) != len(retained)
            or not isinstance(superseded, Mapping)
            or any(not isinstance(reason, str) or not reason.strip() for reason in superseded.values())):
        raise TeachingError('valid/superseded parent teaching requires explicit unique scope and revision reasons')
    for digest in (*retained, *superseded):
        _digest(digest, 'parent teaching batch')
    if set(retained) & set(superseded) or set(retained) | set(superseded) != previous:
        raise TeachingError('valid/superseded teaching must account for exact prior valid parent scope')
    same_phase = {event['teaching_batch_sha256'] for manifest in manifests if manifest['phase'] == phase
        for event in manifest['events']} & set(retained)
    if not same_phase <= set(order) or (set(superseded) | historical_superseded) & set(order):
        raise TeachingError('new admission must replay valid same-phase teaching and exclude superseded targets')
    return sorted(set(retained) | set(order))


class CumulativeSuccessor:
    """Externally pinned native parents plus explicit compatible successor links."""
    @classmethod
    def load(cls, config: IdentityModelConfig, checkpoints: tuple[CheckpointPin, ...], *, successors=(), device='cpu'):
        if (type(config) is not IdentityModelConfig or type(checkpoints) is not tuple
                or len(checkpoints) not in (2, 3) or type(successors) is not tuple
                or any(type(pin) is not SuccessorPin for pin in successors)):
            raise TeachingError('successor needs exact config, N1/N2[/N3] parent pins and lineage pins')
        self = cls()
        self.config, self.parent_checkpoints, self.successors = config, checkpoints, successors
        self.model = None
        verified = self._verify(device=device, deep_custody=True)
        self.model = verified['model']
        return self

    def _verify(self, *, device=None, deep_custody=False):
        if type(deep_custody) is not bool:
            raise TeachingError('successor custody depth requires an actual Boolean')
        # Match native trainer reuse: full custody before first use, then
        # recheck the exact bound artifacts and complete checkpoint evidence.
        deep_custody = self.model is None or deep_custody
        target_device = device or (str(next(self.model.parameters()).device) if self.model is not None else 'cpu')
        manifests, checkpoint_pins = [], list(self.parent_checkpoints)
        for expected, pin in zip(('n1', 'n2', 'calibration'), self.parent_checkpoints):
            model, manifest, _ = _checkpoint(self.config, pin, device=target_device,
                deep_custody=deep_custody)
            if manifest['phase'] != expected:
                raise TeachingError('parent governing stage order changed')
            manifests.append(manifest)
        try:
            DecisionConsumer(model, self.parent_checkpoints)
        except IdentityError as error:
            raise TeachingError('actual parent model lineage differs') from error
        valid, receipts = _batches(manifests), []
        previous = model_fingerprint(model)
        prepared_sha = manifests[-1]['binding']['artifacts']['prepared_receipt']['file_sha256']
        for pin in self.successors:
            path = _path(pin.path)
            receipt = _json(path)
            if (receipt.get('schema') != LINEAGE_SCHEMA or _hash(path) != _digest(pin.file_sha256, 'successor receipt file')
                    or receipt.get('receipt_sha256') != sha256(canonical({k: v for k, v in receipt.items() if k != 'receipt_sha256'})).hexdigest()
                    or receipt.get('helper_sha256') != _hash(Path(__file__).resolve())
                    or receipt.get('parent_checkpoints') != [_pin_mapping(p) for p in self.parent_checkpoints]
                    or receipt.get('preceding_successors') != [{'path': p.path, 'file_sha256': p.file_sha256}
                        for p in self.successors[:len(receipts)]]
                    or receipt.get('parent_model_state_sha256') != previous
                    or receipt.get('optimizer_continuation') != 'NEW_ADMISSION_FRESH_ADAMW'
                    or receipt.get('qualification') != 'UNQUALIFIED' or receipt.get('acceptance_authority') is not False):
                raise TeachingError('successor receipt/pinned parent lineage differs')
            child_pin = CheckpointPin(**receipt['checkpoint'])
            child, manifest, objective = _checkpoint(self.config, child_pin, device=target_device,
                deep_custody=deep_custody)
            if (manifest['phase'] not in ('n1', 'n2')
                    or manifest['binding']['model_initial_state_sha256'] != previous
                    or manifest['binding']['artifacts']['prepared_receipt']['file_sha256'] != prepared_sha
                    or receipt['model_state_sha256'] != manifest['model_state_sha256']
                    or receipt['objective_final_state_sha256'] != fingerprint(objective)):
                raise TeachingError('compatible successor model/provider/objective lineage differs')
            data = manifest['recipe']
            updated = ['n1'] if data['phase'] == 'n1' else ['n1', 'n2'] if data['adapt_n1'] else ['n2']
            if receipt['updated_stages'] != updated or receipt['preserved_stages'] != sorted({'n1', 'n2', 'n3'} - set(updated)):
                raise TeachingError('successor source-stage validity differs from actual phase/adapt_n1')
            for stage in receipt['preserved_stages']:
                if any(not torch.equal(value, child.state_dict()[name]) for name, value in model.state_dict().items()
                        if name.startswith(stage + '.')):
                    raise TeachingError('successor changed a preserved governing stage')
            order = _json(_path(manifest['binding']['artifacts']['reviewed_targets']['path']))['batch_order']['TRAIN']
            computed = _scope(tuple(receipt['retained_parent_batches']), receipt['superseded_parent_batches'],
                valid, order, manifest['phase'], manifests,
                {digest for prior in receipts for digest in prior['superseded_parent_batches']})
            if receipt['valid_governing_batches'] != computed:
                raise TeachingError('successor cumulative valid teaching scope differs')
            transfer = receipt['objective_transfer']
            if transfer is not None:
                source_pin = CheckpointPin(**transfer)
                if source_pin not in checkpoint_pins:
                    raise TeachingError('objective transfer lacks an exact parent checkpoint pin')
                _, source_manifest, source_objective = _checkpoint(self.config, source_pin, device=target_device,
                    deep_custody=deep_custody)
                if receipt['objective_initial_state_sha256'] != fingerprint(source_objective):
                    raise TeachingError('successor initial auxiliary objective lineage differs')
                if manifest['phase'] == 'n1' and source_manifest['phase'] != 'n1':
                    raise TeachingError('N1 continuation requires a trained N1 auxiliary objective')
            elif manifest['phase'] == 'n1':
                raise TeachingError('N1 continuation cannot reset its learned auxiliary objective')
            valid, previous, model = set(computed), manifest['model_state_sha256'], child
            receipts.append(receipt)
            manifests.append(manifest)
            checkpoint_pins.append(child_pin)
        if self.model is not None and model_fingerprint(self.model) != previous:
            raise TeachingError('actual successor model differs from exact optimized lineage')
        return {'model': model, 'manifests': manifests, 'receipts': receipts,
            'valid': valid, 'checkpoints': tuple(checkpoint_pins)}

    def run_phase(self, recipe, admission, *, batch_loader, epochs, destination, lineage_destination,
                  retained_parent_batches, superseded_parent_batches, objective_transfer=None):
        if (type(recipe) is not TrainingRecipe or recipe.phase not in ('n1', 'n2')
                or type(admission) is not TeachingAdmission or type(epochs) is not int or epochs < 1
                or not callable(batch_loader)):
            raise TeachingError('compatible governor continuation needs a new N1/N2 admission and explicit replay')
        verified = self._verify(deep_custody=True)
        reviewed, _, _ = admission.verify(self.model, recipe, initial=True)
        order = reviewed['batch_order'].get('TRAIN')
        if type(order) is not list or not order or len(set(order)) != len(order):
            raise TeachingError('new admission requires explicit complete unique TRAIN replay order')
        valid = _scope(retained_parent_batches, superseded_parent_batches, verified['valid'],
            order, recipe.phase, verified['manifests'],
            {digest for prior in verified['receipts'] for digest in prior['superseded_parent_batches']})
        if admission.artifacts['prepared_receipt'].sha256 != verified['manifests'][-1]['binding']['artifacts']['prepared_receipt']['file_sha256']:
            raise TeachingError('compatible continuation must retain its exact prepared feature provider')
        if objective_transfer is not None:
            if objective_transfer not in verified['checkpoints']:
                raise TeachingError('auxiliary objective needs an exact externally pinned parent checkpoint')
            _, objective_manifest, objective_state = _checkpoint(self.config, objective_transfer)
            if recipe.phase == 'n1' and objective_manifest['phase'] != 'n1':
                raise TeachingError('N1 continuation requires a trained N1 auxiliary objective')
        elif recipe.phase == 'n1':
            raise TeachingError('N1 continuation requires explicit auxiliary objective transfer')
        lineage_path = Path(lineage_destination)
        checkpoint_path = Path(destination)
        for path in (lineage_path, checkpoint_path):
            if (not path.is_absolute() or path.exists() or not path.parent.is_dir()
                    or any(item.is_symlink() or (hasattr(item, 'is_junction') and item.is_junction())
                        for item in (path, *path.parents))):
                raise TeachingError('successor outputs require new absolute paths without links')
        if lineage_path == checkpoint_path or lineage_path.is_relative_to(checkpoint_path):
            raise TeachingError('lineage receipt must remain outside untouched native checkpoint membership')
        parent_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        first = batch_loader(order[0])
        trainer = IdentityTrainer(self.model, recipe, admission, first_batch=first)
        # Validate the complete new admission before any update. Keep streaming.
        for digest in order:
            if trainer._check_batch(batch_loader(digest)) != digest:
                raise TeachingError('streamed replay differs from exact reviewed batch')
        if objective_transfer is not None:
            trainer.objectives.load_state_dict(objective_state, strict=True)
        initial_objective = fingerprint(trainer.objectives.state_dict())
        for _ in range(epochs):
            for digest in order:
                batch = batch_loader(digest)
                if batch.fingerprint() != digest:
                    raise TeachingError('streamed replay differs from exact reviewed batch')
                trainer.step(batch)
        updated = ['n1'] if recipe.phase == 'n1' else ['n1', 'n2'] if recipe.adapt_n1 else ['n2']
        preserved = sorted({'n1', 'n2', 'n3'} - set(updated))
        if any(not torch.equal(value, self.model.state_dict()[name]) for name, value in parent_state.items()
                if name.split('.')[0] in preserved):
            raise TeachingError('successor optimizer changed a preserved governing stage')
        manifest = trainer.save_checkpoint(checkpoint_path)
        receipt = {'schema': LINEAGE_SCHEMA, 'helper_sha256': _hash(Path(__file__).resolve()),
            'parent_checkpoints': [_pin_mapping(pin) for pin in self.parent_checkpoints],
            'preceding_successors': [{'path': pin.path, 'file_sha256': pin.file_sha256} for pin in self.successors],
            'checkpoint': _pin_mapping(CheckpointPin(str(checkpoint_path), manifest['manifest_sha256'])),
            'parent_model_state_sha256': fingerprint(parent_state), 'model_state_sha256': manifest['model_state_sha256'],
            'optimizer_continuation': 'NEW_ADMISSION_FRESH_ADAMW', 'updated_stages': updated, 'preserved_stages': preserved,
            'objective_transfer': None if objective_transfer is None else _pin_mapping(objective_transfer),
            'objective_initial_state_sha256': initial_objective,
            'objective_final_state_sha256': fingerprint(trainer.objectives.state_dict()),
            'retained_parent_batches': list(retained_parent_batches),
            'superseded_parent_batches': dict(superseded_parent_batches), 'valid_governing_batches': valid,
            'qualification': 'UNQUALIFIED', 'acceptance_authority': False}
        receipt['receipt_sha256'] = sha256(canonical(receipt)).hexdigest()
        with lineage_path.open('xb') as stream:
            stream.write(canonical(receipt) + b'\n')
        self.successors += (SuccessorPin(str(lineage_path), _hash(lineage_path)),)
        self.model.set_phase('inference')
        self._verify()
        return {'checkpoint': manifest, 'lineage': deepcopy(receipt)}

    def consumer(self):
        return SuccessorConsumer(self)


class SuccessorConsumer(DecisionConsumer):
    """Native respond inherited unchanged; evidence supplied by verified links."""
    def __init__(self, construction):
        self.construction = construction
        self.model = construction.model
        self.checkpoints = construction._verify()['checkpoints']
        self._evidence()

    def _evidence(self):
        verified = self.construction._verify()
        if verified['checkpoints'] != self.checkpoints or self.model is not self.construction.model:
            raise IdentityError('consumer successor checkpoint/model binding changed')
        fields, labels, n1_fields, classes = set(), {}, set(), set()
        for manifest in verified['manifests']:
            for event in manifest['events']:
                classes.add(event['source_class'])
                if event['teaching_batch_sha256'] not in verified['valid']:
                    continue
                scope = fields if manifest['phase'] == 'n2' else n1_fields if manifest['phase'] == 'n1' else None
                if scope is None:
                    continue
                scope.update(field for field, count in event.get('effective_target_coverage', {}).items() if count > 0)
                if manifest['phase'] == 'n2':
                    for field, values in event.get('reviewed_label_coverage', {}).items():
                        labels.setdefault(field, set()).update(values)
        historical = [pin.manifest_sha256 for pin, manifest in zip(self.checkpoints, verified['manifests'])
            if manifest['phase'] == 'calibration']
        calibration = {'applicable': bool(historical) and not verified['receipts'],
            'historical_checkpoint_manifest_sha256': historical,
            'numerical_scaling': 'RETAINED_HISTORICAL_N3_INAPPLICABLE' if historical and verified['receipts'] else
                'PARENT_N3' if historical else 'UNCALIBRATED_INITIAL_N3',
            'fresh_outcome_calibration_required': bool(verified['receipts'])}
        return {'model_state_sha256': model_fingerprint(self.model),
            'checkpoint_manifest_sha256': [pin.manifest_sha256 for pin in self.checkpoints],
            'prepared_pin': verified['manifests'][-1]['binding']['artifacts']['prepared_receipt'],
            'reviewed_fields': fields, 'reviewed_labels': labels, 'reviewed_n1_fields': n1_fields,
            'source_classes': sorted(classes), 'calibration_applicability': calibration,
            'successor_lineage': {'parent_checkpoint_manifest_sha256': [pin.manifest_sha256
                for pin in self.construction.parent_checkpoints],
                'successor_receipt_file_sha256': [pin.file_sha256 for pin in self.construction.successors],
                'valid_governing_batches': sorted(verified['valid']),
                'updated_stages': [receipt['updated_stages'] for receipt in verified['receipts']]}}


def _annotate_scope(result, evidence, document):
    for field, scope in result['output_scope'].items():
        family = field[5:] if field.startswith('head:') else 'voice' if field.startswith('voice_control_') else None
        scope['n1_taught'] = field in evidence['reviewed_n1_fields']
        if not evidence['calibration_applicability']['applicable']:
            scope['n3_reviewed'] = False
            if 'global_n3_coverage' in scope:
                scope['global_n3_coverage'] = False
            if family is not None:
                scope['reviewed_current_label_ids'] = sorted({spec.label_id
                    for spec in document.labels[family]
                    if label_semantics_fingerprint(family, spec) in evidence['reviewed_labels'].get(field, set())})
        # Preserve the native semantic gate: N3 reviews do not teach meanings
        # or portable voice units, even while parent calibration is applicable.
        scope['taught'] = scope['n2_taught'] if family is not None else \
            scope['n1_taught'] or scope['n2_taught'] or scope['n3_reviewed']
    result['calibration_applicability'] = evidence['calibration_applicability']
    result['successor_lineage'] = evidence['successor_lineage']


def respond_with_governance(consumer, document, produced, closed_session, *, renderer=None):
    """Original full checks; valid scope reaches both renderer and final receipt."""
    if type(consumer) is not SuccessorConsumer:
        raise IdentityError('cumulative governed response requires its exact successor consumer')
    def realize(packet):
        _annotate_scope(packet, consumer._evidence(), document)
        return renderer(packet)
    result = _governed_response(consumer, document, produced, closed_session,
        renderer=None if renderer is None else realize)
    _annotate_scope(result, consumer._evidence(), document)
    result.pop('receipt_sha256')
    result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
    return result
