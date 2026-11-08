"""Public reusable N3 continuation boundary; never a private-data launcher.

The owner worker must establish isolation BEFORE reading protected inputs and
externally pin this helper, checkpoints, cases, responses, allocation, closed
session, source artifacts and current authorization. This module neither reads
environment-selected private paths nor authors targets/authorization/approval.

Call load_n1_n2_checkpoint_chain, then prepare_calibration. Persist its
reviewed_targets as a new protected artifact, create a fresh calibration
authorization bound to those exact artifacts/model/recipe/code, and call
run_calibration. Serialize iter_connected_responses with codec.canonical:
codec tuples and parsed JSON arrays are equivalent only at that boundary.

No source or target is automatically promoted. All output remains unqualified
primary development. Existing identity/teaching source bytes stay unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from typing import Mapping

import torch

from src.alice_personality.identity import IdentityError, IdentityModel, IdentityModelConfig
from src.alice_personality.identity.codec import canonical, decision_to_mapping, fingerprint
from src.alice_personality.identity.consumer import CheckpointPin
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching.cases import OUTCOME_CASE_SCHEMA, ReviewedTeachingCase
from src.alice_personality.teaching.construction import ConnectedConstruction
from src.alice_personality.teaching.contracts import REVIEW_SCHEMA, TARGET_SCHEMA, TeachingError
from src.alice_personality.teaching.lineage import validate_allocation
from src.alice_personality.teaching.materialize import materialize_case
from src.alice_personality.teaching.outcomes import calibration_case_from_review
from src.alice_personality.teaching.trainer import (
    CHECKPOINT_SCHEMA, TeachingAdmission, _hash, _json, _path,
    current_code_hashes, model_fingerprint,
)


def _core_fingerprint(model):
    return fingerprint({name: value for name, value in model.state_dict().items()
                        if name.startswith(('n1.', 'n2.'))})


def load_n1_n2_checkpoint_chain(config: IdentityModelConfig, checkpoints: tuple[CheckpointPin, ...],
                               *, device='cpu') -> ConnectedConstruction:
    """Reconstruct the exact finite complete model, then verify native lineage.

    This is a cross-stage model load, not a resume of the N2 optimizer. N3 gets
    its own fresh optimizer/current admission. No checkpoint is rewritten.
    """
    if (type(config) is not IdentityModelConfig or type(checkpoints) is not tuple or len(checkpoints) != 2
            or any(type(pin) is not CheckpointPin for pin in checkpoints)):
        raise TeachingError('calibration continuation requires exact N1/N2 checkpoint pins and config')
    config.validate()
    model = IdentityModel(config)
    expected = model.state_dict()
    config_sha = sha256(canonical(asdict(config))).hexdigest()
    for phase, pin in zip(('n1', 'n2'), checkpoints):
        path = _path(pin.path, directory=True)
        if {item.name for item in path.iterdir()} != {'manifest.json', 'resume.pt'}:
            raise TeachingError('checkpoint payload membership changed')
        manifest = _json(_path(path / 'manifest.json'))
        body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
        if (manifest.get('manifest_sha256') != pin.manifest_sha256
                or sha256(canonical(body)).hexdigest() != pin.manifest_sha256):
            raise TeachingError('checkpoint manifest differs from the exact external pin')
        if (manifest.get('phase') != phase or manifest.get('schema') != CHECKPOINT_SCHEMA
                or manifest.get('training_state') != 'OPTIMIZED_UNQUALIFIED'
                or manifest.get('qualification') != 'UNQUALIFIED' or manifest.get('acceptance_authority') is not False
                or manifest.get('binding', {}).get('model_config_sha256') != config_sha
                or manifest.get('binding', {}).get('code_sha256') != current_code_hashes()):
            raise TeachingError('checkpoint phase/config/code or development classification differs')
        payload_path = _path(path / 'resume.pt')
        if _hash(payload_path) != manifest.get('resume_sha256'):
            raise TeachingError('checkpoint payload differs from the exact manifest pin')
        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
        state = payload.get('model') if isinstance(payload, dict) else None
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise TeachingError('checkpoint must contain the complete exact named model state')
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise TeachingError('checkpoint state requires finite dense named tensors')
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise TeachingError('checkpoint model geometry/dtype differs from its exact config')
            if not bool(torch.isfinite(value).all()):
                raise TeachingError('checkpoint model state must be finite')
        if fingerprint(state) != manifest.get('model_state_sha256'):
            raise TeachingError('checkpoint complete model fingerprint differs')
        if phase == 'n2':
            model.load_state_dict(state, strict=True)
        # Preserve exact file custody around deserialization, not only before it.
        if _hash(payload_path) != manifest['resume_sha256'] or canonical(_json(path / 'manifest.json')) != canonical(manifest):
            raise TeachingError('checkpoint payload/manifest changed during reconstruction')
        del payload, state
    model.to(torch.device(device))
    model.set_phase('inference')
    try:
        return ConnectedConstruction(model, checkpoints=checkpoints)
    except IdentityError as error:
        raise TeachingError('actual N1/N2 checkpoint/model lineage differs') from error


class PreparedCalibration:
    """Pinned outcome batches replayed without retaining whole source frames."""
    def __init__(self, construction, cases, case_pins, artifacts, allocation, replay, session, reviewed, batch_cases):
        self.construction = construction
        self.initial_model_sha256 = model_fingerprint(construction.model)
        self._core_sha256 = _core_fingerprint(construction.model)
        self._cases, self._case_pins = cases, dict(case_pins)
        self._artifacts, self._allocation = deepcopy(artifacts), deepcopy(allocation)
        self._replay, self._session = replay, deepcopy(session)
        self._reviewed, self._batch_cases = deepcopy(reviewed), dict(batch_cases)

    @property
    def reviewed_targets(self):
        return deepcopy(self._reviewed)

    def batch_loader(self, digest):
        if _core_fingerprint(self.construction.model) != self._core_sha256:
            raise TeachingError('fixed N1/N2 model changed during calibration replay')
        if digest not in self._batch_cases:
            raise TeachingError('calibration batch lacks an exact reviewed replay pin')
        case = self._batch_cases[digest]
        produced = self._replay.produce(case.document)
        batch = materialize_case(case, produced, self._session, self.construction.model.config,
            expected_case_sha256=self._case_pins[case.case_id], expected_artifact_sha256=self._artifacts,
            family_splits=self._allocation).batch
        if batch.fingerprint() != digest:
            raise TeachingError('calibration replay changed its exact reviewed batch')
        return batch


def prepare_calibration(construction: ConnectedConstruction, *, cases, expected_case_sha256,
                        response_receipts, artifact_sha256, family_allocation, feature_store,
                        closed_session, expected_session_sha256) -> PreparedCalibration:
    """Validate explicit actual outcomes against the original optimized model.

    Targets come exclusively from supplied reviewed cases. Actual response
    receipts are reproduced by packet fingerprint and decision fields before
    N3 learning; rendered output remains the separately bound observation.
    Both calibration TRAIN and DEV cases may be prepared; only TRAIN optimizes.
    """
    if type(construction) is not ConnectedConstruction or len(construction.checkpoints) != 2:
        raise TeachingError('actual N1/N2 construction is required before outcome calibration')
    consumer = construction.consumer()
    if (type(cases) is not tuple or not cases or any(type(case) is not ReviewedTeachingCase for case in cases)
            or len({case.case_id for case in cases}) != len(cases)
            or not isinstance(expected_case_sha256, Mapping)
            or set(expected_case_sha256) != {case.case_id for case in cases}
            or not isinstance(response_receipts, Mapping)):
        raise TeachingError('calibration needs unique explicit externally pinned actual-outcome cases')
    validate_allocation(family_allocation)
    try:
        replay = FrozenFrameProducer.replay(feature_store, construction.model.config, closed_session,
            expected_session_sha256=expected_session_sha256)
    except IdentityError as error:
        raise TeachingError('calibration closed-session replay differs') from error
    initial = model_fingerprint(construction.model)
    checkpoints = [pin.manifest_sha256 for pin in construction.checkpoints]
    classes = {case.source_class for case in cases}
    if len(classes) != 1:
        raise TeachingError('one calibration admission cannot mix source authority classes')
    reviewed = {'schema': REVIEW_SCHEMA, 'target_schema': TARGET_SCHEMA, 'reviewed': True,
        'source_class': next(iter(classes)), 'reviewed_batches': {}, 'batch_order': {},
        'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256) for case in cases}}
    batch_cases = {}
    for case in cases:
        case.validate()
        if case.schema != OUTCOME_CASE_SCHEMA or case.phase != 'calibration' or case.sha256 != expected_case_sha256[case.case_id]:
            raise TeachingError('calibration case differs from its exact actual-outcome review pin')
        binding = case.outcome_binding
        if binding['model_state_sha256'] != initial or binding['checkpoint_manifest_sha256'] != checkpoints:
            raise TeachingError('actual outcome model/checkpoint chain differs from calibration initial state')
        response = response_receipts.get(binding['response_receipt_sha256'])
        reconstructed = calibration_case_from_review(response_receipt=response,
            expected_response_sha256=binding['response_receipt_sha256'], document=case.document,
            case_id=case.case_id, split=case.split, source_class=case.source_class, lineage=case.lineage,
            targets=case.targets, artifact_sha256=case.artifact_sha256, review_id=case.review_id,
            annotation_note=case.annotation_note)
        if reconstructed.sha256 != case.sha256:
            raise TeachingError('calibration outcome binding differs from the actual response')
        produced = replay.produce(case.document)
        actual = consumer.respond(case.document, produced, closed_session, tie_tolerance=response['tie_tolerance'])
        for key in ('model_state_sha256', 'checkpoint_manifest_sha256', 'document_sha256', 'frame_sha256',
                    'feature_binding_sha256', 'closed_session_sha256', 'packet_sha256', 'selected_candidate_ids',
                    'response_text', 'status', 'acceptable_candidate_ids', 'intents'):
            if canonical(actual[key]) != canonical(response[key]):
                raise TeachingError('actual reviewed response differs from original optimized decision')
        batch = materialize_case(case, produced, closed_session, construction.model.config,
            expected_case_sha256=expected_case_sha256[case.case_id], expected_artifact_sha256=artifact_sha256,
            family_splits=family_allocation).batch
        digest = batch.fingerprint()
        if digest in batch_cases:
            raise TeachingError('duplicate calibration batch fingerprint')
        batch_cases[digest] = case
        reviewed['reviewed_batches'][digest] = {'split': batch.split, 'case_ids': list(batch.case_ids),
            'source_family_ids': [list(row) for row in batch.source_family_ids],
            'construction_lineage_sha256': fingerprint(batch.lineage)}
        reviewed['batch_order'].setdefault(batch.split, []).append(digest)
        del actual, batch, produced
    if not reviewed['batch_order'].get('CALIBRATION_TRAIN'):
        raise TeachingError('calibration continuation needs eligible reviewed CALIBRATION_TRAIN outcomes')
    return PreparedCalibration(construction, cases, expected_case_sha256, artifact_sha256,
        family_allocation, replay, closed_session, reviewed, batch_cases)


def run_calibration(construction: ConnectedConstruction, recipe, artifacts, *, prepared: PreparedCalibration,
                    epochs, destination, epoch_checkpoint_directory=None):
    """Use fresh six-pin admission and optimizer; preserve actual N1/N2 bytes."""
    if recipe.phase != 'calibration' or recipe.adapt_n1:
        raise TeachingError('continuation requires a separate calibration-only recipe')
    if (type(prepared) is not PreparedCalibration or prepared.construction is not construction
            or model_fingerprint(construction.model) != prepared.initial_model_sha256
            or len(construction.checkpoints) != 2):
        raise TeachingError('calibration initial model/checkpoint or prepared outcome set differs')
    if (not isinstance(artifacts, Mapping) or 'reviewed_targets' not in artifacts or 'source_families' not in artifacts
            or canonical(_json(artifacts['reviewed_targets'].verify())) != canonical(prepared.reviewed_targets)
            or canonical(_json(artifacts['source_families'].verify())) != canonical(prepared._allocation)):
        raise TeachingError('fresh calibration target/allocation artifacts differ from prepared outcomes')
    admission = TeachingAdmission(artifacts, sha256(canonical(asdict(construction.model.config))).hexdigest(),
        prepared.initial_model_sha256, recipe.fingerprint(), current_code_hashes())
    manifest = construction.run_phase(recipe, admission, batch_loader=prepared.batch_loader,
        epochs=epochs, destination=destination, epoch_checkpoint_directory=epoch_checkpoint_directory)
    if _core_fingerprint(construction.model) != prepared._core_sha256:
        raise TeachingError('calibration changed the fixed N1/N2 state')
    return manifest


def iter_connected_responses(construction: ConnectedConstruction, prepared: PreparedCalibration, *, documents,
                             renderer=None):
    """Yield real response receipts AND complete native packets for caller custody."""
    if (type(prepared) is not PreparedCalibration or prepared.construction is not construction
            or len(construction.checkpoints) != 3 or _core_fingerprint(construction.model) != prepared._core_sha256
            or not isinstance(documents, Mapping) or not documents):
        raise TeachingError('full calibrated consumer needs its actual completed governing chain')
    consumer = construction.consumer()
    for name, (split, document) in documents.items():
        if not isinstance(name, str) or not name or not isinstance(split, str) or not split:
            raise TeachingError('connected response documents require explicit IDs and split labels')
        produced = prepared._replay.produce(document)
        response = consumer.respond(document, produced, prepared._session, renderer=renderer)
        with torch.no_grad():
            packet = construction.model(produced.frame)
        if fingerprint(packet) != response['packet_sha256']:
            raise TeachingError('full calibrated decision differs from the actual response receipt')
        yield {'document_id': name, 'split': split, 'response': response, 'decision': decision_to_mapping(packet)}

