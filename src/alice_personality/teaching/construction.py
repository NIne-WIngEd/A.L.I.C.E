"""Stream reviewed construction through governing stages, then actual responses.

Source/target authorship and private isolation belong to the caller. Every
batch still enters the same fully admitted trainer; this adds no shortcut.
"""
from pathlib import Path

from ..identity.consumer import CheckpointPin, DecisionConsumer
from .contracts import TeachingError
from .trainer import IdentityTrainer, _json, _path, model_fingerprint


class ConnectedConstruction:
    def __init__(self, model, *, checkpoints=()):
        if type(checkpoints) is not tuple or len(checkpoints) > 3:
            raise TeachingError("construction requires an ordered governing checkpoint chain")
        self.model, self.checkpoints = model, checkpoints
        if checkpoints:
            DecisionConsumer(model, checkpoints)
            for expected, pin in zip(('n1', 'n2', 'calibration'), checkpoints):
                if _json(Path(pin.path) / 'manifest.json').get('phase') != expected:
                    raise TeachingError("construction checkpoint stage order changed")

    def run_phase(self, recipe, admission, *, batch_loader, epochs, destination,
                  epoch_checkpoint_directory=None):
        expected = ('n1', 'n2', 'calibration')[len(self.checkpoints)] if len(self.checkpoints) < 3 else None
        if recipe.phase != expected:
            raise TeachingError("connected construction must preserve N1/N2/N3 stage order")
        if type(epochs) is not int or epochs < 1 or not callable(batch_loader):
            raise TeachingError("construction needs an explicit positive epoch count and streaming batch loader")
        epoch_directory = None if epoch_checkpoint_directory is None else _path(epoch_checkpoint_directory, directory=True)
        if epoch_directory is not None and any(epoch_directory.iterdir()):
            raise TeachingError("epoch checkpoints require a new empty custody directory")
        if self.checkpoints:
            DecisionConsumer(self.model, self.checkpoints)
            previous = _json(Path(self.checkpoints[-1].path) / 'manifest.json')
            if previous.get('model_state_sha256') != model_fingerprint(self.model):
                raise TeachingError("construction initial model differs from the prior actual governing checkpoint")
        reviewed = _json(admission.artifacts['reviewed_targets'].verify())
        split = 'CALIBRATION_TRAIN' if recipe.phase == 'calibration' else 'TRAIN'
        order = reviewed.get('batch_order', {}).get(split)
        if type(order) is not list or not order or len(set(order)) != len(order):
            raise TeachingError("construction needs complete explicit reviewed replay order")
        first = batch_loader(order[0])
        if first.fingerprint() != order[0]:
            raise TeachingError("streamed batch differs from its exact reviewed fingerprint")
        trainer = IdentityTrainer(self.model, recipe, admission, first_batch=first)
        for epoch in range(epochs):
            for index, digest in enumerate(order):
                batch = first if epoch == 0 and index == 0 else batch_loader(digest)
                if batch.fingerprint() != digest:
                    raise TeachingError("streamed batch differs from its exact reviewed fingerprint")
                trainer.step(batch)
                if epoch == 0 and index == 0:
                    first = None
                del batch
            if epoch_directory is not None:
                trainer.save_checkpoint(epoch_directory / f'epoch-{trainer.epoch:04d}')
        manifest = trainer.save_checkpoint(destination)
        self.checkpoints += (CheckpointPin(str(Path(destination)), manifest['manifest_sha256']),)
        self.model.set_phase('inference')
        return manifest

    def consumer(self):
        if len(self.checkpoints) < 2:
            raise TeachingError("actual response construction requires the N1 and N2 governing stages")
        return DecisionConsumer(self.model, self.checkpoints)
