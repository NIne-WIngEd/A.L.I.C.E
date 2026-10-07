"""Explicit source reconstruction and formation tasks on the fresh MFM head.

Task binding adds a learned vector to positional inputs. The original decoder,
full-vocabulary tied projection and loss implementation are reused unchanged.
The formation vector starts at zero, preserving parent logits exactly. Source
reconstruction teaches exact opened text; it does not form or authorize memory.
"""
import torch
from .formation_v1_specialist import FormationSpecialist
from .formation_decision_specialist_v16 import FormationDecisionSpecialist

TASKS = ('formation', 'source_reconstruction')
INITIALIZATION = 'pinned-formation-parent-plus-zero-formation-task-and-fresh-source-task-v1'


class _TaskPositions:
    def __init__(self, model, task_id):
        self.model, self.task_id = model, task_id

    def __call__(self, positions):
        return (self.model.position_embedding(positions)
                + self.model.task_embedding.weight[self.task_id])


class _TaskView:
    """Ephemeral parameter-sharing view, with no mutation of task/model state.

    Both reference decoding and the established incremental cache see the same
    task-conditioned positional embeddings. A cache is still case-local.
    """
    def __init__(self, model, task_id):
        self.model = model
        self.position_embedding = _TaskPositions(model, task_id)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _decode(self, **payload):
        return FormationSpecialist._decode(self, **payload)

    def forward(self, **payload):
        return FormationSpecialist.forward(self, **payload)

    def next_token_logits(self, **payload):
        return FormationSpecialist.next_token_logits(self, **payload)


class FormationSequenceSpecialist(FormationDecisionSpecialist):
    def __init__(self, config):
        super().__init__(config)
        self.task_embedding = torch.nn.Embedding(len(TASKS), config.width)
        with torch.no_grad():
            self.task_embedding.weight.mul_(.02)
            self.task_embedding.weight[0].zero_()

    def bind_task(self, task='formation'):
        if task not in TASKS:
            raise ValueError('unregistered specialist task')
        return _TaskView(self, TASKS.index(task))

    def forward(self, *, task='formation', **payload):
        return FormationSpecialist.forward(self.bind_task(task), **payload)

    def _decode(self, *, task='formation', **payload):
        return FormationSpecialist._decode(self.bind_task(task), **payload)

    def next_token_logits(self, *, task='formation', **payload):
        return FormationSpecialist.next_token_logits(self.bind_task(task), **payload)

    def load_formation_parent(self, state):
        expected = set(self.state_dict()) - {'task_embedding.weight'}
        if set(state) != expected:
            raise ValueError('parent must contain exactly original formation parameters')
        if any(state[key].shape != self.state_dict()[key].shape for key in expected):
            raise ValueError('parent parameter geometry differs')
        if torch.count_nonzero(self.task_embedding.weight[0]).item():
            raise ValueError('parent migration requires an untouched zero formation task')
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ['task_embedding.weight'] or result.unexpected_keys:
            raise ValueError('parent migration changed the declared parameter set')
