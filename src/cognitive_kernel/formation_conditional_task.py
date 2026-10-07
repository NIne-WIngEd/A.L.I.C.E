"""Explicit parameter-free binding for public conditional foundation learning.

The pinned source processor already supplies the task tag. This ephemeral view
adds no task vector, positional change, pretrained weights or checkpoint prefix.
Later full formation learning has its own caller/source/target contract; this
public task name does not limit the specialist's eventual role capabilities.
"""
from __future__ import annotations


class ConditionalTaskView:
    """Delegate to the underlying specialist without registering any tensors."""

    def __init__(self, specialist, *, task: str):
        if task != 'conditional_meaning_to_text':
            raise ValueError('unregistered public conditional specialist task')
        self.specialist = specialist
        self.task = task

    def __getattr__(self, name):
        # Includes the actual configuration, embeddings, projection and decoder
        # consumed by the existing case-local incremental cache.
        return getattr(self.specialist, name)

    def forward(self, **payload):
        return self.specialist.forward(**payload)

    def next_token_logits(self, **payload):
        return self.specialist.next_token_logits(**payload)


def bind_conditional_task(specialist, *, task: str) -> ConditionalTaskView:
    """Bind the explicit public objective while retaining original weight keys."""
    return ConditionalTaskView(specialist, task=task)
