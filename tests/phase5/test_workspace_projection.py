from dataclasses import replace

import pytest

from cognitive_kernel import CognitiveKernelContractError, WorkspaceProjection
from attention_workspace_helpers import decision, entry, item, layout, scope


def test_projection_binds_selected_attention_entries_and_layout():
    first = entry(key="one", reference="node-1", rank=1)
    second = entry(key="two", reference="node-2", rank=2)
    attention = decision((first, second), limit=2)
    items = (
        item(first, key="one", role="primary"),
        item(second, key="two", role="secondary"),
    )
    projection = WorkspaceProjection.create(
        projection_key="projection-1",
        scope=scope(),
        attention_decision=attention,
        projected_at="2026-08-01T10:12:00Z",
        audience="host",
        layout=layout(items),
        items=items,
        policy_bindings=("workspace-projection-v1",),
    )
    record = projection.metadata_record()
    assert record["layout"]["visible_count"] == 2
    assert "title" not in record and "content" not in record
    with pytest.raises(CognitiveKernelContractError):
        replace(projection, attention_decision_sha256="0" * 64).validate(
            attention_decision=attention
        )


def test_non_host_projection_requires_redaction_and_excludes_restricted_items():
    selected = entry(key="sensitive", reference="node-sensitive", rank=1)
    attention = decision((selected,), limit=1)
    unredacted = item(
        selected,
        key="sensitive",
        role="primary",
        privacy="sensitive",
        redaction="none",
    )
    with pytest.raises(CognitiveKernelContractError):
        WorkspaceProjection.create(
            projection_key="bad-sensitive",
            scope=scope(),
            attention_decision=attention,
            projected_at="2026-08-01T10:12:00Z",
            audience="non_host",
            layout=layout((unredacted,)),
            items=(unredacted,),
        )
    redacted = item(
        selected,
        key="sensitive-redacted",
        role="primary",
        privacy="sensitive",
        redaction="title_hidden",
    )
    projection = WorkspaceProjection.create(
        projection_key="safe-sensitive",
        scope=scope(),
        attention_decision=attention,
        projected_at="2026-08-01T10:12:00Z",
        audience="non_host",
        layout=layout((redacted,)),
        items=(redacted,),
    )
    assert projection.items[0].redaction_state == "title_hidden"

    restricted = item(
        selected,
        key="restricted",
        role="primary",
        privacy="restricted",
        redaction="metadata_only",
    )
    with pytest.raises(CognitiveKernelContractError):
        WorkspaceProjection.create(
            projection_key="bad-restricted",
            scope=scope(),
            attention_decision=attention,
            projected_at="2026-08-01T10:12:00Z",
            audience="non_host",
            layout=layout((restricted,)),
            items=(restricted,),
        )


def test_projection_rejects_unselected_attention_entries():
    hidden = entry(
        key="hidden",
        reference="node-hidden",
        rank=1,
        selected=False,
        host_override="background",
        suppression_reason="host_background",
    )
    attention = decision((hidden,), limit=1)
    projected = item(hidden, key="hidden", role="primary")
    with pytest.raises(CognitiveKernelContractError):
        WorkspaceProjection.create(
            projection_key="bad-selection",
            scope=scope(),
            attention_decision=attention,
            projected_at="2026-08-01T10:12:00Z",
            audience="host",
            layout=layout((projected,)),
            items=(projected,),
        )
