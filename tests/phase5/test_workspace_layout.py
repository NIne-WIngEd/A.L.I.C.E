from dataclasses import replace

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    WorkspaceLayout,
    default_workspace_layout_mode,
)
from attention_workspace_helpers import digest, scope


def test_ratified_adaptive_composition_table():
    cases = {
        (0, 0): "empty",
        (1, 1): "full_workspace",
        (2, 2): "focus_support_split",
        (3, 3): "primary_two_secondary",
        (4, 4): "adaptive_grid",
        (5, 5): "panels_live_cards",
        (6, 6): "panels_live_cards",
        (7, 7): "command_center",
        (10, 10): "command_center",
        (10, 11): "highest_value_set",
    }
    for counts, expected in cases.items():
        assert (
            default_workspace_layout_mode(
                visible_count=counts[0],
                total_candidate_count=counts[1],
            )
            == expected
        )


def test_layout_records_every_visible_or_omitted_candidate_without_slots():
    layout = WorkspaceLayout.create(
        layout_key="layout",
        scope=scope(),
        created_at="2026-08-01T10:00:00Z",
        visible_count=2,
        total_candidate_count=4,
        max_visible=2,
        layout_locked=False,
        stability_anchor_digest=digest("anchor"),
        item_order=("workspace-item-1", "workspace-item-2"),
        omitted_reference_digests=(digest("o1"), digest("o2")),
    )
    assert layout.layout_mode == "focus_support_split"
    with pytest.raises(CognitiveKernelContractError):
        replace(layout, item_order=("workspace-item-1", "placeholder-2")).validate()


def test_layout_rejects_unaccounted_candidates():
    with pytest.raises(CognitiveKernelContractError):
        WorkspaceLayout.create(
            layout_key="bad",
            scope=scope(),
            created_at="2026-08-01T10:00:00Z",
            visible_count=1,
            total_candidate_count=2,
            max_visible=2,
            layout_locked=False,
            stability_anchor_digest=digest("anchor"),
            item_order=("workspace-item-1",),
            omitted_reference_digests=(),
        )
