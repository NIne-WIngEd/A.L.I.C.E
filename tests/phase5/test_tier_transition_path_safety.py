from __future__ import annotations

import pytest

from cognitive_kernel import (
    UnsafeTierTransitionPathError,
    open_tier_transition_store,
)
from tier_transition_helpers import CREATED_AT, make_scope


def test_tier_transition_root_inside_repository_is_rejected(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(UnsafeTierTransitionPathError):
        open_tier_transition_store(
            repository / "tier-store",
            scope=make_scope(),
            repository_root=repository,
            created_at=CREATED_AT,
        )
