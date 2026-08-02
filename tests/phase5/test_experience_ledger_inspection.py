from __future__ import annotations

from cognitive_kernel import open_experience_ledger
from experience_ledger_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    SHA_B,
    make_event,
    make_scope,
    paths,
)


def test_inspection_is_sanitized_and_paginated(tmp_path) -> None:
    repository, database = paths(tmp_path)
    first = make_event()
    second = make_event(
        event_type="answer-produced",
        content_digest=SHA_B,
        payload_reference=None,
    )
    with open_experience_ledger(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as ledger:
        ledger.append_events((first, second), committed_at=COMMITTED_AT)
        records = ledger.inspect(after_sequence=0, limit=1)
        assert len(records) == 1
        assert records[0].sequence == 1
        page = ledger.inspect(after_sequence=1, limit=10)
        assert [item.sequence for item in page] == [2]
        exposed = records[0].record()
        assert "payload_reference" not in exposed
        assert "provenance" not in exposed
        assert "event_json" not in exposed
