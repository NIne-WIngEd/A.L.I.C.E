import pytest

from cognitive_kernel import RawBufferIsolationError, open_raw_buffer_store
from raw_buffer_helpers import capture, make_scope, open_store


def test_store_root_is_bound_to_one_scope(tmp_path):
    repository = tmp_path / "repository-marker"
    repository.mkdir()
    scope = make_scope()
    with open_store(tmp_path, scope=scope) as store:
        capture(store, b"scope-bytes", "logical-scope")
    other = make_scope(host_instance_id="other-host")
    with pytest.raises(RawBufferIsolationError):
        open_raw_buffer_store(
            tmp_path / "raw-buffer",
            scope=other,
            repository_root=repository,
        )


def test_same_bytes_in_two_scopes_have_distinct_physical_identity(tmp_path):
    repository = tmp_path / "repository-marker"
    repository.mkdir()
    alice = make_scope(host_instance_id="host-a")
    other = make_scope(host_instance_id="host-b")
    with open_raw_buffer_store(
        tmp_path / "a",
        scope=alice,
        repository_root=repository,
        created_at="2026-08-02T21:00:00Z",
    ) as first_store:
        first = capture(first_store, b"same", "logical-a")
    with open_raw_buffer_store(
        tmp_path / "b",
        scope=other,
        repository_root=repository,
        created_at="2026-08-02T21:00:00Z",
    ) as second_store:
        second = capture(second_store, b"same", "logical-b")
    assert first.reference.payload_object_id != second.reference.payload_object_id
