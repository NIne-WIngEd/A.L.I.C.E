from cognitive_kernel import open_raw_buffer_store
from raw_buffer_helpers import capture, make_scope


def test_reopen_preserves_references_and_objects(tmp_path):
    scope = make_scope()
    root = tmp_path / "raw-buffer"
    repository = tmp_path / "repository-marker"
    repository.mkdir()
    with open_raw_buffer_store(
        root,
        scope=scope,
        repository_root=repository,
        created_at="2026-08-02T21:00:00Z",
    ) as store:
        receipt = capture(store, b"reopen-bytes", "logical-reopen")
    with open_raw_buffer_store(
        root,
        scope=scope,
        repository_root=repository,
    ) as reopened:
        assert reopened.get_reference(receipt.reference.reference_id) == receipt.reference
        assert reopened.load_opaque_payload(receipt.reference.reference_id) == b"reopen-bytes"
        assert reopened.verify_integrity().logical_reference_count == 1
