from raw_buffer_helpers import capture, open_store


def test_same_scope_deduplicates_bytes_but_not_logical_references(tmp_path):
    with open_store(tmp_path) as store:
        first = capture(store, b"same-sealed-bytes", "logical-first")
        second = capture(store, b"same-sealed-bytes", "logical-second")
        assert first.reference.reference_id != second.reference.reference_id
        assert first.reference.payload_object_id == second.reference.payload_object_id
        assert first.physical_object_created is True
        assert second.deduplicated is True
        accounting = store.accounting()
        assert accounting.logical_reference_count == 2
        assert accounting.physical_object_count == 1
        assert accounting.logical_bytes == 2 * len(b"same-sealed-bytes")
        assert accounting.physical_bytes == len(b"same-sealed-bytes")


def test_identical_logical_reference_is_rejected_without_extra_object(tmp_path):
    from cognitive_kernel import DuplicateRawBufferReferenceError
    with open_store(tmp_path) as store:
        capture(store, b"same-sealed-bytes", "logical-repeat")
        try:
            capture(store, b"same-sealed-bytes", "logical-repeat")
        except DuplicateRawBufferReferenceError:
            pass
        else:
            raise AssertionError("duplicate logical reference was accepted")
        accounting = store.accounting()
        assert accounting.logical_reference_count == 1
        assert accounting.physical_object_count == 1
