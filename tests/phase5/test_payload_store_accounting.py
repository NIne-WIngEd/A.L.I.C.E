from raw_buffer_helpers import capture, open_store


def test_accounting_separates_logical_and_physical_bytes(tmp_path):
    with open_store(tmp_path) as store:
        capture(store, b"abc", "logical-one")
        capture(store, b"abc", "logical-two")
        capture(store, b"longer", "logical-three")
        accounting = store.accounting()
        assert accounting.logical_reference_count == 3
        assert accounting.physical_object_count == 2
        assert accounting.logical_bytes == 3 + 3 + 6
        assert accounting.physical_bytes == 3 + 6
        assert accounting.deduplicated_bytes == 3
