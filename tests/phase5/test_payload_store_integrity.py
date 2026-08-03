import pytest

from cognitive_kernel import RawBufferIntegrityError
from raw_buffer_helpers import capture, open_store


def test_payload_tampering_is_detected(tmp_path):
    with open_store(tmp_path) as store:
        receipt = capture(store, b"integrity-bytes", "logical-integrity")
        object_path = store._object_path(receipt.reference.content_digest)
        object_path.write_bytes(b"tampered")
        with pytest.raises(RawBufferIntegrityError):
            store.verify_integrity()


def test_orphan_object_is_detected(tmp_path):
    with open_store(tmp_path) as store:
        capture(store, b"known", "logical-known")
        orphan = store.objects_root / "aa" / "bb" / "orphan.payload"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"orphan")
        with pytest.raises(RawBufferIntegrityError):
            store.verify_integrity()


def test_reference_metadata_tampering_is_detected(tmp_path):
    with open_store(tmp_path) as store:
        receipt = capture(store, b"metadata-bytes", "logical-metadata")
        store._connection.execute("DROP TRIGGER raw_buffer_references_no_update")
        store._connection.execute(
            "UPDATE raw_buffer_references SET byte_length = byte_length + 1 "
            "WHERE reference_id = ?",
            (receipt.reference.reference_id,),
        )
        store._connection.commit()
        with pytest.raises(RawBufferIntegrityError):
            store.verify_integrity()
