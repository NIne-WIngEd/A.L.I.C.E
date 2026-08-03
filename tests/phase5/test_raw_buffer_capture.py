from raw_buffer_helpers import capture, open_store


def test_capture_and_load_opaque_payload(tmp_path):
    with open_store(tmp_path) as store:
        receipt = capture(store, b"host-sealed-one", "logical-one")
        assert receipt.physical_object_created is True
        assert receipt.deduplicated is False
        assert store.load_opaque_payload(receipt.reference.reference_id) == b"host-sealed-one"
        inspected = store.inspect()
        assert inspected == (receipt.reference,)
        assert "payload" not in receipt.reference.record()
        assert store.verify_integrity().valid is True


def test_plaintext_mode_is_not_accepted(tmp_path):
    with open_store(tmp_path) as store:
        try:
            store.capture(
                b"bytes",
                logical_record_id="logical-unsealed",
                media_type="application/octet-stream",
                sensitivity_class="private",
                retention_class="ordinary_experience",
                captured_at="2026-08-02T21:01:00Z",
                host_sealed=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("unsealed payload was accepted")
