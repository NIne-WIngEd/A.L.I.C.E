from pathlib import Path

from cognitive_kernel import ProductHostScope


def make_scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "test-host",
    encryption_domain: str = "test-domain",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=encryption_domain,
    )


def open_store(tmp_path: Path, *, scope: ProductHostScope | None = None):
    from cognitive_kernel import open_raw_buffer_store
    return open_raw_buffer_store(
        tmp_path / "raw-buffer",
        scope=scope or make_scope(),
        repository_root=Path(__file__).resolve().parents[2],
        created_at="2026-08-02T21:00:00Z",
    )


def capture(store, payload: bytes, logical_record_id: str):
    return store.capture(
        payload,
        logical_record_id=logical_record_id,
        media_type="application/octet-stream",
        sensitivity_class="private",
        retention_class="ordinary_experience",
        captured_at="2026-08-02T21:01:00Z",
        host_sealed=True,
    )
