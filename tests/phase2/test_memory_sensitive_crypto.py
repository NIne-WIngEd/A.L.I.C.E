"""P2.6a sensitive-memory cryptography tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from alice_memory.sensitive_crypto import (
    EncryptedSensitivePayload,
    InMemoryTestKeyProtector,
    SensitiveMasterKeyStore,
    SensitivePayloadIntegrityError,
    WindowsDPAPIKeyProtector,
    decrypt_sensitive_payload,
    encrypt_sensitive_payload,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_aes_gcm_round_trip() -> None:
    key = bytes(range(32))
    content = "Highly sensitive test content"
    digest = _hash(content)
    payload = encrypt_sensitive_payload(
        master_key=key,
        memory_id="memory-1",
        content=content,
        content_sha256=digest,
    )

    assert payload.ciphertext != content.encode("utf-8")
    assert len(payload.nonce) == 12
    assert decrypt_sensitive_payload(
        master_key=key,
        memory_id="memory-1",
        content_sha256=digest,
        payload=payload,
    ) == content


def test_ciphertext_tampering_fails_authenticated_decryption() -> None:
    key = bytes(range(32))
    content = "Sensitive content"
    digest = _hash(content)
    payload = encrypt_sensitive_payload(
        master_key=key,
        memory_id="memory-1",
        content=content,
        content_sha256=digest,
    )
    tampered = bytearray(payload.ciphertext)
    tampered[0] ^= 1

    with pytest.raises(SensitivePayloadIntegrityError):
        decrypt_sensitive_payload(
            master_key=key,
            memory_id="memory-1",
            content_sha256=digest,
            payload=EncryptedSensitivePayload(
                ciphertext=bytes(tampered),
                nonce=payload.nonce,
                algorithm=payload.algorithm,
                key_id=payload.key_id,
                aad_version=payload.aad_version,
            ),
        )


def test_wrong_key_fails_closed() -> None:
    content = "Sensitive content"
    digest = _hash(content)
    payload = encrypt_sensitive_payload(
        master_key=bytes(range(32)),
        memory_id="memory-1",
        content=content,
        content_sha256=digest,
    )

    with pytest.raises(SensitivePayloadIntegrityError):
        decrypt_sensitive_payload(
            master_key=bytes(reversed(range(32))),
            memory_id="memory-1",
            content_sha256=digest,
            payload=payload,
        )


def test_wrong_associated_data_fails_closed() -> None:
    key = bytes(range(32))
    content = "Sensitive content"
    digest = _hash(content)
    payload = encrypt_sensitive_payload(
        master_key=key,
        memory_id="memory-1",
        content=content,
        content_sha256=digest,
    )

    with pytest.raises(SensitivePayloadIntegrityError):
        decrypt_sensitive_payload(
            master_key=key,
            memory_id="different-memory",
            content_sha256=digest,
            payload=payload,
        )


def test_master_key_store_persists_only_protected_blob(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    protector = InMemoryTestKeyProtector()
    store = SensitiveMasterKeyStore(
        vault,
        protector=protector,
        repository_root=repository,
    )

    first = store.load_or_create_key()
    protected_blob = store.path.read_bytes()
    second = store.load_or_create_key()

    assert first == second
    assert len(first) == 32
    assert first not in protected_blob
    assert protected_blob != first


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_windows_dpapi_round_trip() -> None:
    protector = WindowsDPAPIKeyProtector()
    plaintext = os.urandom(32)
    protected = protector.protect(plaintext)

    assert protected != plaintext
    assert protector.unprotect(protected) == plaintext
