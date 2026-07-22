"""Cryptographic primitives and local key protection for sensitive memory.

HIGHLY_SENSITIVE plaintext is encrypted with AES-256-GCM before persistence.
The AES master key is stored separately from the Memory Core database and is
protected by an OS-bound key protector. Windows production uses DPAPI; tests
inject an in-memory protector so CI never depends on a real user profile.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .store import default_repository_root, validate_private_database_path

SENSITIVE_ALGORITHM = "AES-256-GCM"
SENSITIVE_AAD_VERSION = 1
SENSITIVE_MASTER_KEY_BYTES = 32
SENSITIVE_NONCE_BYTES = 12
SENSITIVE_KEY_RELATIVE_PATH = Path(
    "memory",
    "phase2",
    "keys",
    "sensitive-master-key.dpapi",
)


class SensitiveCryptoError(RuntimeError):
    """Base error for sensitive-memory cryptography."""


class SensitiveKeyProtectionError(SensitiveCryptoError):
    """Raised when the sensitive-memory master key cannot be protected."""


class SensitivePayloadIntegrityError(SensitiveCryptoError):
    """Raised when an encrypted payload fails authenticated decryption."""


class SensitiveKeyProtector(Protocol):
    """Protect and unprotect the local AES master key."""

    @property
    def protector_id(self) -> str: ...

    def protect(self, plaintext: bytes) -> bytes: ...

    def unprotect(self, protected: bytes) -> bytes: ...


@dataclass(frozen=True)
class EncryptedSensitivePayload:
    ciphertext: bytes
    nonce: bytes
    algorithm: str
    key_id: str
    aad_version: int


class InMemoryTestKeyProtector:
    """Deterministic injectable key protector for tests only.

    This keeps the wrapping key in process memory and must not be selected as a
    production default. AES-GCM is used so tests still exercise authenticated
    key wrapping and corruption detection.
    """

    def __init__(self, wrapping_key: bytes | None = None) -> None:
        self._wrapping_key = wrapping_key or bytes(range(32))
        if len(self._wrapping_key) != 32:
            raise ValueError("Test wrapping key must be exactly 32 bytes.")

    @property
    def protector_id(self) -> str:
        return "test-in-memory-aesgcm"

    def protect(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(SENSITIVE_NONCE_BYTES)
        ciphertext = AESGCM(self._wrapping_key).encrypt(
            nonce,
            plaintext,
            b"alice-sensitive-master-key-test-wrap-v1",
        )
        return nonce + ciphertext

    def unprotect(self, protected: bytes) -> bytes:
        if len(protected) <= SENSITIVE_NONCE_BYTES:
            raise SensitiveKeyProtectionError("Protected test key blob is truncated.")
        nonce = protected[:SENSITIVE_NONCE_BYTES]
        ciphertext = protected[SENSITIVE_NONCE_BYTES:]
        try:
            return AESGCM(self._wrapping_key).decrypt(
                nonce,
                ciphertext,
                b"alice-sensitive-master-key-test-wrap-v1",
            )
        except InvalidTag as exc:
            raise SensitiveKeyProtectionError(
                "Protected test key blob failed authentication."
            ) from exc


class WindowsDPAPIKeyProtector:
    """Protect the sensitive-memory master key with Windows user-scoped DPAPI."""

    _CRYPTPROTECT_UI_FORBIDDEN = 0x1

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.c_uint32),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise SensitiveKeyProtectionError(
                "Windows DPAPI key protection is available only on Windows."
            )
        self._crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)

        self._crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(self._DATA_BLOB),
            ctypes.c_wchar_p,
            ctypes.POINTER(self._DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(self._DATA_BLOB),
        ]
        self._crypt32.CryptProtectData.restype = ctypes.c_int
        self._crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(self._DATA_BLOB),
            ctypes.c_void_p,
            ctypes.POINTER(self._DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(self._DATA_BLOB),
        ]
        self._crypt32.CryptUnprotectData.restype = ctypes.c_int
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p

    @property
    def protector_id(self) -> str:
        return "windows-dpapi-current-user"

    @classmethod
    def _input_blob(cls, data: bytes) -> tuple[ctypes.Array, _DATA_BLOB]:
        buffer = ctypes.create_string_buffer(data)
        blob = cls._DATA_BLOB(
            len(data),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        return buffer, blob

    def _run(self, *, protect: bool, data: bytes) -> bytes:
        _buffer, input_blob = self._input_blob(data)
        output_blob = self._DATA_BLOB()
        function = (
            self._crypt32.CryptProtectData
            if protect
            else self._crypt32.CryptUnprotectData
        )
        description = "A.L.I.C.E. Phase 2 sensitive-memory master key" if protect else None
        result = function(
            ctypes.byref(input_blob),
            description,
            None,
            None,
            None,
            self._CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        if not result:
            error_code = ctypes.get_last_error()
            raise SensitiveKeyProtectionError(
                f"Windows DPAPI operation failed with error {error_code}."
            )
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            if output_blob.pbData:
                self._kernel32.LocalFree(output_blob.pbData)

    def protect(self, plaintext: bytes) -> bytes:
        return self._run(protect=True, data=plaintext)

    def unprotect(self, protected: bytes) -> bytes:
        return self._run(protect=False, data=protected)


def sensitive_key_path(
    vault_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    vault = Path(vault_root).expanduser().resolve(strict=True)
    repository = (
        default_repository_root()
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=True)
    )
    candidate = (vault / SENSITIVE_KEY_RELATIVE_PATH).resolve(strict=False)
    return validate_private_database_path(
        candidate,
        repository_root=repository,
    )


class SensitiveMasterKeyStore:
    """Load or create the separately protected AES-256 sensitive-memory key."""

    def __init__(
        self,
        vault_root: str | Path,
        *,
        protector: SensitiveKeyProtector,
        repository_root: str | Path | None = None,
    ) -> None:
        self.path = sensitive_key_path(
            vault_root,
            repository_root=repository_root,
        )
        self.protector = protector

    def _decode_key(self, protected: bytes) -> bytes:
        key = self.protector.unprotect(protected)
        if len(key) != SENSITIVE_MASTER_KEY_BYTES:
            raise SensitiveKeyProtectionError(
                "Sensitive-memory master key has an invalid length."
            )
        return key

    def load_or_create_key(self) -> bytes:
        if self.path.exists():
            return self._decode_key(self.path.read_bytes())

        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = AESGCM.generate_key(bit_length=256)
        protected = self.protector.protect(key)

        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return self._decode_key(self.path.read_bytes())

        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(protected)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                raise

        return key


def sensitive_key_id(master_key: bytes) -> str:
    if len(master_key) != SENSITIVE_MASTER_KEY_BYTES:
        raise SensitiveCryptoError("Sensitive-memory AES key must be 256 bits.")
    return hashlib.sha256(master_key).hexdigest()


def sensitive_payload_aad(
    *,
    memory_id: str,
    content_sha256: str,
    aad_version: int = SENSITIVE_AAD_VERSION,
) -> bytes:
    payload = {
        "aad_version": aad_version,
        "content_sha256": content_sha256,
        "data_classification": "HIGHLY_SENSITIVE",
        "memory_id": memory_id,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def encrypt_sensitive_payload(
    *,
    master_key: bytes,
    memory_id: str,
    content: str,
    content_sha256: str,
) -> EncryptedSensitivePayload:
    key_id = sensitive_key_id(master_key)
    nonce = os.urandom(SENSITIVE_NONCE_BYTES)
    aad = sensitive_payload_aad(
        memory_id=memory_id,
        content_sha256=content_sha256,
    )
    ciphertext = AESGCM(master_key).encrypt(
        nonce,
        content.encode("utf-8"),
        aad,
    )
    return EncryptedSensitivePayload(
        ciphertext=ciphertext,
        nonce=nonce,
        algorithm=SENSITIVE_ALGORITHM,
        key_id=key_id,
        aad_version=SENSITIVE_AAD_VERSION,
    )


def decrypt_sensitive_payload(
    *,
    master_key: bytes,
    memory_id: str,
    content_sha256: str,
    payload: EncryptedSensitivePayload,
) -> str:
    if payload.algorithm != SENSITIVE_ALGORITHM:
        raise SensitiveCryptoError(
            f"Unsupported sensitive payload algorithm: {payload.algorithm!r}"
        )
    if payload.aad_version != SENSITIVE_AAD_VERSION:
        raise SensitiveCryptoError(
            f"Unsupported sensitive payload AAD version: {payload.aad_version}"
        )
    if payload.key_id != sensitive_key_id(master_key):
        raise SensitivePayloadIntegrityError(
            "Sensitive payload key identifier does not match the active key."
        )

    aad = sensitive_payload_aad(
        memory_id=memory_id,
        content_sha256=content_sha256,
        aad_version=payload.aad_version,
    )
    try:
        plaintext = AESGCM(master_key).decrypt(
            payload.nonce,
            payload.ciphertext,
            aad,
        )
    except InvalidTag as exc:
        raise SensitivePayloadIntegrityError(
            "Sensitive payload failed authenticated decryption."
        ) from exc

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SensitivePayloadIntegrityError(
            "Sensitive payload decrypted to invalid UTF-8."
        ) from exc
