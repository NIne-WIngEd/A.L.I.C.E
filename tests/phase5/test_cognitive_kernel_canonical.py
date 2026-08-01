from __future__ import annotations

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
)


def test_canonical_json_and_digest_are_order_stable() -> None:
    left = {"b": "é", "a": [2, 1]}
    right = {"a": [2, 1], "b": "é"}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(CognitiveKernelContractError):
        canonical_json_bytes({"value": float("nan")})


def test_timestamp_normalizes_to_utc_microseconds() -> None:
    assert (
        normalize_timestamp("2026-08-01T02:00:00-05:00")
        == "2026-08-01T07:00:00.000000Z"
    )


def test_timestamp_requires_timezone() -> None:
    with pytest.raises(CognitiveKernelContractError):
        normalize_timestamp("2026-08-01T02:00:00")
