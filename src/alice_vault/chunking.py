from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_SCHEMA_VERSION = 1


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkingPolicy:
    policy_id: str
    algorithm_version: str
    normalization_version: str
    max_chars: int
    target_chars: int
    min_chars: int
    overlap_chars: int
    boundary_penalties: dict[str, int]
    digest: str
    source_path: Path


@dataclass(frozen=True)
class ChunkSpan:
    index: int
    start: int
    end: int
    text: str
    text_sha256: str


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "chunking_policy.json"


def load_chunking_policy(path: Path | None = None) -> ChunkingPolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("chunking_policy_schema_version", -1)) != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported chunking-policy schema version")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    policy = ChunkingPolicy(
        policy_id=str(data["policy_id"]),
        algorithm_version=str(data["algorithm_version"]),
        normalization_version=str(data["normalization_version"]),
        max_chars=int(data["max_chars"]),
        target_chars=int(data["target_chars"]),
        min_chars=int(data["min_chars"]),
        overlap_chars=int(data["overlap_chars"]),
        boundary_penalties={str(k): int(v) for k, v in data["boundary_penalties"].items()},
        digest=hashlib.sha256(canonical).hexdigest(),
        source_path=source,
    )
    validate_policy(policy)
    return policy


def validate_policy(policy: ChunkingPolicy) -> None:
    if not 1 <= policy.min_chars <= policy.target_chars <= policy.max_chars:
        raise ValueError("Expected min_chars <= target_chars <= max_chars")
    if not 0 <= policy.overlap_chars < policy.min_chars:
        raise ValueError("overlap_chars must be less than min_chars")
    if {"paragraph", "sentence", "line", "space"} - set(policy.boundary_penalties):
        raise ValueError("Boundary penalties are incomplete")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _boundaries(text: str, lower: int, hard: int) -> list[tuple[int, str]]:
    start = max(0, lower - 2)
    window = text[start:hard]
    found: list[tuple[int, str]] = []
    patterns = [
        (r"\n\n+", "paragraph", False),
        (r"[.!?](?:[\"')\]}]+)?(?=\s)", "sentence", True),
        (r"\n", "line", False),
        (r"[ \t]+", "space", False),
    ]
    for pattern, kind, use_end in patterns:
        for match in re.finditer(pattern, window):
            endpoint = start + (match.end() if use_end else match.start())
            if lower <= endpoint <= hard:
                found.append((endpoint, kind))
    return found


def _choose_end(text: str, start: int, policy: ChunkingPolicy) -> int:
    hard = min(len(text), start + policy.max_chars)
    if hard == len(text):
        return hard
    lower = min(len(text), start + policy.min_chars)
    target = min(len(text), start + policy.target_chars)
    candidates = _boundaries(text, lower, hard)
    if not candidates:
        return hard
    return min(
        candidates,
        key=lambda item: (
            abs(item[0] - target) + policy.boundary_penalties[item[1]],
            -item[0],
        ),
    )[0]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _next_start(text: str, start: int, end: int, policy: ChunkingPolicy) -> int:
    pos = max(start + 1, end - policy.overlap_chars)
    if pos < end and pos > 0 and text[pos - 1].isalnum() and text[pos].isalnum():
        while pos < end and text[pos].isalnum():
            pos += 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def chunk_text(value: str, policy: ChunkingPolicy) -> tuple[str, list[ChunkSpan]]:
    text = normalize_text(value)
    if not text:
        raise ValueError("Cannot chunk empty normalized text")
    spans: list[ChunkSpan] = []
    start = 0
    while start < len(text):
        end = _choose_end(text, start, policy)
        left, right = _trim(text, start, end)
        if right <= left:
            raise RuntimeError("Chunker failed to make progress")
        chunk = text[left:right]
        spans.append(ChunkSpan(len(spans), left, right, chunk, sha256_text(chunk)))
        if end >= len(text):
            break
        next_start = _next_start(text, left, right, policy)
        start = next_start if next_start > start else start + 1
    if spans[0].start != 0 or spans[-1].end != len(text):
        raise RuntimeError("Chunks do not cover normalized text edges")
    for previous, current in zip(spans, spans[1:]):
        if current.start >= previous.end or current.end <= previous.end:
            raise RuntimeError("Invalid overlap or chunk progression")
        if len(current.text) > policy.max_chars:
            raise RuntimeError("Chunk exceeds maximum size")
    return text, spans


def stable_chunk_id(source_sha256: str, policy_digest: str, span: ChunkSpan) -> str:
    payload = "\0".join(
        [
            "alice-chunk-v1",
            source_sha256,
            policy_digest,
            str(span.index),
            str(span.start),
            str(span.end),
            span.text_sha256,
        ]
    )
    return sha256_text(payload)


def stable_chunk_set_id(pilot_hash: str, registry_digest: str, policy_digest: str) -> str:
    return sha256_text(
        "\0".join(["alice-chunk-set-v1", pilot_hash, registry_digest, policy_digest])
    )[:32]
