from __future__ import annotations

import copy
import hashlib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICY_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class AtomicClaimDecompositionPolicy:
    policy_id: str
    enabled: bool
    model: str
    ollama_endpoint: str
    request_timeout_seconds: int
    request_retry_count: int
    request_retry_backoff_seconds: float
    keep_alive: str | int
    maximum_output_tokens: int
    temperature: float
    think: bool
    maximum_atomic_claims_per_parent: int
    maximum_total_atomic_claims: int
    private_output_only: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    digest: str
    source_path: Path


def default_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "atomic_claim_decomposition_policy.json"
    )


def load_atomic_claim_decomposition_policy(
    path: Path | None = None,
) -> AtomicClaimDecompositionPolicy:
    source = (
        path or default_policy_path()
    ).expanduser().resolve(strict=True)

    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    if (
        int(
            data.get(
                "atomic_claim_decomposition_policy_schema_version",
                -1,
            )
        )
        != POLICY_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported atomic-claim decomposition policy schema"
        )

    parsed = urllib.parse.urlparse(
        str(data["ollama_endpoint"])
    )
    if parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError(
            "Atomic claim decomposition requires local Ollama"
        )

    policy = AtomicClaimDecompositionPolicy(
        policy_id=str(data["policy_id"]),
        enabled=bool(data["enabled"]),
        model=str(data["model"]),
        ollama_endpoint=str(data["ollama_endpoint"]),
        request_timeout_seconds=int(
            data["request_timeout_seconds"]
        ),
        request_retry_count=int(
            data["request_retry_count"]
        ),
        request_retry_backoff_seconds=float(
            data["request_retry_backoff_seconds"]
        ),
        keep_alive=data["keep_alive"],
        maximum_output_tokens=int(
            data["maximum_output_tokens"]
        ),
        temperature=float(data["temperature"]),
        think=bool(data["think"]),
        maximum_atomic_claims_per_parent=int(
            data["maximum_atomic_claims_per_parent"]
        ),
        maximum_total_atomic_claims=int(
            data["maximum_total_atomic_claims"]
        ),
        private_output_only=bool(
            data["private_output_only"]
        ),
        memory_write_allowed=bool(
            data["memory_write_allowed"]
        ),
        external_action_allowed=bool(
            data["external_action_allowed"]
        ),
        tool_calling_allowed=bool(
            data["tool_calling_allowed"]
        ),
        web_access_allowed=bool(
            data["web_access_allowed"]
        ),
        digest=_sha256(_canonical_json(data)),
        source_path=source,
    )

    if policy.maximum_atomic_claims_per_parent < 1:
        raise ValueError(
            "maximum_atomic_claims_per_parent must be positive"
        )
    if policy.maximum_total_atomic_claims < 1:
        raise ValueError(
            "maximum_total_atomic_claims must be positive"
        )
    if policy.maximum_output_tokens < 256:
        raise ValueError(
            "maximum_output_tokens is too small"
        )
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError(
            "Atomic decomposition must remain read-only and offline"
        )
    if not policy.private_output_only:
        raise ValueError(
            "Atomic decomposition output must remain private"
        )

    return policy


def _schema(
    *,
    parent_count: int,
    maximum_total_atomic_claims: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "atomic_claims": {
                "type": "array",
                "maxItems": maximum_total_atomic_claims,
                "items": {
                    "type": "object",
                    "properties": {
                        "parent_claim_index": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max(
                                1,
                                parent_count,
                            ),
                        },
                        "text": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "parent_claim_index",
                        "text",
                    ],
                },
            }
        },
        "required": [
            "atomic_claims"
        ],
    }


def _system_prompt() -> str:
    return """You decompose already-generated grounded claims into smaller
atomic claims for private factual verification.

You are NOT answering the user's question and you are NOT adding new facts.

Rules:
1. Preserve only information already asserted by each parent claim.
2. Split compound claims into the smallest independently checkable factual
   propositions that still make sense on their own.
3. Do not add dates, names, roles, achievements, numbers, causal claims, or
   technical details that are not already present in the parent claim.
4. Do not strengthen certainty.
5. Do not combine facts from different parent claims.
6. Make each atomic claim self-contained enough for an NLI verifier.
7. For claims about the user, prefer "The user ..." instead of second-person
   pronouns such as "you".
8. A simple already-atomic claim may remain as one atomic claim.
9. Return every parent claim as at least one atomic claim unless the parent is
   empty or meaningless.
10. Return only the required structured JSON object.
"""


def _user_prompt(
    claims: list[dict[str, Any]],
) -> str:
    lines = [
        "PARENT CLAIMS",
    ]
    for index, claim in enumerate(
        claims,
        start=1,
    ):
        lines.extend(
            [
                f"Parent claim {index}:",
                str(
                    claim.get(
                        "text",
                        "",
                    )
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _is_timeout(
    exc: BaseException,
) -> bool:
    if isinstance(
        exc,
        (TimeoutError, socket.timeout),
    ):
        return True
    if isinstance(
        exc,
        urllib.error.URLError,
    ):
        return isinstance(
            exc.reason,
            (TimeoutError, socket.timeout),
        )
    return False


def ollama_decompose_claims(
    *,
    policy: AtomicClaimDecompositionPolicy,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "model": policy.model,
        "system": _system_prompt(),
        "prompt": _user_prompt(claims),
        "stream": False,
        "think": policy.think,
        "format": _schema(
            parent_count=len(claims),
            maximum_total_atomic_claims=(
                policy.maximum_total_atomic_claims
            ),
        ),
        "keep_alive": policy.keep_alive,
        "options": {
            "temperature": policy.temperature,
            "num_predict": (
                policy.maximum_output_tokens
            ),
        },
    }
    encoded = json.dumps(
        body
    ).encode("utf-8")
    attempts = (
        policy.request_retry_count + 1
    )

    for attempt in range(
        1,
        attempts + 1,
    ):
        request = urllib.request.Request(
            policy.ollama_endpoint,
            data=encoded,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=(
                    policy.request_timeout_seconds
                ),
            ) as response:
                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )
            break
        except (
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
        ) as exc:
            if (
                not _is_timeout(exc)
                or attempt >= attempts
            ):
                raise RuntimeError(
                    "Atomic claim decomposition failed "
                    f"after {attempt} attempt(s)"
                ) from exc

            delay = (
                policy.request_retry_backoff_seconds
                * attempt
            )
            if delay > 0:
                time.sleep(delay)
    else:  # pragma: no cover
        raise RuntimeError(
            "Atomic claim decomposition failed"
        )

    response_text = payload.get(
        "response"
    )
    if not isinstance(
        response_text,
        str,
    ):
        raise RuntimeError(
            "Atomic claim decomposition returned no response text"
        )

    try:
        structured = json.loads(
            response_text
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Atomic claim decomposition returned invalid JSON"
        ) from exc

    return {
        "structured": structured,
        "runtime": {
            "model": payload.get(
                "model",
                policy.model,
            ),
            "done": payload.get(
                "done"
            ),
            "done_reason": payload.get(
                "done_reason"
            ),
            "prompt_eval_count": payload.get(
                "prompt_eval_count"
            ),
            "eval_count": payload.get(
                "eval_count"
            ),
            "attempt_count": attempt,
        },
    }


def _validate_and_rebuild(
    *,
    parent_claims: list[dict[str, Any]],
    atomic_output: dict[str, Any],
    policy: AtomicClaimDecompositionPolicy,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw = atomic_output.get(
        "atomic_claims",
        []
    )
    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "atomic_claims is not a list"
        )

    by_parent: dict[
        int,
        list[str],
    ] = {
        index: []
        for index in range(
            1,
            len(parent_claims) + 1,
        )
    }

    for item in raw:
        if not isinstance(
            item,
            dict,
        ):
            continue

        parent_index = int(
            item.get(
                "parent_claim_index",
                0,
            )
        )
        if parent_index not in by_parent:
            continue

        text = " ".join(
            str(
                item.get(
                    "text",
                    "",
                )
            ).split()
        ).strip()
        if not text:
            continue

        current = by_parent[
            parent_index
        ]
        if (
            text not in current
            and len(current)
            < policy.maximum_atomic_claims_per_parent
        ):
            current.append(text)

    rebuilt: list[
        dict[str, Any]
    ] = []
    provenance: list[
        dict[str, Any]
    ] = []

    for parent_index, parent in enumerate(
        parent_claims,
        start=1,
    ):
        texts = by_parent[
            parent_index
        ]

        # Fail closed on decomposition omissions by falling back to the
        # original parent claim. The NLI gate will still decide whether it is
        # safe to keep.
        if not texts:
            original = " ".join(
                str(
                    parent.get(
                        "text",
                        "",
                    )
                ).split()
            ).strip()
            if original:
                texts = [
                    original
                ]

        for atomic_index, text in enumerate(
            texts,
            start=1,
        ):
            rebuilt.append(
                {
                    "text": text,
                    "claim_type": str(
                        parent.get(
                            "claim_type",
                            "fact",
                        )
                    ),
                    "citations": list(
                        parent.get(
                            "citations",
                            [],
                        )
                    ),
                }
            )
            provenance.append(
                {
                    "parent_claim_index": (
                        parent_index
                    ),
                    "atomic_claim_index": (
                        atomic_index
                    ),
                    "citations_inherited": list(
                        parent.get(
                            "citations",
                            [],
                        )
                    ),
                }
            )

    if (
        len(rebuilt)
        > policy.maximum_total_atomic_claims
    ):
        rebuilt = rebuilt[
            : policy.maximum_total_atomic_claims
        ]
        provenance = provenance[
            : policy.maximum_total_atomic_claims
        ]

    return rebuilt, provenance


def decompose_model_output_claims(
    *,
    model_output: dict[str, Any],
    policy: AtomicClaimDecompositionPolicy,
    client=None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Decompose claims only; NLI verification remains the authority."""
    output = copy.deepcopy(
        model_output
    )
    parent_claims = list(
        output.get(
            "claims",
            [],
        )
    )

    summary = {
        "enabled": bool(
            policy.enabled
        ),
        "policy_id": (
            policy.policy_id
        ),
        "policy_digest": (
            policy.digest
        ),
        "model": policy.model,
        "input_claim_count": len(
            parent_claims
        ),
        "output_atomic_claim_count": len(
            parent_claims
        ),
        "decomposition_applied": False,
        "private_output_only": True,
        "memory_write_allowed": False,
        "external_action_allowed": False,
        "tool_calling_allowed": False,
        "web_access_allowed": False,
    }

    if (
        not policy.enabled
        or not parent_claims
    ):
        return output, summary

    decomposer = (
        client
        or ollama_decompose_claims
    )
    result = decomposer(
        policy=policy,
        claims=parent_claims,
    )
    structured = result.get(
        "structured",
        {}
    )
    if not isinstance(
        structured,
        dict,
    ):
        raise RuntimeError(
            "Atomic decomposition structured output is invalid"
        )

    atomic_claims, provenance = (
        _validate_and_rebuild(
            parent_claims=parent_claims,
            atomic_output=structured,
            policy=policy,
        )
    )

    output[
        "claims"
    ] = atomic_claims

    summary.update(
        {
            "output_atomic_claim_count": len(
                atomic_claims
            ),
            "decomposition_applied": True,
            "atomic_claim_provenance": (
                provenance
            ),
            "runtime": result.get(
                "runtime",
                {},
            ),
        }
    )
    return output, summary
