"""Small JSON-over-HTTP transport boundary for local model adapters."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model import (
    CancellationToken,
    ConversationModelTimeoutError,
)


class ConversationTransportError(RuntimeError):
    """Raised when an HTTP transport cannot reach the configured provider."""


@dataclass(frozen=True)
class JsonHttpResponse:
    """Provider-neutral HTTP response used by model adapters."""

    status_code: int
    body: bytes
    headers: Mapping[str, str]

    def json_object(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Provider response is not valid UTF-8 JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Provider response JSON root must be an object.")
        return payload


class JsonHttpTransport(Protocol):
    """Injectable JSON HTTP transport for deterministic adapter tests."""

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> JsonHttpResponse:
        """POST one JSON object and return a bounded response."""


@dataclass(frozen=True)
class UrllibJsonHttpTransport:
    """Standard-library HTTP transport for loopback model providers."""

    max_response_bytes: int = 2_000_000

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> JsonHttpResponse:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if timeout_seconds <= 0:
            raise ConversationTransportError(
                "HTTP timeout must be greater than zero."
            )
        if self.max_response_bytes <= 0:
            raise ConversationTransportError(
                "HTTP response limit must be greater than zero."
            )
        data = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read(self.max_response_bytes + 1)
                status_code = int(response.status)
                headers = {key: value for key, value in response.headers.items()}
        except HTTPError as exc:
            body = exc.read(self.max_response_bytes + 1)
            status_code = int(exc.code)
            headers = {key: value for key, value in exc.headers.items()}
        except (TimeoutError, socket.timeout) as exc:
            raise ConversationModelTimeoutError(
                "Local model provider exceeded its approved timeout."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ConversationModelTimeoutError(
                    "Local model provider exceeded its approved timeout."
                ) from exc
            raise ConversationTransportError(
                "Unable to reach the configured local model provider."
            ) from exc
        except OSError as exc:
            raise ConversationTransportError(
                "Unable to communicate with the configured local model provider."
            ) from exc
        if len(body) > self.max_response_bytes:
            raise ConversationTransportError(
                "Local model provider response exceeded the approved byte limit."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return JsonHttpResponse(
            status_code=status_code,
            body=body,
            headers=headers,
        )
