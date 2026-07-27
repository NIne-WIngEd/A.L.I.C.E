"""Fail-closed policy for the Phase 4 P4.2 HTTP retrieval boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy import InformationPolicy

DEFAULT_HTTP_RETRIEVAL_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_http_retrieval_policy.json"
)


class InformationHttpRetrievalPolicyError(ValueError):
    """Raised when the P4.2 HTTP retrieval policy is invalid."""


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationHttpRetrievalPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationHttpRetrievalPolicyError(
            f"{field} must remain {str(expected).lower()} in P4.2."
        )
    return expected


def _bounded_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InformationHttpRetrievalPolicyError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise InformationHttpRetrievalPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def _text_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise InformationHttpRetrievalPolicyError(
            f"{field} must be a non-empty list."
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise InformationHttpRetrievalPolicyError(
            f"{field} must contain non-empty text values."
        )
    normalized = tuple(item.strip().lower() for item in value)
    if len(set(normalized)) != len(normalized):
        raise InformationHttpRetrievalPolicyError(
            f"{field} cannot contain duplicates."
        )
    return normalized


@dataclass(frozen=True)
class InformationHttpRetrievalPolicy:
    """Validated deterministic HTTP retrieval safety policy."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    live_network_access_allowed: bool
    deterministic_resolver_required: bool
    deterministic_transport_required: bool
    allowed_schemes: tuple[str, ...]
    http_ports: tuple[int, ...]
    https_ports: tuple[int, ...]
    max_redirects: int
    max_header_bytes: int
    max_wire_bytes: int
    max_decoded_bytes: int
    request_timeout_seconds: int
    allowed_content_types: tuple[str, ...]
    allowed_content_encodings: tuple[str, ...]
    allowed_charsets: tuple[str, ...]
    all_resolved_addresses_must_be_global: bool
    redirect_revalidation_required: bool
    peer_address_pinning_required: bool
    https_downgrade_allowed: bool
    environment_proxy_allowed: bool
    credentials_allowed: bool
    cookies_allowed: bool
    downloads_allowed: bool
    automatic_retries_allowed: bool

    def validate(self, *, information_policy: InformationPolicy | None = None) -> None:
        if self.policy_name != "alice_information_http_retrieval_policy":
            raise InformationHttpRetrievalPolicyError(
                "P4.2 policy_name must be alice_information_http_retrieval_policy."
            )
        if self.version != "1.0.0":
            raise InformationHttpRetrievalPolicyError(
                "P4.2 HTTP retrieval policy version must be 1.0.0."
            )
        if self.phase != "4" or self.milestone != "P4.2":
            raise InformationHttpRetrievalPolicyError(
                "HTTP retrieval policy must be bound to Phase 4 milestone P4.2."
            )
        if self.status != "controlled_http_boundary":
            raise InformationHttpRetrievalPolicyError(
                "P4.2 status must be controlled_http_boundary."
            )
        required_true = {
            "deterministic_resolver_required": self.deterministic_resolver_required,
            "deterministic_transport_required": self.deterministic_transport_required,
            "all_resolved_addresses_must_be_global": (
                self.all_resolved_addresses_must_be_global
            ),
            "redirect_revalidation_required": self.redirect_revalidation_required,
            "peer_address_pinning_required": self.peer_address_pinning_required,
        }
        required_false = {
            "live_network_access_allowed": self.live_network_access_allowed,
            "https_downgrade_allowed": self.https_downgrade_allowed,
            "environment_proxy_allowed": self.environment_proxy_allowed,
            "credentials_allowed": self.credentials_allowed,
            "cookies_allowed": self.cookies_allowed,
            "downloads_allowed": self.downloads_allowed,
            "automatic_retries_allowed": self.automatic_retries_allowed,
        }
        for field, value in required_true.items():
            if value is not True:
                raise InformationHttpRetrievalPolicyError(
                    f"{field} must remain true in P4.2."
                )
        for field, value in required_false.items():
            if value is not False:
                raise InformationHttpRetrievalPolicyError(
                    f"{field} must remain false in P4.2."
                )
        if self.allowed_schemes != ("http", "https"):
            raise InformationHttpRetrievalPolicyError(
                "P4.2 allowed schemes must remain HTTP and HTTPS."
            )
        if self.http_ports != (80,) or self.https_ports != (443,):
            raise InformationHttpRetrievalPolicyError(
                "P4.2 permits only default HTTP and HTTPS ports."
            )
        _bounded_int(self.max_redirects, field="max_redirects", minimum=0, maximum=5)
        _bounded_int(
            self.max_header_bytes,
            field="max_header_bytes",
            minimum=1024,
            maximum=131072,
        )
        _bounded_int(
            self.max_wire_bytes,
            field="max_wire_bytes",
            minimum=1,
            maximum=5_000_000,
        )
        _bounded_int(
            self.max_decoded_bytes,
            field="max_decoded_bytes",
            minimum=1,
            maximum=5_000_000,
        )
        _bounded_int(
            self.request_timeout_seconds,
            field="request_timeout_seconds",
            minimum=1,
            maximum=30,
        )
        if self.max_decoded_bytes > 5_000_000:
            raise InformationHttpRetrievalPolicyError(
                "P4.2 decoded response limit is too large."
            )
        if self.allowed_content_types != (
            "application/xhtml+xml",
            "text/html",
            "text/plain",
        ):
            raise InformationHttpRetrievalPolicyError(
                "P4.2 content types must remain XHTML, HTML, and plain text."
            )
        if self.allowed_content_encodings != ("identity", "gzip", "deflate"):
            raise InformationHttpRetrievalPolicyError(
                "P4.2 content encodings must remain identity, gzip, and deflate."
            )
        if self.allowed_charsets != ("ascii", "us-ascii", "utf-8"):
            raise InformationHttpRetrievalPolicyError(
                "P4.2 character sets must remain ASCII and UTF-8."
            )
        if information_policy is not None:
            information_policy.validate()
            if self.max_redirects > information_policy.max_redirects:
                raise InformationHttpRetrievalPolicyError(
                    "P4.2 redirect budget exceeds the foundation policy."
                )
            if self.max_wire_bytes > information_policy.max_response_bytes:
                raise InformationHttpRetrievalPolicyError(
                    "P4.2 wire-byte budget exceeds the foundation policy."
                )
            if self.max_decoded_bytes > information_policy.max_response_bytes:
                raise InformationHttpRetrievalPolicyError(
                    "P4.2 decoded-byte budget exceeds the foundation policy."
                )
            if self.request_timeout_seconds > information_policy.request_timeout_seconds:
                raise InformationHttpRetrievalPolicyError(
                    "P4.2 request timeout exceeds the foundation policy."
                )
            if self.allowed_schemes != information_policy.allowed_schemes:
                raise InformationHttpRetrievalPolicyError(
                    "P4.2 URL schemes must match the foundation policy."
                )

    def allowed_ports_for(self, scheme: str) -> tuple[int, ...]:
        if scheme == "http":
            return self.http_ports
        if scheme == "https":
            return self.https_ports
        return ()


def parse_information_http_retrieval_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
) -> InformationHttpRetrievalPolicy:
    """Validate and project one decoded P4.2 retrieval policy."""

    raw_ports = payload.get("allowed_ports")
    if not isinstance(raw_ports, dict):
        raise InformationHttpRetrievalPolicyError(
            "allowed_ports must be an object."
        )

    def ports(name: str) -> tuple[int, ...]:
        raw = raw_ports.get(name)
        if not isinstance(raw, list) or not raw:
            raise InformationHttpRetrievalPolicyError(
                f"allowed_ports.{name} must be a non-empty list."
            )
        projected = tuple(
            _bounded_int(item, field=f"allowed_ports.{name}", minimum=1, maximum=65535)
            for item in raw
        )
        if len(set(projected)) != len(projected):
            raise InformationHttpRetrievalPolicyError(
                f"allowed_ports.{name} cannot contain duplicates."
            )
        return projected

    policy = InformationHttpRetrievalPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        live_network_access_allowed=_strict_bool(
            payload.get("live_network_access_allowed"),
            field="live_network_access_allowed",
            expected=False,
        ),
        deterministic_resolver_required=_strict_bool(
            payload.get("deterministic_resolver_required"),
            field="deterministic_resolver_required",
            expected=True,
        ),
        deterministic_transport_required=_strict_bool(
            payload.get("deterministic_transport_required"),
            field="deterministic_transport_required",
            expected=True,
        ),
        allowed_schemes=_text_tuple(
            payload.get("allowed_schemes"),
            field="allowed_schemes",
        ),
        http_ports=ports("http"),
        https_ports=ports("https"),
        max_redirects=_bounded_int(
            payload.get("max_redirects"),
            field="max_redirects",
            minimum=0,
            maximum=5,
        ),
        max_header_bytes=_bounded_int(
            payload.get("max_header_bytes"),
            field="max_header_bytes",
            minimum=1024,
            maximum=131072,
        ),
        max_wire_bytes=_bounded_int(
            payload.get("max_wire_bytes"),
            field="max_wire_bytes",
            minimum=1,
            maximum=5_000_000,
        ),
        max_decoded_bytes=_bounded_int(
            payload.get("max_decoded_bytes"),
            field="max_decoded_bytes",
            minimum=1,
            maximum=5_000_000,
        ),
        request_timeout_seconds=_bounded_int(
            payload.get("request_timeout_seconds"),
            field="request_timeout_seconds",
            minimum=1,
            maximum=30,
        ),
        allowed_content_types=_text_tuple(
            payload.get("allowed_content_types"),
            field="allowed_content_types",
        ),
        allowed_content_encodings=_text_tuple(
            payload.get("allowed_content_encodings"),
            field="allowed_content_encodings",
        ),
        allowed_charsets=_text_tuple(
            payload.get("allowed_charsets"),
            field="allowed_charsets",
        ),
        all_resolved_addresses_must_be_global=_strict_bool(
            payload.get("all_resolved_addresses_must_be_global"),
            field="all_resolved_addresses_must_be_global",
            expected=True,
        ),
        redirect_revalidation_required=_strict_bool(
            payload.get("redirect_revalidation_required"),
            field="redirect_revalidation_required",
            expected=True,
        ),
        peer_address_pinning_required=_strict_bool(
            payload.get("peer_address_pinning_required"),
            field="peer_address_pinning_required",
            expected=True,
        ),
        https_downgrade_allowed=_strict_bool(
            payload.get("https_downgrade_allowed"),
            field="https_downgrade_allowed",
            expected=False,
        ),
        environment_proxy_allowed=_strict_bool(
            payload.get("environment_proxy_allowed"),
            field="environment_proxy_allowed",
            expected=False,
        ),
        credentials_allowed=_strict_bool(
            payload.get("credentials_allowed"),
            field="credentials_allowed",
            expected=False,
        ),
        cookies_allowed=_strict_bool(
            payload.get("cookies_allowed"),
            field="cookies_allowed",
            expected=False,
        ),
        downloads_allowed=_strict_bool(
            payload.get("downloads_allowed"),
            field="downloads_allowed",
            expected=False,
        ),
        automatic_retries_allowed=_strict_bool(
            payload.get("automatic_retries_allowed"),
            field="automatic_retries_allowed",
            expected=False,
        ),
    )
    policy.validate(information_policy=information_policy)
    return policy


def load_information_http_retrieval_policy(
    path: str | Path = DEFAULT_HTTP_RETRIEVAL_POLICY_PATH,
    *,
    information_policy: InformationPolicy | None = None,
) -> InformationHttpRetrievalPolicy:
    """Load and validate the public P4.2 HTTP retrieval policy."""

    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationHttpRetrievalPolicyError(
            f"Unable to load HTTP retrieval policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise InformationHttpRetrievalPolicyError(
            "HTTP retrieval policy root must be a JSON object."
        )
    return parse_information_http_retrieval_policy(
        payload,
        information_policy=information_policy,
    )
