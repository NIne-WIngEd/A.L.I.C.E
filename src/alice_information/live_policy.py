"""Versioned activation policy for Phase 4 P4.2b direct live HTTPS access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy import InformationPolicy
from .retrieval_policy import InformationHttpRetrievalPolicy

DEFAULT_LIVE_HTTP_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_live_http_policy.json"
)


class InformationLiveHttpPolicyError(ValueError):
    """Raised when the P4.2b live HTTP activation policy is invalid."""


@dataclass(frozen=True)
class InformationLiveHttpPolicy:
    """Exact fail-closed policy for the unregistered direct HTTPS adapter."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    live_network_access_allowed: bool
    approved_resolver_type: str
    approved_transport_type: str
    allowed_schemes: tuple[str, ...]
    operating_system_dns_allowed: bool
    direct_socket_connections_allowed: bool
    environment_proxies_allowed: bool
    automatic_retries_allowed: bool
    dns_cache_allowed: bool
    connection_reuse_allowed: bool
    cookies_allowed: bool
    credentials_allowed: bool
    client_certificates_allowed: bool
    custom_ca_bundle_allowed: bool
    tls_certificate_validation_required: bool
    tls_hostname_validation_required: bool
    environment_tls_overrides_allowed: bool
    tls_key_logging_allowed: bool
    minimum_tls_version: str
    transfer_encoding_allowed: bool
    max_dns_addresses: int
    max_header_count: int
    max_status_line_bytes: int
    socket_read_chunk_bytes: int
    max_connect_attempts: int

    def validate(
        self,
        *,
        information_policy: InformationPolicy,
        retrieval_policy: InformationHttpRetrievalPolicy,
    ) -> None:
        """Revalidate this activation policy against the frozen lower layers."""

        information_policy.validate()
        retrieval_policy.validate(information_policy=information_policy)
        if self.policy_name != "alice_information_live_http_policy":
            raise InformationLiveHttpPolicyError(
                "P4.2b policy_name must be alice_information_live_http_policy."
            )
        if self.version != "1.0.0":
            raise InformationLiveHttpPolicyError(
                "P4.2b live HTTP policy version must be 1.0.0."
            )
        if self.phase != "4" or self.milestone != "P4.2b":
            raise InformationLiveHttpPolicyError(
                "Live HTTP policy must be bound to Phase 4 milestone P4.2b."
            )
        if self.status != "controlled_live_transport":
            raise InformationLiveHttpPolicyError(
                "P4.2b live HTTP policy status must be controlled_live_transport."
            )
        if self.permission_id != "web.search":
            raise InformationLiveHttpPolicyError(
                "P4.2b live HTTP access must map to web.search."
            )
        if self.live_network_access_allowed is not True:
            raise InformationLiveHttpPolicyError(
                "P4.2b live_network_access_allowed must be true."
            )
        if self.approved_resolver_type != "system_getaddrinfo":
            raise InformationLiveHttpPolicyError(
                "P4.2b approved resolver type must be system_getaddrinfo."
            )
        if self.approved_transport_type != "direct_https_socket":
            raise InformationLiveHttpPolicyError(
                "P4.2b approved transport type must be direct_https_socket."
            )
        if self.allowed_schemes != ("https",):
            raise InformationLiveHttpPolicyError(
                "P4.2b live transport must remain HTTPS-only."
            )
        if "https" not in information_policy.allowed_schemes:
            raise InformationLiveHttpPolicyError(
                "P4.0 policy does not permit HTTPS source identities."
            )
        if "https" not in retrieval_policy.allowed_schemes:
            raise InformationLiveHttpPolicyError(
                "P4.2a retrieval policy does not permit HTTPS."
            )
        if retrieval_policy.allowed_ports_for("https") != (443,):
            raise InformationLiveHttpPolicyError(
                "P4.2b requires the exact default HTTPS port allowlist."
            )
        for field_name in (
            "operating_system_dns_allowed",
            "direct_socket_connections_allowed",
            "tls_certificate_validation_required",
            "tls_hostname_validation_required",
        ):
            if getattr(self, field_name) is not True:
                raise InformationLiveHttpPolicyError(
                    f"P4.2b {field_name} must remain true."
                )
        for field_name in (
            "environment_proxies_allowed",
            "automatic_retries_allowed",
            "dns_cache_allowed",
            "connection_reuse_allowed",
            "cookies_allowed",
            "credentials_allowed",
            "client_certificates_allowed",
            "custom_ca_bundle_allowed",
            "environment_tls_overrides_allowed",
            "tls_key_logging_allowed",
            "transfer_encoding_allowed",
        ):
            if getattr(self, field_name) is not False:
                raise InformationLiveHttpPolicyError(
                    f"P4.2b {field_name} must remain false."
                )
        if self.minimum_tls_version != "TLSv1.2":
            raise InformationLiveHttpPolicyError(
                "P4.2b minimum TLS version must be TLSv1.2."
            )
        _bounded_int(
            self.max_dns_addresses,
            field="max_dns_addresses",
            minimum=1,
            maximum=16,
        )
        _bounded_int(
            self.max_header_count,
            field="max_header_count",
            minimum=1,
            maximum=100,
        )
        _bounded_int(
            self.max_status_line_bytes,
            field="max_status_line_bytes",
            minimum=128,
            maximum=8192,
        )
        _bounded_int(
            self.socket_read_chunk_bytes,
            field="socket_read_chunk_bytes",
            minimum=1024,
            maximum=65536,
        )
        if self.max_connect_attempts != 1:
            raise InformationLiveHttpPolicyError(
                "P4.2b permits exactly one pinned connection attempt."
            )
        if information_policy.capabilities.live_network_access_allowed is not False:
            raise InformationLiveHttpPolicyError(
                "P4.0 capability declarations must remain frozen and disabled."
            )
        if retrieval_policy.live_network_access_allowed is not False:
            raise InformationLiveHttpPolicyError(
                "P4.2a retrieval policy must remain network-free."
            )
        if retrieval_policy.environment_proxy_allowed is not False:
            raise InformationLiveHttpPolicyError(
                "P4.2a environment proxy boundary was weakened."
            )
        if retrieval_policy.automatic_retries_allowed is not False:
            raise InformationLiveHttpPolicyError(
                "P4.2a automatic retry boundary was weakened."
            )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationLiveHttpPolicyError(f"{field} must be an object.")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    *,
    field: str,
    expected: tuple[str, ...],
) -> None:
    actual = set(value)
    approved = set(expected)
    if actual == approved:
        return
    missing = sorted(approved - actual)
    unexpected = sorted(actual - approved)
    details: list[str] = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if unexpected:
        details.append("unexpected=" + ",".join(unexpected))
    raise InformationLiveHttpPolicyError(
        f"{field} must contain exactly the approved P4.2b keys ({'; '.join(details)})."
    )


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationLiveHttpPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationLiveHttpPolicyError(
            f"{field} must remain {str(expected).lower()} in P4.2b."
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
        raise InformationLiveHttpPolicyError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise InformationLiveHttpPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def parse_information_live_http_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy,
    retrieval_policy: InformationHttpRetrievalPolicy,
) -> InformationLiveHttpPolicy:
    """Validate and project one decoded P4.2b live transport policy."""

    _require_exact_keys(
        payload,
        field="policy",
        expected=(
            "policy_name",
            "version",
            "phase",
            "milestone",
            "status",
            "permission_id",
            "live_network_access_allowed",
            "approved_resolver_type",
            "approved_transport_type",
            "allowed_schemes",
            "network",
            "privacy",
            "tls",
            "limits",
        ),
    )
    network = _mapping(payload.get("network"), field="network")
    privacy = _mapping(payload.get("privacy"), field="privacy")
    tls = _mapping(payload.get("tls"), field="tls")
    limits = _mapping(payload.get("limits"), field="limits")
    _require_exact_keys(
        network,
        field="network",
        expected=(
            "operating_system_dns_allowed",
            "direct_socket_connections_allowed",
            "environment_proxies_allowed",
            "automatic_retries_allowed",
            "dns_cache_allowed",
            "connection_reuse_allowed",
            "transfer_encoding_allowed",
        ),
    )
    _require_exact_keys(
        privacy,
        field="privacy",
        expected=(
            "cookies_allowed",
            "credentials_allowed",
            "client_certificates_allowed",
            "custom_ca_bundle_allowed",
        ),
    )
    _require_exact_keys(
        tls,
        field="tls",
        expected=(
            "certificate_validation_required",
            "hostname_validation_required",
            "environment_overrides_allowed",
            "key_logging_allowed",
            "minimum_version",
        ),
    )
    _require_exact_keys(
        limits,
        field="limits",
        expected=(
            "max_dns_addresses",
            "max_header_count",
            "max_status_line_bytes",
            "socket_read_chunk_bytes",
            "max_connect_attempts",
        ),
    )

    allowed_schemes = payload.get("allowed_schemes")
    if allowed_schemes != ["https"]:
        raise InformationLiveHttpPolicyError(
            "allowed_schemes must equal the approved P4.2b value: ['https']."
        )

    policy = InformationLiveHttpPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        permission_id=_text(payload.get("permission_id"), field="permission_id"),
        live_network_access_allowed=_strict_bool(
            payload.get("live_network_access_allowed"),
            field="live_network_access_allowed",
            expected=True,
        ),
        approved_resolver_type=_text(
            payload.get("approved_resolver_type"),
            field="approved_resolver_type",
        ),
        approved_transport_type=_text(
            payload.get("approved_transport_type"),
            field="approved_transport_type",
        ),
        allowed_schemes=("https",),
        operating_system_dns_allowed=_strict_bool(
            network.get("operating_system_dns_allowed"),
            field="network.operating_system_dns_allowed",
            expected=True,
        ),
        direct_socket_connections_allowed=_strict_bool(
            network.get("direct_socket_connections_allowed"),
            field="network.direct_socket_connections_allowed",
            expected=True,
        ),
        environment_proxies_allowed=_strict_bool(
            network.get("environment_proxies_allowed"),
            field="network.environment_proxies_allowed",
            expected=False,
        ),
        automatic_retries_allowed=_strict_bool(
            network.get("automatic_retries_allowed"),
            field="network.automatic_retries_allowed",
            expected=False,
        ),
        dns_cache_allowed=_strict_bool(
            network.get("dns_cache_allowed"),
            field="network.dns_cache_allowed",
            expected=False,
        ),
        connection_reuse_allowed=_strict_bool(
            network.get("connection_reuse_allowed"),
            field="network.connection_reuse_allowed",
            expected=False,
        ),
        cookies_allowed=_strict_bool(
            privacy.get("cookies_allowed"),
            field="privacy.cookies_allowed",
            expected=False,
        ),
        credentials_allowed=_strict_bool(
            privacy.get("credentials_allowed"),
            field="privacy.credentials_allowed",
            expected=False,
        ),
        client_certificates_allowed=_strict_bool(
            privacy.get("client_certificates_allowed"),
            field="privacy.client_certificates_allowed",
            expected=False,
        ),
        custom_ca_bundle_allowed=_strict_bool(
            privacy.get("custom_ca_bundle_allowed"),
            field="privacy.custom_ca_bundle_allowed",
            expected=False,
        ),
        tls_certificate_validation_required=_strict_bool(
            tls.get("certificate_validation_required"),
            field="tls.certificate_validation_required",
            expected=True,
        ),
        tls_hostname_validation_required=_strict_bool(
            tls.get("hostname_validation_required"),
            field="tls.hostname_validation_required",
            expected=True,
        ),
        environment_tls_overrides_allowed=_strict_bool(
            tls.get("environment_overrides_allowed"),
            field="tls.environment_overrides_allowed",
            expected=False,
        ),
        tls_key_logging_allowed=_strict_bool(
            tls.get("key_logging_allowed"),
            field="tls.key_logging_allowed",
            expected=False,
        ),
        minimum_tls_version=_text(
            tls.get("minimum_version"),
            field="tls.minimum_version",
        ),
        transfer_encoding_allowed=_strict_bool(
            network.get("transfer_encoding_allowed"),
            field="network.transfer_encoding_allowed",
            expected=False,
        ),
        max_dns_addresses=_bounded_int(
            limits.get("max_dns_addresses"),
            field="limits.max_dns_addresses",
            minimum=1,
            maximum=16,
        ),
        max_header_count=_bounded_int(
            limits.get("max_header_count"),
            field="limits.max_header_count",
            minimum=1,
            maximum=100,
        ),
        max_status_line_bytes=_bounded_int(
            limits.get("max_status_line_bytes"),
            field="limits.max_status_line_bytes",
            minimum=128,
            maximum=8192,
        ),
        socket_read_chunk_bytes=_bounded_int(
            limits.get("socket_read_chunk_bytes"),
            field="limits.socket_read_chunk_bytes",
            minimum=1024,
            maximum=65536,
        ),
        max_connect_attempts=_bounded_int(
            limits.get("max_connect_attempts"),
            field="limits.max_connect_attempts",
            minimum=1,
            maximum=1,
        ),
    )
    policy.validate(
        information_policy=information_policy,
        retrieval_policy=retrieval_policy,
    )
    return policy


def load_information_live_http_policy(
    *,
    information_policy: InformationPolicy,
    retrieval_policy: InformationHttpRetrievalPolicy,
    path: str | Path = DEFAULT_LIVE_HTTP_POLICY_PATH,
) -> InformationLiveHttpPolicy:
    """Load and validate the repository P4.2b live transport policy."""

    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationLiveHttpPolicyError(
            "P4.2b live HTTP policy could not be loaded safely."
        ) from exc
    if not isinstance(payload, dict):
        raise InformationLiveHttpPolicyError(
            "P4.2b live HTTP policy root must be an object."
        )
    return parse_information_live_http_policy(
        payload,
        information_policy=information_policy,
        retrieval_policy=retrieval_policy,
    )
