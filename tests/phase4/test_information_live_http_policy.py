"""P4.2b live HTTPS activation policy tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alice_information.live_policy import (
    InformationLiveHttpPolicyError,
    load_information_live_http_policy,
    parse_information_live_http_policy,
)
from alice_information.policy import load_information_policy
from alice_information.retrieval_policy import load_information_http_retrieval_policy

ROOT = Path(__file__).resolve().parents[2]
INFO_PATH = ROOT / "policies" / "information_policy.json"
RETRIEVAL_PATH = ROOT / "policies" / "information_http_retrieval_policy.json"
LIVE_PATH = ROOT / "policies" / "information_live_http_policy.json"


def _policies():
    information = load_information_policy(INFO_PATH)
    retrieval = load_information_http_retrieval_policy(
        RETRIEVAL_PATH,
        information_policy=information,
    )
    live = load_information_live_http_policy(
        information_policy=information,
        retrieval_policy=retrieval,
        path=LIVE_PATH,
    )
    return information, retrieval, live


def _payload() -> dict:
    return json.loads(LIVE_PATH.read_text(encoding="utf-8"))


def test_repository_live_policy_is_exact_and_https_only() -> None:
    information, retrieval, live = _policies()
    live.validate(information_policy=information, retrieval_policy=retrieval)
    assert live.milestone == "P4.2b"
    assert live.allowed_schemes == ("https",)
    assert live.live_network_access_allowed is True
    assert live.environment_proxies_allowed is False
    assert live.max_connect_attempts == 1


def test_live_policy_does_not_mutate_frozen_foundation_policies() -> None:
    information, retrieval, live = _policies()
    assert information.capabilities.live_network_access_allowed is False
    assert retrieval.live_network_access_allowed is False
    live.validate(information_policy=information, retrieval_policy=retrieval)


def test_live_policy_identity_and_version_are_exactly_bound() -> None:
    information, retrieval, _live = _policies()
    payload = _payload()
    payload["policy_name"] = "lookalike_live_policy"
    with pytest.raises(InformationLiveHttpPolicyError, match="policy_name"):
        parse_information_live_http_policy(
            payload,
            information_policy=information,
            retrieval_policy=retrieval,
        )
    payload = _payload()
    payload["version"] = "9.9.9"
    with pytest.raises(InformationLiveHttpPolicyError, match="version"):
        parse_information_live_http_policy(
            payload,
            information_policy=information,
            retrieval_policy=retrieval,
        )


def test_live_policy_rejects_http_proxy_retry_and_connection_reuse() -> None:
    information, retrieval, _live = _policies()
    payload = _payload()
    payload["allowed_schemes"] = ["http", "https"]
    with pytest.raises(InformationLiveHttpPolicyError, match="allowed_schemes"):
        parse_information_live_http_policy(
            payload,
            information_policy=information,
            retrieval_policy=retrieval,
        )
    for field in (
        "environment_proxies_allowed",
        "automatic_retries_allowed",
        "connection_reuse_allowed",
    ):
        payload = _payload()
        payload["network"][field] = True
        with pytest.raises(InformationLiveHttpPolicyError, match=field):
            parse_information_live_http_policy(
                payload,
                information_policy=information,
                retrieval_policy=retrieval,
            )


def test_live_policy_rejects_credentials_custom_ca_and_transfer_encoding() -> None:
    information, retrieval, _live = _policies()
    for section, field in (
        ("privacy", "credentials_allowed"),
        ("privacy", "client_certificates_allowed"),
        ("privacy", "custom_ca_bundle_allowed"),
        ("tls", "environment_overrides_allowed"),
        ("tls", "key_logging_allowed"),
        ("network", "transfer_encoding_allowed"),
    ):
        payload = _payload()
        payload[section][field] = True
        with pytest.raises(InformationLiveHttpPolicyError, match=field):
            parse_information_live_http_policy(
                payload,
                information_policy=information,
                retrieval_policy=retrieval,
            )


def test_live_policy_rejects_weakened_tls_and_multiple_connect_attempts() -> None:
    information, retrieval, live = _policies()
    with pytest.raises(InformationLiveHttpPolicyError, match="minimum TLS"):
        replace(live, minimum_tls_version="TLSv1.0").validate(
            information_policy=information,
            retrieval_policy=retrieval,
        )
    with pytest.raises(InformationLiveHttpPolicyError, match="exactly one"):
        replace(live, max_connect_attempts=2).validate(
            information_policy=information,
            retrieval_policy=retrieval,
        )


def test_live_policy_rejects_unknown_root_and_nested_keys() -> None:
    information, retrieval, _live = _policies()
    payload = _payload()
    payload["future_capability"] = True
    with pytest.raises(InformationLiveHttpPolicyError, match="unexpected"):
        parse_information_live_http_policy(
            payload,
            information_policy=information,
            retrieval_policy=retrieval,
        )

    payload = _payload()
    payload["network"]["proxy_url"] = "https://proxy.invalid"
    with pytest.raises(InformationLiveHttpPolicyError, match="unexpected"):
        parse_information_live_http_policy(
            payload,
            information_policy=information,
            retrieval_policy=retrieval,
        )


def test_live_policy_loader_fails_closed_for_invalid_json(tmp_path: Path) -> None:
    information, retrieval, _live = _policies()
    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(InformationLiveHttpPolicyError, match="could not be loaded"):
        load_information_live_http_policy(
            information_policy=information,
            retrieval_policy=retrieval,
            path=path,
        )
