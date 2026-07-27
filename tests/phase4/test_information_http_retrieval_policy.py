from __future__ import annotations

import json
from dataclasses import replace

import pytest
from alice_information.policy import load_information_policy
from alice_information.retrieval_policy import (
    InformationHttpRetrievalPolicyError,
    load_information_http_retrieval_policy,
    parse_information_http_retrieval_policy,
)


def _payload() -> dict[str, object]:
    with open("policies/information_http_retrieval_policy.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_http_retrieval_policy_loads_and_is_bound_to_foundation() -> None:
    foundation = load_information_policy()
    policy = load_information_http_retrieval_policy(
        information_policy=foundation,
    )
    assert policy.milestone == "P4.2"
    assert policy.live_network_access_allowed is False
    assert policy.allowed_ports_for("http") == (80,)
    assert policy.allowed_ports_for("https") == (443,)
    assert policy.max_decoded_bytes <= foundation.max_response_bytes


def test_http_retrieval_policy_rejects_live_network() -> None:
    payload = _payload()
    payload["live_network_access_allowed"] = True
    with pytest.raises(InformationHttpRetrievalPolicyError, match="false"):
        parse_information_http_retrieval_policy(payload)


def test_http_retrieval_policy_rejects_custom_ports() -> None:
    payload = _payload()
    payload["allowed_ports"] = {"http": [80, 8080], "https": [443]}
    with pytest.raises(InformationHttpRetrievalPolicyError, match="default"):
        parse_information_http_retrieval_policy(payload)


def test_http_retrieval_policy_rejects_foundation_budget_expansion() -> None:
    policy = replace(
        load_information_http_retrieval_policy(),
        max_wire_bytes=100,
        max_decoded_bytes=200,
    )
    foundation = replace(load_information_policy(), max_response_bytes=100)
    with pytest.raises(InformationHttpRetrievalPolicyError, match="decoded-byte"):
        policy.validate(information_policy=foundation)


def test_http_retrieval_policy_rejects_extra_content_type() -> None:
    payload = _payload()
    payload["allowed_content_types"] = [
        "application/pdf",
        "application/xhtml+xml",
        "text/html",
        "text/plain",
    ]
    with pytest.raises(InformationHttpRetrievalPolicyError, match="content types"):
        parse_information_http_retrieval_policy(payload)


def test_http_retrieval_policy_rejects_wire_budget_expansion() -> None:
    policy = replace(load_information_http_retrieval_policy(), max_wire_bytes=200)
    foundation = replace(load_information_policy(), max_response_bytes=100)
    with pytest.raises(InformationHttpRetrievalPolicyError, match="wire-byte"):
        policy.validate(information_policy=foundation)
