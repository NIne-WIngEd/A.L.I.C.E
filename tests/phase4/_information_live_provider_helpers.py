from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from alice_information.brave_search_live import BraveSearchRawResponse
from alice_information.contracts import InformationQuery
from alice_information.live_provider_config import (
    InformationLiveProviderConfiguration,
    InformationSecretValue,
)
from alice_information.live_provider_policy import InformationLiveProviderRuntimePolicy

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "policies/information_live_provider_runtime_policy.json"


def policy() -> InformationLiveProviderRuntimePolicy:
    return InformationLiveProviderRuntimePolicy.load(POLICY_PATH)


def configuration(token: str = "test-secret-not-for-network") -> InformationLiveProviderConfiguration:
    value = {
        "provider": "brave-search-v1",
        "country": "US",
        "search_lang": "en",
        "ui_lang": "en-US",
        "safesearch": "off",
    }
    from alice_information.live_provider_policy import canonical_sha256

    return InformationLiveProviderConfiguration(
        provider=value["provider"],
        country=value["country"],
        search_lang=value["search_lang"],
        ui_lang=value["ui_lang"],
        safesearch=value["safesearch"],
        configuration_path="C:/ALICE_Vault/config/phase4-live-provider.json",
        configuration_sha256=canonical_sha256(value),
        credential=InformationSecretValue(token),
        metadata=MappingProxyType(value),
    )


def query(text: str = "OpenAI official documentation") -> InformationQuery:
    return InformationQuery.create(
        query_id="p410-test-query",
        text=text,
        created_at="2026-07-30T12:00:00Z",
    )


class FixtureBraveTransport:
    transport_type = "deterministic_fixture"

    def __init__(self, *, status: int = 200, payload: dict | None = None):
        self.status = status
        self.payload = payload or {
            "query": {"original": "OpenAI official documentation"},
            "web": {
                "results": [
                    {
                        "title": "OpenAI Docs",
                        "url": "https://platform.openai.com/docs/",
                        "description": "Official public documentation for OpenAI APIs.",
                    }
                ]
            },
        }
        self.calls = 0
        self.last_credential = None

    def perform(self, **kwargs):
        self.calls += 1
        self.last_credential = kwargs["credential"].reveal_for_exact_header()
        body = json.dumps(self.payload).encode("utf-8")
        return BraveSearchRawResponse(
            status_code=self.status,
            headers=(
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("X-RateLimit-Limit", "1"),
                ("X-RateLimit-Remaining", "1"),
                ("X-RateLimit-Reset", "1"),
                ("X-RateLimit-Policy", "1;w=1"),
            ),
            body=body,
            peer_address="1.1.1.1",
        )


class FixtureResolver:
    def __init__(self, address: str = "1.1.1.1"):
        self.address = address
        self.calls = 0

    def resolve(self, canonical_url, *, policy, cancellation=None):
        self.calls += 1
        return SimpleNamespace(addresses=(self.address,), canonical_url=canonical_url)


class FixtureSocket:
    def __init__(self, response: bytes, *, peer: str = "1.1.1.1"):
        self.response = bytearray(response)
        self.peer = peer
        self.sent = b""
        self.closed = False

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, value):
        self.sent += value

    def recv(self, amount):
        if not self.response:
            return b""
        value = bytes(self.response[:amount])
        del self.response[:amount]
        return value

    def getpeername(self):
        return (self.peer, 443)

    def close(self):
        self.closed = True


class FixtureSocketBackend:
    def __init__(self, response: bytes, *, peer: str = "1.1.1.1"):
        self.socket = FixtureSocket(response, peer=peer)

    def open(self, *, target, address, timeout_seconds):
        return self.socket
