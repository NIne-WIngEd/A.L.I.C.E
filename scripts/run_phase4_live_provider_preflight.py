#!/usr/bin/env python3
"""Run one private metadata-only Brave provider preflight outside Git."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


def _outside(path: Path, repository: Path) -> bool:
    try:
        path.resolve().relative_to(repository.resolve())
    except ValueError:
        return True
    return False


def _load_live_http_policy(
    loader,
    *,
    policy_path: Path,
    information_policy,
    retrieval_policy,
):
    """Call the repository loader without assuming positional parameters."""
    signature = inspect.signature(loader)
    kwargs = {}

    for candidate in ("path", "policy_path", "live_policy_path"):
        if candidate in signature.parameters:
            kwargs[candidate] = policy_path
            break

    if "information_policy" in signature.parameters:
        kwargs["information_policy"] = information_policy
    if "retrieval_policy" in signature.parameters:
        kwargs["retrieval_policy"] = retrieval_policy

    return loader(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--policy", default="policies/information_live_provider_runtime_policy.json")
    parser.add_argument("--configuration", default=r"C:\ALICE_Vault\config\phase4-live-provider.json")
    parser.add_argument("--query", default="OpenAI official homepage")
    parser.add_argument("--output")
    args = parser.parse_args()

    repository = Path(args.repository_root).resolve(strict=True)
    sys.path.insert(0, str(repository / "src"))
    from alice_information.brave_search import BraveInformationSearchProvider
    from alice_information.brave_search_live import StrictBraveSearchHttpsTransport
    from alice_information.contracts import InformationQuery, utc_now_text
    from alice_information.live_http import (
        StdlibInformationSocketBackend,
        SystemInformationNameResolver,
    )
    from alice_information import live_policy as live_policy_module
    from alice_information.live_provider_config import load_live_provider_configuration
    from alice_information.live_provider_contracts import InformationLiveProviderExecutionError
    from alice_information.live_provider_policy import InformationLiveProviderRuntimePolicy
    from alice_information.policy import load_information_policy
    from alice_information.retrieval_policy import load_information_http_retrieval_policy

    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = repository / policy_path
    runtime_policy = InformationLiveProviderRuntimePolicy.load(policy_path)
    configuration = load_live_provider_configuration(
        args.configuration,
        repository_root=repository,
        policy=runtime_policy,
    )
    information_policy = load_information_policy(
        repository / "policies/information_policy.json"
    )
    retrieval_policy = load_information_http_retrieval_policy(
        repository / "policies/information_http_retrieval_policy.json",
        information_policy=information_policy,
    )
    live_policy_path = repository / "policies/information_live_http_policy.json"
    live_policy_loader = None
    for loader_name in (
        "load_information_live_http_policy",
        "load_live_information_http_policy",
        "load_live_http_policy",
    ):
        candidate = getattr(live_policy_module, loader_name, None)
        if callable(candidate):
            live_policy_loader = candidate
            break
    if live_policy_loader is None:
        raise RuntimeError("The repository live HTTP policy loader is unavailable.")
    live_policy = _load_live_http_policy(
        live_policy_loader,
        policy_path=live_policy_path,
        information_policy=information_policy,
        retrieval_policy=retrieval_policy,
    )
    resolver = SystemInformationNameResolver(
        information_policy=information_policy,
        retrieval_policy=retrieval_policy,
        live_policy=live_policy,
    )
    transport = StrictBraveSearchHttpsTransport(
        resolver=resolver,
        retrieval_policy=retrieval_policy,
        socket_backend=StdlibInformationSocketBackend(),
        policy=runtime_policy,
    )
    transport.validate_live_boundary()
    provider = BraveInformationSearchProvider(
        policy=runtime_policy,
        configuration=configuration,
        transport=transport,
    )
    query = InformationQuery.create(
        query_id="p410a-private-preflight",
        text=args.query,
        created_at=utc_now_text(),
    )
    try:
        response = provider.search_with_receipt(
            query,
            max_results=1,
            timeout_seconds=10.0,
            operation="preflight",
        )
    except InformationLiveProviderExecutionError as exc:
        print(exc.failure.code)
        return 2
    record = {
        "preflight_version": "p4.10a-v1",
        "approved": bool(response.results),
        "provider": runtime_policy.search_provider_id,
        "configuration_sha256": configuration.configuration_sha256,
        "egress": response.receipt.to_metadata_record(),
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        if not _outside(output, repository):
            raise SystemExit("Preflight output must remain outside the repository.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(output)
    else:
        print(rendered)
    return 0 if record["approved"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
