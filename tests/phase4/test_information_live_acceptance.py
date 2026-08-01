from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_information.live_acceptance import (
    InformationLiveAcceptanceError,
    InformationLiveAcceptancePolicy,
    load_live_acceptance_record,
    write_live_acceptance_record,
)
from _information_live_acceptance_helpers import POLICY_PATH, policy, record


def test_exact_acceptance_policy_loads():
    value = policy()
    assert value.package_version == "0.18.0"
    assert value.required_search_provider == "brave-search-v1"
    assert value.required_fetch_provider == "controlled-live-http-v1"
    assert len(value.required_acceptance_domains) == 11
    assert value.raw["required_regression"]["p410_targeted_passed"] == 101


def test_policy_mutation_and_duplicate_key_are_rejected(tmp_path):
    data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    data["execution_controls"]["no_retry"] = False
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InformationLiveAcceptanceError):
        InformationLiveAcceptancePolicy.load(path)
    duplicate = POLICY_PATH.read_text(encoding="utf-8").replace(
        '"phase": "4",', '"phase": "4",\n  "phase": "4",'
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(InformationLiveAcceptanceError, match="Duplicate"):
        InformationLiveAcceptancePolicy.load(path)


def test_record_is_approved_tamper_evident_and_reloadable(tmp_path):
    value = record()
    value.validate()
    assert value.deterministic_test_passed == 101
    assert value.repository_regression_passed == 2124
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    output = vault / "reports" / "phase4-live-information-acceptance.json"
    written = write_live_acceptance_record(
        value,
        output,
        repository_root=repository,
        private_root=vault,
    )
    assert load_live_acceptance_record(written) == value
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["deterministic_test_passed"] = 16
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InformationLiveAcceptanceError):
        load_live_acceptance_record(output)


def test_record_output_must_be_under_private_vault_and_outside_repo(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    elsewhere = tmp_path / "elsewhere"
    repository.mkdir(); vault.mkdir(); elsewhere.mkdir()
    with pytest.raises(InformationLiveAcceptanceError, match="vault"):
        write_live_acceptance_record(
            record(),
            elsewhere / "record.json",
            repository_root=repository,
            private_root=vault,
        )
    with pytest.raises(InformationLiveAcceptanceError, match="outside Git"):
        write_live_acceptance_record(
            record(),
            repository / "record.json",
            repository_root=repository,
            private_root=vault,
        )


def test_record_uses_real_git_sha1_width_not_sha256_width():
    value = record()
    assert len(value.repository_commit) == 40
    with pytest.raises(InformationLiveAcceptanceError, match="40-character"):
        replace(value, repository_commit="a" * 64).validate()
