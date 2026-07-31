import json
from pathlib import Path

import pytest

from alice_information.live_provider_config import (
    InformationLiveProviderConfigurationError,
    load_live_provider_configuration,
)
from alice_information.live_provider_policy import InformationLiveProviderRuntimePolicy

ROOT = Path(__file__).resolve().parents[2]
POLICY = InformationLiveProviderRuntimePolicy.load(
    ROOT / "policies/information_live_provider_runtime_policy.json"
)


def write_config(path: Path, **changes):
    value = {
        "provider": "brave-search-v1",
        "country": "US",
        "search_lang": "en",
        "ui_lang": "en-US",
        "safesearch": "off",
    }
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_external_config_loads_without_serializing_token(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    config = tmp_path / "private.json"
    write_config(config)
    result = load_live_provider_configuration(
        config,
        repository_root=repository,
        policy=POLICY,
        environment={"ALICE_BRAVE_SEARCH_API_KEY": "super-private-token"},
    )
    assert result.provider == "brave-search-v1"
    assert "super-private-token" not in repr(result)
    assert "super-private-token" not in json.dumps(result.to_metadata_record())
    assert result.credential.reveal_for_exact_header() == "super-private-token"


def test_config_inside_repository_is_rejected(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    config = repository / "private.json"
    write_config(config)
    with pytest.raises(InformationLiveProviderConfigurationError):
        load_live_provider_configuration(
            config,
            repository_root=repository,
            policy=POLICY,
            environment={"ALICE_BRAVE_SEARCH_API_KEY": "token"},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "other"},
        {"country": "USA"},
        {"safesearch": "unknown"},
        {"api_token": "forbidden"},
    ],
)
def test_invalid_configuration_fails_closed(tmp_path, changes):
    repository = tmp_path / "repo"
    repository.mkdir()
    config = tmp_path / "private.json"
    write_config(config, **changes)
    with pytest.raises(InformationLiveProviderConfigurationError):
        load_live_provider_configuration(
            config,
            repository_root=repository,
            policy=POLICY,
            environment={"ALICE_BRAVE_SEARCH_API_KEY": "token"},
        )


def test_missing_environment_credential_is_rejected(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    config = tmp_path / "private.json"
    write_config(config)
    with pytest.raises(InformationLiveProviderConfigurationError):
        load_live_provider_configuration(
            config,
            repository_root=repository,
            policy=POLICY,
            environment={},
        )
