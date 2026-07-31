import runpy
from pathlib import Path


def test_preflight_script_is_metadata_only():
    text = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_phase4_live_provider_preflight.py"
    ).read_text(encoding="utf-8")
    assert "response.receipt.to_metadata_record()" in text
    assert "response.body" not in text
    assert "source_body" not in text
    assert "--output" in text
    assert "load_information_policy" in text
    assert "load_information_http_retrieval_policy" in text
    assert "live_policy_loader" in text
    assert "load_information_live_http_policy" in text
    assert "InformationPolicy.load" not in text


def test_preflight_supports_keyword_only_live_policy_loader():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_phase4_live_provider_preflight.py"
    )
    namespace = runpy.run_path(str(script))
    helper = namespace["_load_live_http_policy"]
    observed = {}

    def loader(*, path, information_policy, retrieval_policy):
        observed.update(
            path=path,
            information_policy=information_policy,
            retrieval_policy=retrieval_policy,
        )
        return "loaded"

    policy_path = Path("policies/information_live_http_policy.json")
    information_policy = object()
    retrieval_policy = object()

    result = helper(
        loader,
        policy_path=policy_path,
        information_policy=information_policy,
        retrieval_policy=retrieval_policy,
    )

    assert result == "loaded"
    assert observed == {
        "path": policy_path,
        "information_policy": information_policy,
        "retrieval_policy": retrieval_policy,
    }
