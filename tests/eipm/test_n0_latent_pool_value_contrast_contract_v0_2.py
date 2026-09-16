from __future__ import annotations

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2 as builder
import compile_n0_v02_latent_pool_value_contrast_contract_v0_2 as contract


def _rows() -> list[dict]:
    return [
        builder.build_row(family, index)
        for family in builder.prior.base.FAMILIES
        for index in range(builder.prior.base.ROWS_PER_FAMILY)
    ]


def test_contract_only_covers_counterfactual_required_rows() -> None:
    rows = _rows()
    cases = contract.compile_cases(rows)
    expected = {row["id"] for row in rows if row.get("counterfactual_required")}
    assert len(cases) == 32
    assert {case["row_id"] for case in cases} == expected
    assert {case["family"] for case in cases} == set(contract.EXPECTED_COUNTERFACTUAL_FAMILIES)


def test_historical_and_uncertainty_families_are_not_forced_into_current_value_contract() -> None:
    cases = contract.compile_cases(_rows())
    families = {case["family"] for case in cases}
    assert "supersession_historical" not in families
    assert "correction_chain_historical" not in families
    assert "temporal_successor_previous" not in families
    assert "paired_historical_state" not in families
    assert "unresolved_balanced_conflict" not in families


def test_every_contrast_is_materialized_and_query_conditioned() -> None:
    cases = contract.compile_cases(_rows())
    for case in cases:
        assert case["query_answer_role"] == "authoritative_current_value"
        assert case["target_text"] != case["foil_text"]
        assert case["target_value"] in case["target_text"]
        assert case["foil_value"] in case["foil_text"]
        assert case["target_value"] != case["foil_value"]
        assert case["required_source_contains_target_value"] is True
