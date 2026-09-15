from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/eipm/n0/build_n0_v02_evidence_graph_curriculum.py"
SPEC = importlib.util.spec_from_file_location("n0_evidence_graph_curriculum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def all_rows():
    rows = []
    for _name, builder in MODULE.FAMILIES:
        rows.extend(builder(index) for index in range(MODULE.EXAMPLES_PER_FAMILY))
    return rows


def test_curriculum_has_train_and_heldout_dev_for_every_family() -> None:
    rows = all_rows()
    assert len(rows) == 240
    families = {name for name, _builder in MODULE.FAMILIES}
    assert len(families) == 10
    for family in families:
        train = [row for row in rows if row["family"] == family and row["split"] == "train"]
        dev = [row for row in rows if row["family"] == family and row["split"] == "dev"]
        assert len(train) == 18
        assert len(dev) == 6


def test_every_row_is_public_and_has_normalized_soft_evidence_target() -> None:
    for row in all_rows():
        MODULE.validate_row(row)
        assert row["private_identity_content"] is False
        assert row["synthetic_public_template"] is True
        distribution = row["target_evidence_distribution"]
        assert abs(sum(distribution) - 1.0) < 1e-9
        assert all(value >= 0.0 for value in distribution)


def test_curriculum_covers_history_uncertainty_and_multihop_reasoning() -> None:
    rows = all_rows()
    families = {row["family"] for row in rows}
    assert "supersession_historical" in families
    assert "unresolved_conflict" in families
    assert "causal_explanation" in families
    assert "mixed_update_support" in families

    unresolved = next(row for row in rows if row["family"] == "unresolved_conflict")
    nonzero = [value for value in unresolved["target_evidence_distribution"] if value > 0.0]
    assert len(nonzero) == 2
    assert nonzero == [0.5, 0.5]


def test_historical_query_can_select_superseded_record() -> None:
    row = next(row for row in all_rows() if row["family"] == "supersession_historical")
    assert row["relations"][0]["relation"] == "supersedes"
    assert row["relations"][0]["source"] == 1
    assert row["relations"][0]["target"] == 0
    assert row["target_evidence_distribution"] == [1.0, 0.0, 0.0]


def test_counterfactual_supervision_is_present() -> None:
    for row in all_rows():
        assert row["counterfactual_required"] is True
        assert row["counterfactual_policy"] == "drop_first_decisive_relation"
