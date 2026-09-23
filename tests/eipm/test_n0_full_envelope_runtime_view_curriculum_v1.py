from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_runtime_view_train_dev_curriculum_is_executable_and_not_internal_view_only() -> None:
    contract_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_runtime_view_curriculum_contract_v1.json"
    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_runtime_view_curriculum_v1.py"
    audit_path=ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_runtime_view_curriculum_v1.py"
    assert contract_path.is_file(), "additional runtime-view TRAIN/DEV contract missing"
    assert builder_path.is_file(), "additional runtime-view TRAIN/DEV builder missing"
    assert audit_path.is_file(), "additional runtime-view TRAIN/DEV audit missing"

    contract=json.loads(contract_path.read_text())
    assert contract["schema"]=="alice.eipm.n0.full-envelope-runtime-view-curriculum-contract.v1"
    assert contract["authority"]["private_identity_data"] is False
    assert contract["authority"]["final_rows_training_allowed"] is False
    assert contract["runtime_views"]["fixed_view_taxonomy"] is False
    assert contract["runtime_views"]["view_count_ceiling"] is None
    assert contract["runtime_views"]["source_text_training_adapter_required"] is True
    assert contract["required_interventions"]["relevant_vs_merely_available"] is True
    assert contract["required_interventions"]["constant_reliability_relevance_flip"] is True
    assert contract["required_interventions"]["unavailable_view_exact_inertness"] is True
    assert contract["required_interventions"]["recoverability_not_equal_availability"] is True
    assert contract["validation_before_gradient"]["materialized_train_dev_rows_required"] is True
    assert contract["validation_before_gradient"]["row_to_registered_system_compiler_required"] is True


def test_runtime_view_rows_compile_into_optimizer_facing_additional_views() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "runtime_view_builder",
            scripts/"build_n0_v02_full_envelope_runtime_view_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
            compile_behavioral_batch,
        )
        from test_n0_full_envelope_trainable_system_v1 import _TinyTokenizer

        rows=module.materialize_rows("train")[:2]
        compiled=compile_behavioral_batch(rows=rows,tokenizer=_TinyTokenizer())
        primary=compiled["primary_batch"]
        decisive=compiled["decisive_ablated_batch"]
        irrelevant=compiled["irrelevant_removed_batch"]
        targets=compiled["behavioral_targets"]

        assert primary["additional_view_source_input_ids"].shape[:2] == (2,3)
        assert primary["additional_view_descriptor_input_ids"].shape[:2] == (2,3)
        assert primary["additional_view_available"].shape == (2,3)
        assert primary["additional_view_reliability"].shape == (2,3)
        assert targets["recoverable_view_mask"].shape == (2,9)
        assert bool(primary["additional_view_available"][:,:2].all())
        assert not bool(primary["additional_view_available"][:,2].any())
        assert bool(targets["recoverable_view_mask"][:,6:].any())
        assert bool(
            (
                primary["additional_view_available"]
                & ~targets["recoverable_view_mask"][:,6:]
            ).any()
        )

        for b,row in enumerate(rows):
            for i,item in enumerate(row["additional_views"]):
                if item["decisive"]:
                    assert not bool(decisive["additional_view_available"][b,i])
                if item["irrelevant"]:
                    assert not bool(irrelevant["additional_view_available"][b,i])
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_runtime_view_reliability_reversal_changes_only_reliability_and_target() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "runtime_view_builder_reliability",
            scripts/"build_n0_v02_full_envelope_runtime_view_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows=[
            row for row in module.materialize_rows("train")
            if row["scenario_family"]=="runtime_view_reliability_reversal"
        ]
        assert len(rows)==2
        first,second=rows
        assert first["query"]==second["query"]
        assert first["candidate_answers"]==second["candidate_answers"]
        assert [
            x["source_text"] for x in first["additional_views"]
        ] == [
            x["source_text"] for x in second["additional_views"]
        ]
        assert [
            x["descriptor_text"] for x in first["additional_views"]
        ] == [
            x["descriptor_text"] for x in second["additional_views"]
        ]
        assert [
            x["available"] for x in first["additional_views"]
        ] == [
            x["available"] for x in second["additional_views"]
        ]
        r1=[x["reliability"] for x in first["additional_views"]]
        r2=[x["reliability"] for x in second["additional_views"]]
        assert r1[0]==r2[1] and r1[1]==r2[0]
        assert r1[2]==r2[2]==0.95
        assert first["public_target_index"] != second["public_target_index"]
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)
