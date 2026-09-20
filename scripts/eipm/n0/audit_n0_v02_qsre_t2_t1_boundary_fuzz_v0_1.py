from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_t1_executor import (
    QSRET1Config,
    QSRET1Executor,
    QSRET1OracleOperator,
)

from train_n0_v02_qsre_t2_operator_v0_1 import (
    STRUCTURAL_KEYS,
    _load_executor,
    _safe_operator_for_frozen_t1_eval,
)

NONE_RELATION = 6
CONTROL_FALLBACK = 0
CONTROL_RELATIONAL = 1
CONTROL_DEFER = 2


def repeat_first(tensor: torch.Tensor, count: int) -> torch.Tensor:
    index = torch.zeros(count, dtype=torch.long)
    return tensor.index_select(0, index).clone()


def build_cases(fields: int) -> tuple[QSRET1OracleOperator, torch.Tensor, list[dict]]:
    relation_patterns = [
        (0, NONE_RELATION),
        (5, NONE_RELATION),
        (0, 1),
        (NONE_RELATION, NONE_RELATION),
    ]
    cases = []
    rel_rows = []
    rel_masks = []
    roles = []
    operations = []
    controls = []
    focus_rows = []
    applicability = []

    for control, operation, role, relation_pair, focus_present in itertools.product(
        (CONTROL_FALLBACK, CONTROL_RELATIONAL, CONTROL_DEFER),
        range(5),
        range(4),
        relation_patterns,
        (False, True),
    ):
        cases.append(
            {
                "control": control,
                "operation": operation,
                "role": role,
                "relation_pair": list(relation_pair),
                "focus_present": focus_present,
            }
        )
        rel_rows.append([min(int(x), 5) for x in relation_pair])
        if control == CONTROL_RELATIONAL:
            mask = [int(x) != NONE_RELATION for x in relation_pair]
            if not mask[0]:
                mask[1] = False
        else:
            mask = [False, False]
        rel_masks.append(mask)
        roles.append(role)
        operations.append(operation)
        controls.append(control)

        focus = torch.zeros(fields)
        if focus_present:
            focus[0] = 1.0
        focus_rows.append(focus)
        applicability.append({0: 0.10, 1: 0.90, 2: 0.50}[control])

    operator = QSRET1OracleOperator(
        relation_sequence_id=torch.tensor(rel_rows, dtype=torch.long),
        relation_sequence_mask=torch.tensor(rel_masks, dtype=torch.bool),
        role_id=torch.tensor(roles, dtype=torch.long),
        operation_id=torch.tensor(operations, dtype=torch.long),
        focus_field_weight=torch.stack(focus_rows, dim=0),
        context=torch.zeros(len(cases), 4),
        applicability=torch.tensor(applicability, dtype=torch.float32),
    )
    return operator, torch.tensor(controls, dtype=torch.long), cases


def execute_fuzz(
    *,
    executor: QSRET1Executor,
    structural_row: dict[str, torch.Tensor],
) -> dict:
    fields = int(structural_row["field_state"].size(1))
    operator, predicted_control, cases = build_cases(fields)
    safe_operator, invalid_relational = _safe_operator_for_frozen_t1_eval(
        operator,
        predicted_control=predicted_control,
    )

    batch = len(cases)
    structural = {
        key: repeat_first(structural_row[key], batch)
        for key in STRUCTURAL_KEYS
    }

    with torch.inference_mode():
        result = executor(**structural, operator=safe_operator)

    for key in ("relational_probability", "control_state", "readout_field_mask"):
        if key not in result:
            raise RuntimeError(f"executor output missing {key}")
        tensor = result[key]
        if tensor.size(0) != batch:
            raise RuntimeError(f"executor output batch drift for {key}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"nonfinite executor output in {key}")

    expected_invalid = sum(
        int(
            c["control"] == CONTROL_RELATIONAL
            and c["operation"] == 1
            and not c["focus_present"]
        )
        for c in cases
    )
    observed_invalid = int(invalid_relational.sum().item())
    if observed_invalid != expected_invalid:
        raise RuntimeError(
            f"invalid relational path count drift expected={expected_invalid} observed={observed_invalid}"
        )

    return {
        "cases": batch,
        "expected_invalid_relational_path_without_focus": expected_invalid,
        "observed_invalid_relational_path_without_focus": observed_invalid,
        "executor_completed_all_cases": True,
        "nonfinite_outputs": False,
    }


def dummy_structural(config: QSRET1Config) -> dict[str, torch.Tensor]:
    batch = 1
    fields = 6
    edges = 4
    return {
        "field_state": torch.randn(batch, fields, config.field_state_dim),
        "field_metadata": torch.randn(batch, fields, config.field_metadata_dim),
        "field_valid_mask": torch.ones(batch, fields, dtype=torch.bool),
        "edge_index": torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]], dtype=torch.long),
        "edge_relation_id": torch.tensor([[0, 1, 4, 5]], dtype=torch.long),
        "edge_metadata": torch.randn(batch, edges, config.edge_metadata_dim),
        "edge_valid_mask": torch.ones(batch, edges, dtype=torch.bool),
        "field_support_weight": torch.zeros(batch, fields),
        "edge_support_weight": torch.ones(batch, edges),
    }


def self_test() -> None:
    torch.manual_seed(17)
    config = QSRET1Config(
        field_state_dim=16,
        field_metadata_dim=3,
        edge_metadata_dim=2,
        operator_context_dim=4,
        model_dim=24,
        num_relations=6,
        num_roles=4,
        num_operations=5,
        dropout=0.0,
    )
    executor = QSRET1Executor(config).eval()
    result = execute_fuzz(executor=executor, structural_row=dummy_structural(config))
    if result["cases"] != 480:
        raise RuntimeError("fuzz case count drift")
    print("PASS_QSRE_T2_T1_BOUNDARY_FUZZ_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--contract")
    parser.add_argument("--prepared-cache")
    parser.add_argument("--t1-checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for name in ("contract", "prepared_cache", "t1_checkpoint", "output"):
        if not getattr(args, name):
            raise SystemExit(f"missing --{name.replace('_', '-')}")

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    model_contract = contract["t1_executor_model"]
    prepared = torch.load(Path(args.prepared_cache), map_location="cpu")
    if prepared.get("schema") != "alice.eipm.n0.qsre-t2-tokenized-preparation.v0.1":
        raise SystemExit("prepared cache schema drift")

    executor = _load_executor(
        Path(args.t1_checkpoint),
        model_contract,
        torch.device("cpu"),
    )

    structural_row = {
        key: prepared["dev"][key][:1].clone()
        for key in STRUCTURAL_KEYS
    }
    result = execute_fuzz(executor=executor, structural_row=structural_row)
    payload = {
        "schema": "alice.eipm.n0.qsre-t2-t1-boundary-fuzz.v0.1",
        "status": "PASS_QSRE_T2_T1_BOUNDARY_TOTALITY_FUZZ",
        **result,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "private_identity_data": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
