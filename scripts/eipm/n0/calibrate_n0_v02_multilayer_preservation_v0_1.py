#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter

import train_n0_v02_relation_repair as rr
import train_n0_v02_evidence_selector_repair_v0_1 as sr


SCHEMA = "alice.eipm.n0.v02-multilayer-preservation-calibration.v0.1"
POLICY_SCHEMA = "alice.eipm.n0.v02-multilayer-preservation-policy.v0.1"
ABSOLUTE_FLOOR = 1e-6
REPEATABILITY_MULTIPLIER = 2.0
DEFAULT_REPEATS = 2

METRICS = {
    "ordinary.family_macro_target_support_mass": "higher_is_better",
    "ordinary.family_min_target_support_mass": "higher_is_better",
    "ordinary.family_macro_top1_support_accuracy": "higher_is_better",
    "ordinary.family_min_top1_support_accuracy": "higher_is_better",
    "endpoint.pair_accuracy": "higher_is_better",
    "endpoint.family_min_pair_accuracy": "higher_is_better",
    "endpoint.row_accuracy": "higher_is_better",
    "endpoint.mean_graph_target_margin": "higher_is_better",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def metric_value(payload: dict[str, Any], path: str) -> float:
    current: Any = payload
    for token in path.split("."):
        current = current[token]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise SystemExit(f"non-numeric preservation metric: {path}")
    value = float(current)
    if not math.isfinite(value):
        raise SystemExit(f"non-finite preservation metric: {path}")
    return value


def max_pairwise_delta(values: list[float]) -> float:
    return max(
        abs(values[i] - values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )


def load_parent(
    adapter_path: Path,
    graph_path: Path,
    device: torch.device,
) -> tuple[EvidenceViewAdapter, DualEndpointEvidenceGraphEncoder]:
    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    graph = DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config()).to(device)
    graph.load_state_dict(load_file(str(graph_path), device="cpu"), strict=True)
    for module in (adapter, graph):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()
    return adapter, graph


def one_repeat(
    *,
    ordinary_payload: dict[str, Any],
    endpoint_payload: dict[str, Any],
    adapter_path: Path,
    graph_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    adapter, graph = load_parent(adapter_path, graph_path, device)

    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev = rr.PairDataset(
        endpoint_payload,
        rr.pair_indices(endpoint_payload, "dev"),
    )
    ordinary_dev = rr.RowDataset(ordinary_payload, ordinary_dev_indices)
    if not ordinary_dev_indices:
        raise SystemExit("ordinary replay has no dev rows")
    if len(endpoint_dev) < 1:
        raise SystemExit("endpoint replay has no dev pairs")

    ordinary = sr.evaluate_replay(
        graph, adapter, ordinary_dev, ordinary_payload, device, batch_size
    )
    endpoint = sr.evaluate_pairs(
        graph, adapter, endpoint_dev, endpoint_payload, device, batch_size
    )
    return {"ordinary": ordinary, "endpoint": endpoint}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--preparation-receipt", required=True)
    p.add_argument("--preservation-contract", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--policy-output", required=True)
    p.add_argument("--receipt-output", required=True)
    p.add_argument("--repeat-count", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if args.repeat_count < 2:
        raise SystemExit("repeat-count must be >= 2")

    root = Path(args.repo_root).resolve()
    prep_path = Path(args.preparation_receipt).resolve()
    contract_path = Path(args.preservation_contract).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_path = Path(args.endpoint_replay_cache).resolve()
    policy_path = Path(args.policy_output).resolve()
    receipt_path = Path(args.receipt_output).resolve()

    for path in (
        prep_path,
        contract_path,
        adapter_path,
        graph_path,
        ordinary_path,
        endpoint_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing calibration input: {path}")
    if policy_path.exists() or receipt_path.exists():
        raise SystemExit("refusing to overwrite preservation calibration output")

    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("schema") != (
        "alice.eipm.n0.v02-relation-conditioned-multilayer-causal-study-preparation.v0.1"
    ):
        raise SystemExit("causal-study preparation schema drift")
    if prep.get("status") != (
        "PASS_CAUSAL_STUDY_INPUTS_FROZEN_TRAINING_DECISION_STILL_REQUIRED"
    ):
        raise SystemExit("causal-study preparation did not pass")

    expected = prep.get("artifact_sha256", {})
    for key, path in {
        "parent_adapter": adapter_path,
        "parent_graph": graph_path,
        "ordinary_replay_cache": ordinary_path,
        "endpoint_replay_cache": endpoint_path,
        "preservation_contract": contract_path,
    }.items():
        if expected.get(key) != sha(path):
            raise SystemExit(f"preparation artifact drift: {key}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != (
        "alice.eipm.n0.v02-relation-conditioned-multilayer-preservation-contract.v0.1"
    ):
        raise SystemExit("preservation contract schema drift")
    if (
        contract["numerical_tolerance_policy"][
            "candidate_result_may_not_define_tolerances"
        ]
        is not True
    ):
        raise SystemExit("candidate-independent tolerance contract drift")
    if (
        contract["numerical_tolerance_policy"]["tolerance_source"]
        != "canonical_parent_repeatability_only"
    ):
        raise SystemExit("calibration source drift")
    for key in (
        "training_authorized",
        "optimizer_authorized",
        "gradient_authorized",
        "new_gpu_run_authorized",
    ):
        if contract.get(key) is not False:
            raise SystemExit(f"preservation contract prematurely authorizes {key}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    ordinary_payload = torch.load(ordinary_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_path, map_location="cpu")

    repeats: list[dict[str, Any]] = []
    for index in range(args.repeat_count):
        result = one_repeat(
            ordinary_payload=ordinary_payload,
            endpoint_payload=endpoint_payload,
            adapter_path=adapter_path,
            graph_path=graph_path,
            device=device,
            batch_size=args.batch_size,
        )
        repeats.append(result)
        print(
            "canonical_repeat="
            + json.dumps({"index": index + 1, **result}, sort_keys=True),
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] = {}
    for metric, direction in METRICS.items():
        values = [metric_value(repeat, metric) for repeat in repeats]
        drift = max_pairwise_delta(values)
        atol = max(ABSOLUTE_FLOOR, REPEATABILITY_MULTIPLIER * drift)
        baseline[metric] = sum(values) / len(values)
        rows.append(
            {
                "path": metric,
                "direction": direction,
                "canonical_repeat_values": values,
                "max_pairwise_abs_delta": drift,
                "baseline_mean": baseline[metric],
                "atol": atol,
                "rtol": 0.0,
            }
        )

    policy = {
        "schema": POLICY_SCHEMA,
        "status": "FROZEN_BEFORE_CANDIDATE_TRAINING",
        "selected_before_candidate_result": True,
        "candidate_result_observed": False,
        "source": "canonical_parent_repeatability_only",
        "repeat_count": args.repeat_count,
        "absolute_floor": ABSOLUTE_FLOOR,
        "repeatability_multiplier": REPEATABILITY_MULTIPLIER,
        "metrics": rows,
        "pass_rule": (
            "for every higher-is-better metric, candidate value must be >= "
            "baseline_mean - atol"
        ),
        "candidate_may_change_policy": False,
    }
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_PARENT_ONLY_PRESERVATION_CALIBRATION",
        "git_revision": git_revision(root),
        "preparation_receipt_sha256": sha(prep_path),
        "preservation_contract_sha256": sha(contract_path),
        "parent_adapter_sha256": sha(adapter_path),
        "parent_graph_sha256": sha(graph_path),
        "ordinary_replay_cache_sha256": sha(ordinary_path),
        "endpoint_replay_cache_sha256": sha(endpoint_path),
        "candidate_supplied": False,
        "candidate_result_observed": False,
        "gradient_performed": False,
        "optimizer_created": False,
        "gpu_required": False,
        "repeat_count": args.repeat_count,
        "policy": policy,
        "interface_training_gate_satisfied": True,
        "authorization_scope": (
            "one bounded relation-conditioned multilayer interface experiment only"
        ),
        "semantic_backbone_trainable": False,
        "structured_state_trainable": False,
        "parent_adapter_trainable": False,
        "parent_graph_trainable": False,
        "scale_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
    }

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("status=PASS_PARENT_ONLY_PRESERVATION_CALIBRATION")
    print(f"policy={policy_path}")
    print(f"receipt={receipt_path}")
    print("candidate_result_observed=false")
    print("gradient_performed=false")
    print("optimizer_created=false")
    print("gpu_required=false")
    print("interface_training_gate_satisfied=true")
    print("scale_authorized=false")
    print("heldout_opening_authorized=false")
    print("n0_complete=false")


if __name__ == "__main__":
    main()
