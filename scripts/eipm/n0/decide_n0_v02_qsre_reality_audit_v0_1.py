from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--shortcut", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--fuzz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = load(Path(args.contract))
    runtime = load(Path(args.runtime))
    shortcut = load(Path(args.shortcut))
    semantic = load(Path(args.semantic))
    fuzz = load(Path(args.fuzz))

    expected = {
        "runtime": "PASS_QSRE_RUNTIME_BEHAVIORAL_PARITY_AUDIT",
        "shortcut": "PASS_QSRE_T2_SHORTCUT_AUDIT",
        "semantic": "PASS_QSRE_FROZEN_SEMANTIC_REALITY_PROBE",
        "fuzz": "PASS_QSRE_T2_T1_BOUNDARY_TOTALITY_FUZZ",
    }
    observed = {
        "runtime": runtime.get("status"),
        "shortcut": shortcut.get("status"),
        "semantic": semantic.get("status"),
        "fuzz": fuzz.get("status"),
    }
    for key, status in expected.items():
        if observed[key] != status:
            raise SystemExit(f"required audit track did not pass: {key}={observed[key]!r}")

    rules = contract["decision_rules"]
    semantic_summary = semantic["summary"]
    shortcut_summary = shortcut["summary"]

    mean_lift = float(semantic_summary["mean_best_layer_normalized_lift"])
    min_lift = float(semantic_summary["min_best_layer_normalized_lift"])
    critical_heads = semantic_summary["critical_heads"]

    critical_lifts = {}
    for head in critical_heads:
        critical_lifts[head] = float(
            semantic["reality"][head]["best"]["normalized_lift_over_majority"]
        )

    reopen = rules["semantic_reopen"]
    low_threshold = float(reopen["or_critical_heads_below_lift"]["threshold"])
    low_count_required = int(reopen["or_critical_heads_below_lift"]["count"])
    low_count = sum(int(v < low_threshold) for v in critical_lifts.values())

    dev_sig = float(shortcut_summary["unigram_nb_signature_t2_dev"])
    reality_sig = float(shortcut_summary["unigram_nb_signature_reality"])
    signature_gap = dev_sig - reality_sig

    if (
        mean_lift < float(reopen["mean_best_layer_normalized_lift_below"])
        or low_count >= low_count_required
    ):
        decision = str(reopen["decision"])
        reason = (
            "Frozen semantic states do not retain enough transferable operator signal "
            "on the realistic public audit distribution."
        )
    else:
        rebuild = rules["benchmark_rebuild"]
        if (
            dev_sig >= float(rebuild["t2_dev_unigram_signature_at_least"])
            and signature_gap
            >= float(rebuild["t2_dev_minus_reality_unigram_signature_at_least"])
            and mean_lift >= float(rebuild["semantic_mean_lift_at_least"])
        ):
            decision = str(rebuild["decision"])
            reason = (
                "The frozen semantic backbone retains transferable signal, but a shallow "
                "query-only model solves the synthetic T2 DEV distribution much better "
                "than the realistic audit distribution. The benchmark is the weak link."
            )
        else:
            keep = rules["keep_current_family"]
            if (
                mean_lift >= float(keep["semantic_mean_lift_at_least"])
                and min_lift >= float(keep["semantic_min_lift_at_least"])
                and signature_gap
                < float(keep["t2_dev_minus_reality_unigram_signature_below"])
            ):
                decision = str(keep["decision"])
                reason = (
                    "Runtime/evaluator totality is healthy and the frozen semantic "
                    "representation transfers without a large synthetic-to-reality "
                    "shortcut gap."
                )
            else:
                decision = str(rules["otherwise"])
                reason = (
                    "The combined evidence does not justify either semantic retraining "
                    "or preserving the current operator interface unchanged."
                )

    payload = {
        "schema": "alice.eipm.n0.qsre-reality-audit-decision.v0.1",
        "status": "PASS_QSRE_REALITY_AUDIT_DECISION",
        "decision": decision,
        "reason": reason,
        "evidence": {
            "runtime_status": runtime["status"],
            "fuzz_status": fuzz["status"],
            "unigram_nb_signature_t2_dev": dev_sig,
            "unigram_nb_signature_reality": reality_sig,
            "unigram_signature_distribution_gap": signature_gap,
            "semantic_mean_best_layer_normalized_lift": mean_lift,
            "semantic_min_best_layer_normalized_lift": min_lift,
            "semantic_critical_head_lifts": critical_lifts,
            "semantic_best_accuracy": semantic_summary["best_accuracy"],
            "semantic_best_layers": semantic_summary["best_layers"],
        },
        "governance": {
            "optimizer": False,
            "gradient": False,
            "gpu": False,
            "automatic_training_authorization": False,
            "learned_support": False,
            "test_open": False,
            "private_identity_gradient": False,
            "production_promotion": False,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
