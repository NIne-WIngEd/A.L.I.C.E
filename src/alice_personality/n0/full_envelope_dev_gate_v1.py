from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PASS_STATIC="PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"


def resolve_metric(metrics: Mapping[str,Any], path: str) -> float:
    value: Any=metrics
    for part in str(path).split("."):
        if not isinstance(value,Mapping) or part not in value:
            raise KeyError(f"DEV metric path missing: {path}")
        value=value[part]
    if isinstance(value,bool):
        return float(value)
    if not isinstance(value,(int,float)):
        raise TypeError(f"DEV metric path is not scalar: {path}")
    return float(value)


def compare(value: float, *, comparison: str, threshold: float) -> bool:
    if comparison==">=":
        return value>=threshold
    if comparison=="<=":
        return value<=threshold
    if comparison==">":
        return value>threshold
    if comparison=="<":
        return value<threshold
    if comparison=="==":
        return value==threshold
    raise ValueError(f"unsupported DEV comparison: {comparison!r}")


def _empirical_result(
    *,
    spec: Mapping[str,Any],
    metrics: Mapping[str,Any],
) -> dict[str,Any]:
    path=str(spec["metric"])
    value=resolve_metric(metrics,path)
    threshold=float(spec["threshold"])
    op=str(spec["comparison"])
    metric_pass=compare(value,comparison=op,threshold=threshold)

    coverage_path=spec.get("coverage_metric")
    minimum_coverage=spec.get("minimum_coverage")
    if (coverage_path is None) != (minimum_coverage is None):
        raise ValueError(
            "coverage_metric and minimum_coverage must be supplied together"
        )
    coverage_value=None
    coverage_pass=True
    if coverage_path is not None:
        coverage_path=str(coverage_path)
        coverage_value=resolve_metric(metrics,coverage_path)
        minimum_coverage=float(minimum_coverage)
        coverage_pass=coverage_value >= minimum_coverage

    return {
        "kind":"empirical",
        "metric":path,
        "value":value,
        "comparison":op,
        "threshold":threshold,
        "metric_passed":metric_pass,
        "coverage_metric":coverage_path,
        "coverage_value":coverage_value,
        "minimum_coverage":minimum_coverage,
        "coverage_passed":coverage_pass,
        "passed":metric_pass and coverage_pass,
    }


def _composite_result(
    *,
    spec: Mapping[str,Any],
    metrics: Mapping[str,Any],
) -> dict[str,Any]:
    parts=[]
    for item in spec.get("all") or []:
        parts.append(_empirical_result(spec=item,metrics=metrics))
    if not parts:
        raise ValueError("composite DEV gate has no criteria")
    return {
        "kind":"composite",
        "criteria":parts,
        "passed":all(bool(x["passed"]) for x in parts),
    }


def _static_result(
    *,
    spec: Mapping[str,Any],
    proof_contract: Mapping[str,Any],
    static_receipt: Mapping[str,Any],
) -> dict[str,Any]:
    if static_receipt.get("status")!=PASS_STATIC:
        raise ValueError("exact-head static proof receipt is not PASS")
    obligations={
        str(item.get("id")):item
        for item in proof_contract.get("obligations") or []
    }
    required=list(map(str,spec.get("obligation_ids") or []))
    if not required:
        raise ValueError("static DEV gate has no proof obligations")
    bad=[]
    for oid in required:
        item=obligations.get(oid)
        if item is None or item.get("kind")!="STATIC_REQUIRED":
            bad.append(oid)
    return {
        "kind":"static_source_proof",
        "obligation_ids":required,
        "proof_contract_resolved":not bad,
        "unresolved_obligation_ids":bad,
        "passed":not bad,
    }


def evaluate_stage_gate_registry(
    *,
    registry: Mapping[str,Any],
    stage: str,
    metrics: Mapping[str,Any],
    proof_contract: Mapping[str,Any],
    static_receipt: Mapping[str,Any],
    _stage_stack: tuple[str,...] = (),
) -> dict[str,Any]:
    if registry.get("schema")!="alice.eipm.n0.full-envelope-dev-gate-registry.v1":
        raise ValueError("DEV gate registry schema drift")
    stages=registry.get("stages") or {}
    if stage not in stages:
        raise ValueError(f"DEV gate stage not registered: {stage}")
    if stage in _stage_stack:
        raise ValueError(
            "cyclic retained DEV stage mapping: "
            + " -> ".join((*_stage_stack,stage))
        )
    stage_spec=stages[stage]
    gates=stage_spec.get("gates") or {}
    results={}
    errors=[]
    for name,spec in gates.items():
        try:
            kind=str(spec.get("kind"))
            if kind=="empirical":
                value=_empirical_result(spec=spec,metrics=metrics)
            elif kind=="composite":
                value=_composite_result(spec=spec,metrics=metrics)
            elif kind=="static_source_proof":
                value=_static_result(
                    spec=spec,
                    proof_contract=proof_contract,
                    static_receipt=static_receipt,
                )
            elif kind=="retained_stage_gates":
                retained=list(map(str,spec.get("stages") or []))
                if not retained:
                    raise ValueError("retained-stage DEV gate has no stages")
                retained_results=[
                    evaluate_stage_gate_registry(
                        registry=registry,
                        stage=retained_stage,
                        metrics=metrics,
                        proof_contract=proof_contract,
                        static_receipt=static_receipt,
                        _stage_stack=(*_stage_stack,stage),
                    )
                    for retained_stage in retained
                ]
                value={
                    "kind":"retained_stage_gates",
                    "stages":retained,
                    "stage_results":retained_results,
                    "passed":all(
                        bool(item.get("stage_gate_pass"))
                        for item in retained_results
                    ),
                }
            else:
                raise ValueError(f"unsupported DEV gate kind: {kind!r}")
            results[str(name)]=value
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            results[str(name)]={
                "kind":str(spec.get("kind")),
                "passed":False,
                "error":str(exc),
            }
    complete=bool(stage_spec.get("mapping_complete")) and not errors
    passed=complete and bool(results) and all(
        bool(value.get("passed")) for value in results.values()
    )
    return {
        "stage":stage,
        "mapping_complete":bool(stage_spec.get("mapping_complete")),
        "gate_count":len(results),
        "gate_results":results,
        "mapping_errors":errors,
        "stage_gate_coverage_complete":complete,
        "stage_gate_pass":passed,
        "final_results_used":False,
    }
