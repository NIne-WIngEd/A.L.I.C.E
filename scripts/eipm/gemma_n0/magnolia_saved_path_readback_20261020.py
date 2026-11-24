"""Read the retained native representations; no forward, training, or payload export."""
import importlib.util
import json
import math
import os
from hashlib import sha256
from pathlib import Path

ROOT = Path("/homes/01/mxrayan/rayan-compute")
REV = "333fcd0c8239fc81a6459b13d53bbe592b33fda6"
GUARD = "f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4"
PHASE_SHA = "34f0e49703985d022b92846cecc297e2d69eee7e6e81248e911d01e7a47ad8f1"
NAME = "rayan-saved-path-readback-20261020"
KINDS = {"original", "decisive_condition", "paraphrase", "irrelevant_social", "neutral_extra_text"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def seal(value):
    if sha256(canonical({k: v for k, v in value.items() if k != "receipt_sha256"})).hexdigest() != value["receipt_sha256"]:
        raise ValueError("saved receipt changed")
    return value


def difference(left, right):
    """Matched coordinates only; never silently zip mismatched shapes."""
    pairs = []

    def visit(a, b):
        if type(a) is list and type(b) is list:
            if len(a) != len(b):
                raise ValueError("native coordinate shape changed")
            for x, y in zip(a, b):
                visit(x, y)
        elif type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
        else:
            raise ValueError("native coordinate must be finite numeric")

    visit(left, right)
    if not pairs:
        raise ValueError("native coordinates are empty")
    delta = [b-a for a, b in pairs]
    rms = lambda xs: math.sqrt(math.fsum(x*x for x in xs) / len(xs))
    scale = rms([a for a, _ in pairs])
    return {"coordinates": len(pairs), "mean_absolute_delta": math.fsum(map(abs, delta))/len(delta),
            "max_absolute_delta": max(map(abs, delta)), "rms_delta": rms(delta),
            "baseline_rms": scale, "relative_rms_delta": rms(delta)/scale if scale else None}


def summarize(rows):
    if len(rows) != 10 or len({row["document_id"] for row in rows}) != 10:
        raise ValueError("all ten unique contexts required")
    groups = {}
    for row in rows:
        group = groups.setdefault(row["parent_document_id"], {})
        kind = row["comparison_kind"]
        if kind in group or kind not in KINDS:
            raise ValueError("duplicate or unknown comparison")
        group[kind] = row
    if len(groups) != 2 or any(set(group) != KINDS for group in groups.values()):
        raise ValueError("both complete five-context families required")
    observations = {}
    for group in groups.values():
        original = group["original"]["response"]["decision"]
        for kind, row in group.items():
            if kind == "original":
                continue
            other = row["response"]["decision"]
            for key in ("candidate_ids", "concept_ids", "candidate_mask"):
                if original[key] != other[key]:
                    raise ValueError("native semantic coordinates changed")
            for key in ("concept_mask", "view_masks"):
                if original["native_latent"][key] != other["native_latent"][key]:
                    raise ValueError("native authority masks changed")
            if original["conditional_concepts"]["concept_source_mask"] != other["conditional_concepts"]["concept_source_mask"]:
                raise ValueError("native conditional authority changed")
            fields = {}
            for field in ("concept_applicability", "concept_uncertainty", "concept_source_support", "concept_source_disconfirmation"):
                fields["n1:"+field] = (original["conditional_concepts"][field], other["conditional_concepts"][field])
            for field in ("concepts", "candidates"):
                fields[("n1:" if field == "concepts" else "n2:")+field] = (original["native_latent"][field], other["native_latent"][field])
            if original["native_latent"]["views"].keys() != other["native_latent"]["views"].keys():
                raise ValueError("native view set changed")
            for view, values in original["native_latent"]["views"].items():
                fields["n1:view:"+view] = (values, other["native_latent"]["views"][view])
            for field in ("co_valid_probabilities", "contraindications", "preferences"):
                fields["output:"+field] = (original[field], other[field])
            for field, pair in fields.items():
                observations.setdefault(kind, {}).setdefault(field, []).append(difference(*pair))
    result = {}
    for kind, fields in observations.items():
        result[kind] = {}
        for field, values in fields.items():
            result[kind][field] = {
                "paired_families": len(values),
                "coordinates": sum(v["coordinates"] for v in values),
                "max_absolute_delta": max(v["max_absolute_delta"] for v in values),
                "mean_family_absolute_delta": math.fsum(v["mean_absolute_delta"] for v in values)/len(values),
                "mean_family_rms_delta": math.fsum(v["rms_delta"] for v in values)/len(values),
                "mean_family_relative_rms_delta": (
                    math.fsum(v["relative_rms_delta"] for v in values)/len(values)
                    if all(v["relative_rms_delta"] is not None for v in values) else None),
            }
    return result


def run(namespace):
    isolation = namespace._p2_guard()  # FIRST, including before protected stat/hash.
    os.umask(0o077)
    code = namespace._bounded_path(os.environ["PERSONALITY_REPO_ROOT"], ROOT, file=False)
    namespace._clean_code(code, REV)
    held = {}

    def read(path, digest):
        path = namespace._bounded_path(str(path), ROOT, file=True)
        info = path.stat()
        if info.st_mode & 0o777 != 0o600 or info.st_size > 128*1024**2:
            raise ValueError("saved input mode or bound differs")
        data = path.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("saved input identity differs")
        held[path] = digest
        return data

    phase = seal(json.loads(read(ROOT/"jobs/rayan-cumulative-phase-attribution-20261018.json", PHASE_SHA)))
    assert phase["job_id"] == 578460 and phase["endpoint_diagnostic_packet_reproductions"] == 20
    assert phase["complete_source_and_checkpoint_custody_verified"]
    results = {}
    matched = None
    for stage in ("parent", "after_n1", "successor"):
        path = ROOT/"rayan-personality/private-compile/rayan-cumulative-phase-attribution-578460"/(stage+"-raw-native-diagnostics.protected.jsonl")
        rows = [json.loads(line) for line in read(path, phase["diagnostic_private_file_sha256"][stage]).splitlines()]
        for row in rows:
            response = seal(row["response"])
            selected = seal(response["preference_selection"])
            assert selected["document_sha256"] == row["document_sha256"]
            assert selected["model_state_sha256"] == phase["phase_states"][stage]["model_state_sha256"]
            assert response["packet_sha256"] == selected["packet_sha256"]
            assert selected["checkpoint_manifest_sha256"] == phase["phase_states"][stage]["checkpoint_manifest_sha256"]
        bindings = {row["document_id"]: (row["document_sha256"], row["parent_document_id"], row["comparison_kind"]) for row in rows}
        if matched is None:
            matched = bindings
        assert matched == bindings
        results[stage] = summarize(rows)
    for path, digest in held.items():
        assert sha256(path.read_bytes()).hexdigest() == digest
    namespace._clean_code(code, REV)
    result = {"schema": "alice-personality-saved-native-path-readback-v1", "job_id": int(os.environ["SLURM_JOB_ID"]),
              "owner": "mxrayan", "source_phase_sha256": PHASE_SHA, "namespace_sha256": GUARD, "isolation": isolation,
              "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "native_records": 30,
              "source_record_seals_verified": True, "matched_coordinate_changes": results,
              "new_forward_passes": 0, "new_gradient_steps": 0, "private_payload_exported": False,
              "causal_intervention": False, "independent_qualification": False, "n0_approved": False,
              "personality_approved": False}
    result["receipt_sha256"] = sha256(canonical(result)).hexdigest()
    with (ROOT/"jobs"/(NAME+".json")).open("xb") as stream:
        stream.write(canonical(result)+b"\n")
    print(json.dumps({"stage": "complete_saved_path_readback", "receipt_sha256": result["receipt_sha256"]}), flush=True)


if __name__ == "__main__":
    path = Path(os.environ["PERSONALITY_REPO_ROOT"])/"scripts/eipm/n1/enter_private_network_namespace.py"
    assert sha256(path.read_bytes()).hexdigest() == GUARD
    spec = importlib.util.spec_from_file_location("saved_path_namespace", path)
    namespace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(namespace)
    try:
        run(namespace)
    except BaseException as error:
        import traceback
        print(json.dumps({"state": "SAVED_PATH_READBACK_REFUSED", "exception_type": type(error).__name__,
                          "public_code_frames": [{"module": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                                 for f in traceback.extract_tb(error.__traceback__)]}), flush=True)
        raise SystemExit(2)
