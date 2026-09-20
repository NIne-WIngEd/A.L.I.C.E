from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from qsre_production_runtime import sha256


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--general-corpus",required=True)
    p.add_argument("--relational-corpus",required=True)
    p.add_argument("--final-schema-cache",required=True)
    p.add_argument("--relational-cache",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    contract_path=Path(args.contract)
    manifest_path=Path(args.manifest)
    general_path=Path(args.general_corpus)
    relational_path=Path(args.relational_corpus)
    schema_path=Path(args.final_schema_cache)
    cache_path=Path(args.relational_cache)
    output=Path(args.output)

    contract=json.loads(contract_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    if contract.get("status")!="FROZEN_NATIVE_N0_OBJECTIVE_GATE_BEFORE_PRODUCTION_RESULTS":
        raise SystemExit("final validation contract drift")
    if manifest.get("status")!="FROZEN_BY_BUILD_SOURCE_BEFORE_PRODUCTION_MODEL_RESULTS":
        raise SystemExit("final validation manifest drift")
    if manifest.get("general_sha256")!=sha256(general_path):
        raise SystemExit("general final-validation corpus drift")
    if manifest.get("relational_sha256")!=sha256(relational_path):
        raise SystemExit("relational final-validation corpus drift")
    cache=torch_load(cache_path)
    if cache.get("schema")!="alice.eipm.n0.qsre-final-self-validation-cache.v1":
        raise SystemExit("final relational semantic cache drift")
    if cache.get("relational_corpus_sha256")!=sha256(relational_path):
        raise SystemExit("final relational cache/corpus lineage drift")
    if cache.get("training_authorized") is not False:
        raise SystemExit("final relational cache unexpectedly authorizes training")

    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    receipt={
        "schema":"alice.eipm.n0.final-self-validation-freeze.v1",
        "status":"FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS",
        "git_revision":head,
        "contract_sha256":sha256(contract_path),
        "manifest_sha256":sha256(manifest_path),
        "general_sha256":sha256(general_path),
        "relational_sha256":sha256(relational_path),
        "final_schema_cache_sha256":sha256(schema_path),
        "relational_cache_sha256":sha256(cache_path),
        "training_authorized":False,
        "checkpoint_selection_authorized":False,
        "threshold_changes_after_results_forbidden":True,
        "results_observed_before_freeze":False,
        "external_validator":False,
        "validation_owner":"alice_native_self_validation_harness",
        "private_identity_data":False,
        "private_identity_gradient":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2,sort_keys=True))


def torch_load(path: Path):
    import torch
    return torch.load(path,map_location="cpu")


if __name__=="__main__":
    main()
