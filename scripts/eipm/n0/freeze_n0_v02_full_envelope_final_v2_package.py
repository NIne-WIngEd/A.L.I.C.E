#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STATUS="FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--package-config",required=True)
    p.add_argument("--evaluator-contract",required=True)
    p.add_argument("--final-contract",required=True)
    p.add_argument("--synthetic-final-rows",required=True)
    p.add_argument("--semantic-final-rows",required=True)
    p.add_argument("--package-manifest",required=True)
    p.add_argument("--fewrel-final-rows",required=True)
    p.add_argument("--fewrel-final-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--audit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    paths={name:Path(getattr(args,name)) for name in (
        "package_config","evaluator_contract","final_contract","synthetic_final_rows","semantic_final_rows",
        "package_manifest","fewrel_final_rows","fewrel_final_bank","fewrel_manifest","audit","output"
    )}
    if paths["output"].exists():
        raise SystemExit("refusing to overwrite final-v2 freeze receipt")
    audit=json.loads(paths["audit"].read_text(encoding="utf-8"))
    package=json.loads(paths["package_config"].read_text(encoding="utf-8"))
    evaluator=json.loads(paths["evaluator_contract"].read_text(encoding="utf-8"))
    manifest=json.loads(paths["package_manifest"].read_text(encoding="utf-8"))
    if audit.get("status")!="PASS_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1":
        raise SystemExit("cannot freeze final-v2 package without audit PASS")
    if any([
        audit.get("results_observed") is not False,
        package.get("results_observed") is not False,
        evaluator.get("results_observed") is not False,
        manifest.get("results_observed") is not False,
    ]):
        raise SystemExit("cannot freeze package after observing candidate results")
    if evaluator.get("final_opening_authorized") is not False:
        raise SystemExit("evaluator contract unexpectedly authorizes final opening")

    receipt={
        "schema":"alice.eipm.n0.full-envelope-final-v2-freeze-receipt.v1",
        "status":STATUS,
        "hashes":{
            name:sha256(path)
            for name,path in paths.items()
            if name!="output"
        },
        "registered_system":"N0FullEnvelopeTrainableSystemV1",
        "legacy_final_v1_authority":False,
        "legacy_final_v3_evaluator_authority":False,
        "results_observed":False,
        "training_authorized":False,
        "model_selection_authorized":False,
        "final_opening_authorized":False,
        "gradient_performed":False,
        "optimizer_performed":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    paths["output"].parent.mkdir(parents=True,exist_ok=True)
    paths["output"].write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
