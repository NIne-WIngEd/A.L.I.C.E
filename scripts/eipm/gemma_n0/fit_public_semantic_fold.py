"""Closed PUBLIC cached-feature consumer; never a personality approval."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.alice_personality.gemma_n0.public_semantic_fold import fit_public_semantic_fold

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--package-directory",required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--closure-sha256",required=True)
    parser.add_argument("--output-directory",required=True)
    parser.add_argument("--cpu-threads",type=int,default=4)
    args=parser.parse_args(argv)
    receipt=fit_public_semantic_fold(args.package_directory,expected_manifest_sha256=args.manifest_sha256,
        expected_closure_sha256=args.closure_sha256,output_directory=args.output_directory,cpu_threads=args.cpu_threads)
    print(json.dumps({key:receipt[key] for key in ("schema","state","status","receipt_sha256","n0_approved","repair_selected")},sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
