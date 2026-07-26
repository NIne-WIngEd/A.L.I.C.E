from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from alice_conversation.final_evaluation import report_to_dict, run_conversation_final_evaluation
from alice_conversation.final_evaluation_contract import load_conversation_evaluation_submissions, load_conversation_final_evaluation_benchmark, load_conversation_final_evaluation_policy

def _within(path: Path, root: Path) -> bool:
    try: path.relative_to(root); return True
    except ValueError: return False

def main(argv=None):
    parser=argparse.ArgumentParser(description="Run the offline synthetic P3.10 conversational evaluation.")
    parser.add_argument("--submissions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--benchmark",type=Path)
    args=parser.parse_args(argv)
    output=args.output.expanduser().resolve()
    if _within(output,ROOT.resolve()): raise SystemExit("Evaluation output must remain outside the repository.")
    policy=load_conversation_final_evaluation_policy(args.policy)
    benchmark=load_conversation_final_evaluation_benchmark(args.benchmark,policy=policy)
    submissions=load_conversation_evaluation_submissions(args.submissions,benchmark=benchmark)
    report=run_conversation_final_evaluation(submissions=submissions,benchmark=benchmark,policy=policy)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise SystemExit("Refusing to overwrite an existing evaluation report.")
    output.write_text(json.dumps(report_to_dict(report),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("passed="+str(report.passed).lower()); print("report_digest="+report.report_digest); print("output="+str(output))
    return 0 if report.passed else 1
if __name__=="__main__": raise SystemExit(main())
