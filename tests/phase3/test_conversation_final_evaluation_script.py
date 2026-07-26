from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from alice_conversation.final_evaluation_contract import canonical_json
from _final_evaluation_helpers import ROOT, passing_submissions

def _bundle(path):
    items=passing_submissions(); value={"conversation_final_evaluation_submission_schema_version":1,"benchmark_id":"phase3-conversation-adversarial-v1","test_set_version":"p3.10-v1","submissions":[{"case_id":x.case_id,"actual_outcome":x.actual_outcome,"signals":list(x.signals),"violation_codes":list(x.violation_codes),"observation_digest":x.observation_digest} for x in items]}; path.write_text(json.dumps(value),encoding="utf-8")

def test_script_writes_private_report_outside_repository(tmp_path):
    submissions=tmp_path/"submissions.json"; output=tmp_path/"report.json"; _bundle(submissions)
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_phase3_conversation_evaluation.py"),"--submissions",str(submissions),"--output",str(output)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0 and output.exists() and "passed=true" in result.stdout

def test_script_refuses_repository_output(tmp_path):
    submissions=tmp_path/"submissions.json"; _bundle(submissions); output=ROOT/f".p310-forbidden-{tmp_path.name}.json"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_phase3_conversation_evaluation.py"),"--submissions",str(submissions),"--output",str(output)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode!=0 and not output.exists()

def test_script_refuses_overwrite(tmp_path):
    submissions=tmp_path/"submissions.json"; output=tmp_path/"report.json"; _bundle(submissions); output.write_text("existing")
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_phase3_conversation_evaluation.py"),"--submissions",str(submissions),"--output",str(output)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode!=0 and output.read_text()=="existing"
