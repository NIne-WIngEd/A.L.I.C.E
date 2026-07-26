from __future__ import annotations
import json
from dataclasses import replace
import pytest
from alice_conversation.final_evaluation_contract import ConversationFinalEvaluationContractError, EVALUATION_SUITES, REQUIRED_METRIC_BASELINES, load_conversation_evaluation_submissions, load_conversation_final_evaluation_benchmark, load_conversation_final_evaluation_policy, parse_conversation_evaluation_submission
from _final_evaluation_helpers import ROOT, benchmark, policy

def test_policy_and_benchmark_load():
    p=policy(); b=benchmark(); assert set(p.required_suites)==set(EVALUATION_SUITES); assert len(b.cases)==18

def test_policy_has_exact_metric_coverage(): assert {x.metric_id for x in policy().metric_gates}==set(REQUIRED_METRIC_BASELINES)

def test_benchmark_is_synthetic_only(): assert benchmark().synthetic_only is True

def test_case_ids_and_suites_are_unique_and_complete():
    b=benchmark(); assert len({x.case_id for x in b.cases})==len(b.cases); assert {x.suite for x in b.cases}==set(EVALUATION_SUITES)

def test_weakened_policy_is_rejected(tmp_path):
    raw=json.loads((ROOT/"policies/conversation_final_evaluation_policy.json").read_text())
    raw["metric_gates"][0]["threshold"]=0.99; path=tmp_path/"policy.json"; path.write_text(json.dumps(raw))
    with pytest.raises(ConversationFinalEvaluationContractError): load_conversation_final_evaluation_policy(path)

def test_enabled_web_boundary_is_rejected(tmp_path):
    raw=json.loads((ROOT/"policies/conversation_final_evaluation_policy.json").read_text()); raw["boundaries"]["web_access_allowed"]=True
    path=tmp_path/"policy.json"; path.write_text(json.dumps(raw))
    with pytest.raises(ConversationFinalEvaluationContractError): load_conversation_final_evaluation_policy(path)

def test_missing_suite_is_rejected(tmp_path):
    raw=json.loads((ROOT/"benchmarks/phase3/conversation_final_evaluation_v1.json").read_text()); raw["cases"]=raw["cases"][:-1]
    path=tmp_path/"benchmark.json"; path.write_text(json.dumps(raw))
    with pytest.raises(ConversationFinalEvaluationContractError): load_conversation_final_evaluation_benchmark(path,policy=policy())

def test_unknown_content_field_is_rejected():
    value={"case_id":"case-001","actual_outcome":"accepted","signals":[],"violation_codes":[],"observation_digest":"0"*64,"raw_response":"secret"}
    with pytest.raises(ConversationFinalEvaluationContractError): parse_conversation_evaluation_submission(value)

def test_invalid_observation_digest_is_rejected():
    value={"case_id":"case-001","actual_outcome":"accepted","signals":[],"violation_codes":[],"observation_digest":"bad"}
    with pytest.raises(ConversationFinalEvaluationContractError): parse_conversation_evaluation_submission(value)


def test_submission_bundle_must_match_benchmark(tmp_path):
    value={"conversation_final_evaluation_submission_schema_version":1,"benchmark_id":"other-benchmark","test_set_version":"p3.10-v1","submissions":[]}
    path=tmp_path/"submissions.json"; path.write_text(json.dumps(value))
    with pytest.raises(ConversationFinalEvaluationContractError):
        load_conversation_evaluation_submissions(path,benchmark=benchmark())
