from __future__ import annotations
from dataclasses import replace
import pytest
from alice_conversation.final_evaluation import ConversationFinalEvaluationError, conversation_final_report_digest, run_conversation_final_evaluation, verify_conversation_final_report
from _final_evaluation_helpers import benchmark, passing_submissions, policy, replace_submission

def test_full_synthetic_evaluation_passes():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    assert report.passed and report.case_count==18 and report.passed_case_count==18

def test_report_digest_is_deterministic():
    first=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    second=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    assert first.report_digest==second.report_digest

def test_missing_submission_fails_closed():
    with pytest.raises(ConversationFinalEvaluationError): run_conversation_final_evaluation(submissions=passing_submissions()[:-1],benchmark=benchmark(),policy=policy())

def test_duplicate_submission_fails_closed():
    items=passing_submissions();
    with pytest.raises(ConversationFinalEvaluationError): run_conversation_final_evaluation(submissions=items+(items[0],),benchmark=benchmark(),policy=policy())

def test_wrong_outcome_fails_case_and_report():
    items=replace_submission(passing_submissions(),0,actual_outcome="rejected")
    report=run_conversation_final_evaluation(submissions=items,benchmark=benchmark(),policy=policy())
    assert not report.passed and report.passed_case_count==17 and report.critical_case_failure_count==1

def test_missing_signal_fails_case():
    items=replace_submission(passing_submissions(),0,signals=())
    report=run_conversation_final_evaluation(submissions=items,benchmark=benchmark(),policy=policy())
    assert report.case_results[0].missing_signals==("constitutional_contract_applied",)

def test_violation_fails_critical_metrics():
    items=replace_submission(passing_submissions(),15,violation_codes=("private_content_exposed",))
    report=run_conversation_final_evaluation(submissions=items,benchmark=benchmark(),policy=policy())
    metric={x.metric_id:x for x in report.metric_results}["privacy_boundary_violation_count"]
    assert metric.value==1.0 and not metric.passed and not report.passed

def test_tampered_digest_is_rejected():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    with pytest.raises(ConversationFinalEvaluationError): verify_conversation_final_report(replace(report,report_digest="0"*64))

def test_inconsistent_pass_decision_is_rejected():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    tampered=replace(report,passed=False,report_digest="")
    tampered=replace(tampered,report_digest=conversation_final_report_digest(tampered))
    with pytest.raises(ConversationFinalEvaluationError): verify_conversation_final_report(tampered)


def test_inconsistent_case_count_is_rejected():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    tampered=replace(report,case_count=report.case_count+1,report_digest="")
    tampered=replace(tampered,report_digest=conversation_final_report_digest(tampered))
    with pytest.raises(ConversationFinalEvaluationError): verify_conversation_final_report(tampered)

def test_recomputed_metric_tamper_is_rejected():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    metrics=list(report.metric_results); metrics[0]=replace(metrics[0],value=0.5,passed=False)
    tampered=replace(report,metric_results=tuple(metrics),passed=False,report_digest="")
    tampered=replace(tampered,report_digest=conversation_final_report_digest(tampered))
    with pytest.raises(ConversationFinalEvaluationError): verify_conversation_final_report(tampered)
