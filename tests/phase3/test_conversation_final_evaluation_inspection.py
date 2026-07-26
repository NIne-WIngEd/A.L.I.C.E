from __future__ import annotations
from alice_conversation.final_evaluation import run_conversation_final_evaluation
from alice_conversation.final_evaluation_inspection import inspect_conversation_final_evaluation, render_conversation_final_evaluation_inspection
from _final_evaluation_helpers import benchmark, passing_submissions, policy

def test_metadata_only_inspection():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    inspection=inspect_conversation_final_evaluation(report); text=render_conversation_final_evaluation_inspection(inspection)
    assert inspection.passed is True and "passed=true" in text and "prompt" not in text.lower() and "response" not in text.lower()

def test_inspection_lists_no_failures_for_passing_report():
    report=run_conversation_final_evaluation(submissions=passing_submissions(),benchmark=benchmark(),policy=policy())
    inspection=inspect_conversation_final_evaluation(report); assert inspection.failed_case_ids==() and inspection.failed_metric_ids==()
