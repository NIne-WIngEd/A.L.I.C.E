"""Metadata-only inspection for P3.10 final conversational evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from .final_evaluation import ConversationFinalEvaluationReport, verify_conversation_final_report

@dataclass(frozen=True)
class ConversationFinalEvaluationInspection:
    evaluation_version: str
    benchmark_id: str
    test_set_version: str
    policy_id: str
    case_count: int
    passed_case_count: int
    critical_case_failure_count: int
    failed_case_ids: tuple[str, ...]
    failed_metric_ids: tuple[str, ...]
    passed: bool
    report_digest: str

def inspect_conversation_final_evaluation(report: ConversationFinalEvaluationReport) -> ConversationFinalEvaluationInspection:
    verify_conversation_final_report(report)
    return ConversationFinalEvaluationInspection(report.evaluation_version,report.benchmark_id,report.test_set_version,report.policy_id,report.case_count,report.passed_case_count,report.critical_case_failure_count,tuple(x.case_id for x in report.case_results if not x.passed),tuple(x.metric_id for x in report.metric_results if not x.passed),report.passed,report.report_digest)

def render_conversation_final_evaluation_inspection(value: ConversationFinalEvaluationInspection) -> str:
    return "\n".join((f"evaluation_version={value.evaluation_version}",f"benchmark_id={value.benchmark_id}",f"test_set_version={value.test_set_version}",f"policy_id={value.policy_id}",f"case_count={value.case_count}",f"passed_case_count={value.passed_case_count}",f"critical_case_failure_count={value.critical_case_failure_count}",f"failed_case_ids={','.join(value.failed_case_ids)}",f"failed_metric_ids={','.join(value.failed_metric_ids)}",f"passed={str(value.passed).lower()}",f"report_digest={value.report_digest}"))
