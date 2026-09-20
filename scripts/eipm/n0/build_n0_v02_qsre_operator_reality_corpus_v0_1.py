from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REL = {
    "SUPPORTS": 0,
    "CORRECTS": 1,
    "SUPERSEDES": 2,
    "DERIVED_FROM": 3,
    "CAUSES": 4,
    "TEMPORAL_SUCCESSOR": 5,
}
CONTROL = {"FALLBACK": 0, "RELATIONAL": 1, "DEFER": 2}
OP = {"ROLE_SELECT": 0, "PATH_FOLLOW": 1, "AGGREGATE": 2, "RELIABILITY": 3, "TEMPORAL": 4}
ROLE = {"SOURCE": 0, "TARGET": 1, "NONE": 3}


def row(
    rid: str,
    query: str,
    *,
    family: str,
    relations: list[str] | tuple[str, ...] = (),
    role: str = "NONE",
    operation: str = "ROLE_SELECT",
    control: str = "RELATIONAL",
    paraphrase_group: str | None = None,
    contrast_group: str | None = None,
) -> dict:
    return {
        "schema": "alice.eipm.n0.qsre-operator-reality-row.v0.1",
        "id": rid,
        "split": "audit",
        "family": family,
        "query_text": query,
        "operator_target": {
            "relation_sequence_id": [REL[x] for x in relations],
            "role_id": ROLE[role],
            "operation_id": OP[operation],
            "control_id": CONTROL[control],
        },
        "paraphrase_group": paraphrase_group,
        "contrast_group": contrast_group,
        "public_n0_only": True,
        "private_identity_data": False,
        "training_allowed": False,
    }


ROWS = [
    # Endpoint-role and relation semantics. These are deliberately phrased as
    # ordinary questions rather than the T2 synthetic instruction grammar.
    row("R001", "Mei's calibration log strengthens Jordan's claim that the sensor drifted. Which record is supplying the evidence?", family="endpoint_role", relations=["SUPPORTS"], role="SOURCE", paraphrase_group="p_support_src", contrast_group="c_support_role"),
    row("R002", "Jordan's claim about sensor drift is backed by Mei's calibration log. Which record is the claim receiving that backing?", family="endpoint_role", relations=["SUPPORTS"], role="TARGET", paraphrase_group="p_support_tgt", contrast_group="c_support_role"),
    row("R003", "One note gives evidence for the conclusion in another note. I want the evidence-giving side of that connection.", family="endpoint_role", relations=["SUPPORTS"], role="SOURCE", paraphrase_group="p_support_src"),
    row("R004", "A conclusion is strengthened by a separate test record. Which side of the link contains the conclusion being strengthened?", family="endpoint_role", relations=["SUPPORTS"], role="TARGET", paraphrase_group="p_support_tgt"),

    row("R005", "The erratum fixes the voltage printed in the original lab note. Which record contains the fix?", family="endpoint_role", relations=["CORRECTS"], role="SOURCE", paraphrase_group="p_correct_src", contrast_group="c_correct_role"),
    row("R006", "The original lab note had the wrong voltage and an erratum repaired it. Which record is the one that needed correction?", family="endpoint_role", relations=["CORRECTS"], role="TARGET", paraphrase_group="p_correct_tgt", contrast_group="c_correct_role"),
    row("R007", "A later annotation repairs a mistaken date in an earlier record. Identify the annotation doing the repairing.", family="endpoint_role", relations=["CORRECTS"], role="SOURCE", paraphrase_group="p_correct_src"),
    row("R008", "A later annotation repairs a mistaken date in an earlier record. Identify the earlier record whose date was wrong.", family="endpoint_role", relations=["CORRECTS"], role="TARGET", paraphrase_group="p_correct_tgt"),

    row("R009", "The June operating procedure replaced the March procedure as the version people should use. Which record took over?", family="endpoint_role", relations=["SUPERSEDES"], role="SOURCE", paraphrase_group="p_super_src", contrast_group="c_super_role"),
    row("R010", "The March procedure stopped being authoritative when the June procedure took over. Which record lost precedence?", family="endpoint_role", relations=["SUPERSEDES"], role="TARGET", paraphrase_group="p_super_tgt", contrast_group="c_super_role"),
    row("R011", "A new policy displaced the older policy without deleting its history. Which record is the replacement?", family="endpoint_role", relations=["SUPERSEDES"], role="SOURCE", paraphrase_group="p_super_src"),
    row("R012", "A new policy displaced the older policy without deleting its history. Which record is the one that was displaced?", family="endpoint_role", relations=["SUPERSEDES"], role="TARGET", paraphrase_group="p_super_tgt"),

    row("R013", "The weekly summary was assembled from the raw survey export. Which record is the derived summary?", family="endpoint_role", relations=["DERIVED_FROM"], role="SOURCE", paraphrase_group="p_derived_src", contrast_group="c_derived_role"),
    row("R014", "The weekly summary was assembled from the raw survey export. Which record is the underlying source material?", family="endpoint_role", relations=["DERIVED_FROM"], role="TARGET", paraphrase_group="p_derived_tgt", contrast_group="c_derived_role"),
    row("R015", "A cleaned dataset was produced from the original instrument dump. Point to the produced dataset.", family="endpoint_role", relations=["DERIVED_FROM"], role="SOURCE", paraphrase_group="p_derived_src"),
    row("R016", "A cleaned dataset was produced from the original instrument dump. Point to the instrument dump it came from.", family="endpoint_role", relations=["DERIVED_FROM"], role="TARGET", paraphrase_group="p_derived_tgt"),

    row("R017", "A leaking seal made the pressure test fail. Which record represents the cause rather than the failure?", family="endpoint_role", relations=["CAUSES"], role="SOURCE", paraphrase_group="p_cause_src", contrast_group="c_cause_role"),
    row("R018", "A leaking seal made the pressure test fail. Which record represents the resulting failure?", family="endpoint_role", relations=["CAUSES"], role="TARGET", paraphrase_group="p_cause_tgt", contrast_group="c_cause_role"),
    row("R019", "The scheduler outage led to the missed nightly backup. Which side of the causal link is the outage?", family="endpoint_role", relations=["CAUSES"], role="SOURCE", paraphrase_group="p_cause_src"),
    row("R020", "The scheduler outage led to the missed nightly backup. Which side is the consequence?", family="endpoint_role", relations=["CAUSES"], role="TARGET", paraphrase_group="p_cause_tgt"),

    row("R021", "Revision C came after revision B in the documented sequence. Which record is the later successor?", family="endpoint_role", relations=["TEMPORAL_SUCCESSOR"], role="SOURCE", paraphrase_group="p_time_src", contrast_group="c_time_role"),
    row("R022", "Revision C came after revision B in the documented sequence. Which record is the earlier predecessor?", family="endpoint_role", relations=["TEMPORAL_SUCCESSOR"], role="TARGET", paraphrase_group="p_time_tgt", contrast_group="c_time_role"),
    row("R023", "The incident update follows the initial report chronologically. Which record is the update that came afterward?", family="endpoint_role", relations=["TEMPORAL_SUCCESSOR"], role="SOURCE", paraphrase_group="p_time_src"),
    row("R024", "The incident update follows the initial report chronologically. Which record is the report that came first?", family="endpoint_role", relations=["TEMPORAL_SUCCESSOR"], role="TARGET", paraphrase_group="p_time_tgt"),

    # Ordered composition. Relation order is part of the target and wording is
    # varied so that one literal 'first/second' template is not required.
    row("R025", "Begin with the incident summary. It was built from a source note, and that source note was later corrected by an erratum. Where do you end if you trace those two connections in that order?", family="ordered_path", relations=["DERIVED_FROM", "CORRECTS"], role="TARGET", operation="PATH_FOLLOW"),
    row("R026", "From the old policy, move to the policy that replaced it; from there, move to the record that came after that replacement in time. What is the terminal record?", family="ordered_path", relations=["SUPERSEDES", "TEMPORAL_SUCCESSOR"], role="TARGET", operation="PATH_FOLLOW"),
    row("R027", "The analysis note comes from a raw run, and that raw run supports a later finding. Starting at the analysis note, follow the provenance connection and then the evidence connection.", family="ordered_path", relations=["DERIVED_FROM", "SUPPORTS"], role="TARGET", operation="PATH_FOLLOW"),
    row("R028", "Start with the faulty configuration. Its defect caused an outage, and a postmortem later corrected the outage record. Which record sits at the end of that chain?", family="ordered_path", relations=["CAUSES", "CORRECTS"], role="TARGET", operation="PATH_FOLLOW"),
    row("R029", "A draft is replaced by a release, then that release becomes the basis for a generated deployment record. Trace that sequence from the draft to its endpoint.", family="ordered_path", relations=["SUPERSEDES", "DERIVED_FROM"], role="TARGET", operation="PATH_FOLLOW"),
    row("R030", "The baseline measurement supports a diagnosis; that diagnosis then leads to a maintenance action. Starting from the baseline, which record is reached after both links?", family="ordered_path", relations=["SUPPORTS", "CAUSES"], role="TARGET", operation="PATH_FOLLOW"),

    # Multi-support/plurality without the synthetic 'return all records' wording.
    row("R031", "Three independent test notes each strengthen the same conclusion. I need the evidence-giving records, not the conclusion.", family="multi_support_aggregate", relations=["SUPPORTS"], role="SOURCE", operation="AGGREGATE"),
    row("R032", "Several errata each repair a different mistaken record. Which records contain the corrections themselves?", family="multi_support_aggregate", relations=["CORRECTS"], role="SOURCE", operation="AGGREGATE"),
    row("R033", "There are multiple summaries, each produced from a different raw source. Which records are the summaries rather than the sources?", family="multi_support_aggregate", relations=["DERIVED_FROM"], role="SOURCE", operation="AGGREGATE"),
    row("R034", "More than one failure was triggered by the same upstream event. Which records are the resulting failures?", family="ambiguity_plurality", relations=["CAUSES"], role="TARGET", operation="AGGREGATE"),
    row("R035", "Several newer revisions each displace an older version in separate branches. Which records are on the replacing side of those links?", family="multi_support_aggregate", relations=["SUPERSEDES"], role="SOURCE", operation="AGGREGATE"),
    row("R036", "The timeline contains several predecessor-successor links that are all relevant. Which records are the earlier members of those links?", family="ambiguity_plurality", relations=["TEMPORAL_SUCCESSOR"], role="TARGET", operation="AGGREGATE"),

    # Reliability arbitration. The query implies evidential quality without
    # relying on the exact T2 cue phrase 'most reliable'.
    row("R037", "Two supporting reports disagree. One comes from a calibrated instrument with complete logs; the other comes from a sensor already flagged for drift. Which supported conclusion should carry the decision?", family="reliability_arbitration", relations=["SUPPORTS"], role="TARGET", operation="RELIABILITY"),
    row("R038", "Two corrections conflict. One is an audited erratum signed by the maintainer; the other is an unverified margin note. Which correction record should control the result?", family="reliability_arbitration", relations=["CORRECTS"], role="SOURCE", operation="RELIABILITY"),
    row("R039", "Two source records could explain where the summary came from. One has intact provenance and matching checksums; the other's origin is uncertain. Which source should anchor the derivation?", family="reliability_arbitration", relations=["DERIVED_FROM"], role="TARGET", operation="RELIABILITY"),
    row("R040", "Two candidate causes are linked to the same failure. One is backed by reproduced logs and the other by a single unverified recollection. Which cause should the system rely on?", family="reliability_arbitration", relations=["CAUSES"], role="SOURCE", operation="RELIABILITY"),
    row("R041", "Two records each claim to replace the same old procedure. One is the signed release and the other is an abandoned draft. Which replacing record should govern?", family="reliability_arbitration", relations=["SUPERSEDES"], role="SOURCE", operation="RELIABILITY"),
    row("R042", "Two timeline links disagree about what came next. One comes from the authoritative event log and one from an incomplete cache. Which successor should be used?", family="reliability_arbitration", relations=["TEMPORAL_SUCCESSOR"], role="SOURCE", operation="RELIABILITY"),

    # Temporal arbitration. Avoid the exact generated 'latest/newest' formulas.
    row("R043", "A claim was backed twice: once before the rerun and once after the rerun finished. For the state after the rerun, which supported record is relevant?", family="temporal_arbitration", relations=["SUPPORTS"], role="TARGET", operation="TEMPORAL"),
    row("R044", "The same mistake was corrected in April and corrected again after the June review. Which correction governs the post-review record?", family="temporal_arbitration", relations=["CORRECTS"], role="SOURCE", operation="TEMPORAL"),
    row("R045", "A procedure was replaced during spring and then replaced again after the fall audit. Which replacing record reflects the post-audit state?", family="temporal_arbitration", relations=["SUPERSEDES"], role="SOURCE", operation="TEMPORAL"),
    row("R046", "The summary was regenerated from one source before cleanup and from another source after cleanup. Which source belongs to the regenerated post-cleanup version?", family="temporal_arbitration", relations=["DERIVED_FROM"], role="TARGET", operation="TEMPORAL"),
    row("R047", "An initial fault caused an early failure, and a later fault caused a second failure after the patch. For the post-patch state, which cause should be selected?", family="temporal_arbitration", relations=["CAUSES"], role="SOURCE", operation="TEMPORAL"),
    row("R048", "The history records one successor before the migration and another successor after migration completed. Which successor reflects the post-migration timeline?", family="temporal_arbitration", relations=["TEMPORAL_SUCCESSOR"], role="SOURCE", operation="TEMPORAL"),

    # Realistic non-relational and underspecified queries. They do not announce
    # the control label with words like 'fallback' or 'defer'.
    row("R049", "In the sentence 'Again, the printer jammed during the test,' what does 'again' normally presuppose?", family="fallback_defer", control="FALLBACK"),
    row("R050", "A passenger asks a driver, 'Could you stop near the library?' What speech act is most likely intended?", family="fallback_defer", control="FALLBACK"),
    row("R051", "The report says the package has not shipped, while a structured field says delivered. What inconsistency should be noted?", family="fallback_defer", control="FALLBACK"),
    row("R052", "A witness remembers either 8KJ or BKJ from a plate seen at night. How should confidence be expressed?", family="fallback_defer", control="FALLBACK"),
    row("R053", "These two records appear connected somehow, but the request does not say whether the connection is causal, evidential, temporal, corrective, or provenance-based. What relation should be used?", family="fallback_defer", control="DEFER"),
    row("R054", "One note mentions the other, but nothing here says whether it supports it, corrects it, replaces it, or merely cites it. Which relational operator applies?", family="fallback_defer", control="DEFER"),
    row("R055", "A later document and an earlier document refer to the same incident, but no update, correction, or succession relation is stated. How should their relationship be interpreted?", family="fallback_defer", control="DEFER"),
    row("R056", "The query asks for 'the related record' without specifying what kind of relationship matters, and several different links are present. Which one should be executed?", family="fallback_defer", control="DEFER"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate audit ids")
    if len(rows) != 56:
        raise SystemExit(f"expected 56 curated audit rows, got {len(rows)}")
    if any(r["private_identity_data"] is not False for r in rows):
        raise SystemExit("private identity data present")
    if any(r["training_allowed"] is not False for r in rows):
        raise SystemExit("audit rows must not be training data")
    controls = {r["operator_target"]["control_id"] for r in rows}
    if controls != {0, 1, 2}:
        raise SystemExit("control coverage drift")
    operations = {
        r["operator_target"]["operation_id"]
        for r in rows
        if r["operator_target"]["control_id"] == CONTROL["RELATIONAL"]
    }
    if operations != {0, 1, 2, 3, 4}:
        raise SystemExit("operation coverage drift")
    relations = {
        relation
        for r in rows
        for relation in r["operator_target"]["relation_sequence_id"]
    }
    if relations != set(REL.values()):
        raise SystemExit("relation coverage drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(ROWS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ROWS),
        encoding="utf-8",
    )
    print(f"status=PASS_QSRE_OPERATOR_REALITY_CORPUS")
    print(f"rows={len(ROWS)}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
