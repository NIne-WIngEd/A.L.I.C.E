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


def row(rid, text, *, family, relations=(), role="NONE", operation="ROLE_SELECT", control="RELATIONAL"):
    return {
        "schema": "alice.eipm.n0.qsre-t2-operator-challenge-row.v0.2",
        "id": rid,
        "split": "challenge",
        "family": family,
        "query_text": text,
        "operator_target": {
            "relation_sequence_id": [REL[x] for x in relations],
            "role_id": ROLE[role],
            "operation_id": OP[operation],
            "control_id": CONTROL[control],
        },
        "training_allowed": False,
        "private_identity_data": False,
    }


ROWS = [
    row("C001","An acoustic trace makes the vibration diagnosis more credible. Which record is supplying that evidential support?",family="endpoint_role",relations=["SUPPORTS"],role="SOURCE"),
    row("C002","The vibration diagnosis becomes more credible because of an acoustic trace. Which record is the supported conclusion?",family="endpoint_role",relations=["SUPPORTS"],role="TARGET"),
    row("C003","A witness photo corroborates the timeline note. Identify the record doing the corroborating.",family="endpoint_role",relations=["SUPPORTS"],role="SOURCE"),
    row("C004","A timeline note is corroborated by a witness photo. Identify the record whose claim is being corroborated.",family="endpoint_role",relations=["SUPPORTS"],role="TARGET"),

    row("C005","A maintenance bulletin fixes a serial number error in an inspection record. Which record carries the correction?",family="endpoint_role",relations=["CORRECTS"],role="SOURCE"),
    row("C006","A maintenance bulletin fixes a serial number error in an inspection record. Which record contained the error?",family="endpoint_role",relations=["CORRECTS"],role="TARGET"),
    row("C007","A signed amendment repairs the date in an earlier notice. Which item is doing the repairing?",family="endpoint_role",relations=["CORRECTS"],role="SOURCE"),
    row("C008","A signed amendment repairs the date in an earlier notice. Which item needed the repair?",family="endpoint_role",relations=["CORRECTS"],role="TARGET"),

    row("C009","Policy Gamma takes effect in place of Policy Beta. Which record is now the governing version?",family="endpoint_role",relations=["SUPERSEDES"],role="SOURCE"),
    row("C010","Policy Gamma takes effect in place of Policy Beta. Which record is the displaced version?",family="endpoint_role",relations=["SUPERSEDES"],role="TARGET"),
    row("C011","The final release replaces the release candidate as authoritative. Which record takes precedence?",family="endpoint_role",relations=["SUPERSEDES"],role="SOURCE"),
    row("C012","The final release replaces the release candidate as authoritative. Which record loses precedence?",family="endpoint_role",relations=["SUPERSEDES"],role="TARGET"),

    row("C013","A dashboard snapshot was generated from a telemetry archive. Which record is the generated artifact?",family="endpoint_role",relations=["DERIVED_FROM"],role="SOURCE"),
    row("C014","A dashboard snapshot was generated from a telemetry archive. Which record is the underlying source?",family="endpoint_role",relations=["DERIVED_FROM"],role="TARGET"),
    row("C015","A normalized table came from the original survey dump. Which record is the derived table?",family="endpoint_role",relations=["DERIVED_FROM"],role="SOURCE"),
    row("C016","A normalized table came from the original survey dump. Which record supplied its source material?",family="endpoint_role",relations=["DERIVED_FROM"],role="TARGET"),

    row("C017","A coolant blockage produced the overheating event. Which record represents the initiating condition?",family="endpoint_role",relations=["CAUSES"],role="SOURCE"),
    row("C018","A coolant blockage produced the overheating event. Which record represents the resulting event?",family="endpoint_role",relations=["CAUSES"],role="TARGET"),
    row("C019","A credential-expiry event led to the failed deployment. Which record is the causal antecedent?",family="endpoint_role",relations=["CAUSES"],role="SOURCE"),
    row("C020","A credential-expiry event led to the failed deployment. Which record is the consequence?",family="endpoint_role",relations=["CAUSES"],role="TARGET"),

    row("C021","Build 43 follows Build 42 in the release history. Which record is the successor?",family="endpoint_role",relations=["TEMPORAL_SUCCESSOR"],role="SOURCE"),
    row("C022","Build 43 follows Build 42 in the release history. Which record is the predecessor?",family="endpoint_role",relations=["TEMPORAL_SUCCESSOR"],role="TARGET"),
    row("C023","The review addendum appears after the original review in the chronology. Which record is later?",family="endpoint_role",relations=["TEMPORAL_SUCCESSOR"],role="SOURCE"),
    row("C024","The review addendum appears after the original review in the chronology. Which record is earlier?",family="endpoint_role",relations=["TEMPORAL_SUCCESSOR"],role="TARGET"),

    row("C025","Start at the synthesized report. It came from a raw capture, and that raw capture was later corrected by an amendment. Which record is reached after both links?",family="ordered_path",relations=["DERIVED_FROM","CORRECTS"],role="TARGET",operation="PATH_FOLLOW"),
    row("C026","Begin with the retired standard. Move to the standard that replaced it, then to the revision that followed that replacement in time. What is the endpoint?",family="ordered_path",relations=["SUPERSEDES","TEMPORAL_SUCCESSOR"],role="TARGET",operation="PATH_FOLLOW"),
    row("C027","A baseline observation supports a diagnosis, and that diagnosis produces a remediation action. Starting at the observation, where does the two-step chain end?",family="ordered_path",relations=["SUPPORTS","CAUSES"],role="TARGET",operation="PATH_FOLLOW"),
    row("C028","A fault condition causes an incident, and an official erratum repairs the incident record. Follow those links in sequence. What is reached last?",family="ordered_path",relations=["CAUSES","CORRECTS"],role="TARGET",operation="PATH_FOLLOW"),
    row("C029","A draft is replaced by a signed release; a deployment manifest is then generated from that release. Trace the two relationships from the draft.",family="ordered_path",relations=["SUPERSEDES","DERIVED_FROM"],role="TARGET",operation="PATH_FOLLOW"),
    row("C030","A source memo is used to produce a summary, and that summary supports a recommendation. Starting from the summary, use the provenance link and then the evidence link.",family="ordered_path",relations=["DERIVED_FROM","SUPPORTS"],role="TARGET",operation="PATH_FOLLOW"),
    row("C031","From an early report, take the chronology link to its successor, then use the successor as the evidence source for a conclusion. Which terminal record is selected?",family="ordered_path",relations=["TEMPORAL_SUCCESSOR","SUPPORTS"],role="TARGET",operation="PATH_FOLLOW"),
    row("C032","A correction note repairs an old record, and the corrected record is later replaced by a new authority. Execute those two links in order.",family="ordered_path",relations=["CORRECTS","SUPERSEDES"],role="TARGET",operation="PATH_FOLLOW"),

    row("C033","Four independent measurements each back the same conclusion. Which records are the evidence providers?",family="multi_support_aggregate",relations=["SUPPORTS"],role="SOURCE",operation="AGGREGATE"),
    row("C034","Several amendments each fix separate inaccurate notes. Which records contain the fixes?",family="multi_support_aggregate",relations=["CORRECTS"],role="SOURCE",operation="AGGREGATE"),
    row("C035","Multiple processed artifacts were each produced from raw inputs. Which records are the produced artifacts?",family="multi_support_aggregate",relations=["DERIVED_FROM"],role="SOURCE",operation="AGGREGATE"),
    row("C036","A single upstream event led to several distinct failures. Which records are the consequences?",family="multi_support_aggregate",relations=["CAUSES"],role="TARGET",operation="AGGREGATE"),
    row("C037","Several current releases each replace an older version on different branches. Which records are the replacements?",family="multi_support_aggregate",relations=["SUPERSEDES"],role="SOURCE",operation="AGGREGATE"),
    row("C038","Several predecessor-successor links are simultaneously relevant. Which records are the predecessors?",family="multi_support_aggregate",relations=["TEMPORAL_SUCCESSOR"],role="TARGET",operation="AGGREGATE"),

    row("C039","Two reports back the same claim. One comes from a calibrated logger with full provenance; the other comes from a sensor under investigation. Which supported claim should govern?",family="reliability_arbitration",relations=["SUPPORTS"],role="TARGET",operation="RELIABILITY"),
    row("C040","Two amendments disagree. One is signed and audited; the other is an unsigned scratch note. Which correction record should control?",family="reliability_arbitration",relations=["CORRECTS"],role="SOURCE",operation="RELIABILITY"),
    row("C041","Two raw records could be the provenance base for a summary. One has matching checksums and complete chain-of-custody; the other's origin is uncertain. Which source should be used?",family="reliability_arbitration",relations=["DERIVED_FROM"],role="TARGET",operation="RELIABILITY"),
    row("C042","Two possible causes are linked to an outage. One is reproduced in logs; the other is a single unverified memory. Which cause should carry the decision?",family="reliability_arbitration",relations=["CAUSES"],role="SOURCE",operation="RELIABILITY"),
    row("C043","Two files claim to replace the same procedure. One is the signed release; one is an abandoned draft. Which replacing record should govern?",family="reliability_arbitration",relations=["SUPERSEDES"],role="SOURCE",operation="RELIABILITY"),
    row("C044","Two chronology records disagree about the successor. One comes from the authoritative event stream; one comes from a partial cache. Which successor should be used?",family="reliability_arbitration",relations=["TEMPORAL_SUCCESSOR"],role="SOURCE",operation="RELIABILITY"),

    row("C045","A conclusion was supported once before recalibration and again after recalibration. For the post-recalibration state, which supported conclusion is relevant?",family="temporal_arbitration",relations=["SUPPORTS"],role="TARGET",operation="TEMPORAL"),
    row("C046","The same value was corrected before the audit and corrected again after the audit. Which correction applies to the post-audit state?",family="temporal_arbitration",relations=["CORRECTS"],role="SOURCE",operation="TEMPORAL"),
    row("C047","A procedure was replaced before migration and replaced again after migration completed. Which replacement represents the post-migration state?",family="temporal_arbitration",relations=["SUPERSEDES"],role="SOURCE",operation="TEMPORAL"),
    row("C048","A summary was generated from one source before cleanup and from another after cleanup. Which source belongs to the post-cleanup summary?",family="temporal_arbitration",relations=["DERIVED_FROM"],role="TARGET",operation="TEMPORAL"),
    row("C049","An early fault caused one failure; after a patch, a different fault caused another. For the post-patch state, which cause applies?",family="temporal_arbitration",relations=["CAUSES"],role="SOURCE",operation="TEMPORAL"),
    row("C050","The history has one successor before the cutover and a different successor after the cutover. Which successor belongs to the post-cutover timeline?",family="temporal_arbitration",relations=["TEMPORAL_SUCCESSOR"],role="SOURCE",operation="TEMPORAL"),

    row("C051","In the phrase 'Only Devin submitted the form,' what does 'only' contribute to the meaning?",family="fallback",control="FALLBACK"),
    row("C052","A user says, 'Would you mind lowering the volume?' What action is most likely intended?",family="fallback",control="FALLBACK"),
    row("C053","A database row says pending while the signed approval letter says approved. What inconsistency should be surfaced?",family="fallback",control="FALLBACK"),
    row("C054","A smudged identifier might read O51 or 051. How should the answer represent uncertainty?",family="fallback",control="FALLBACK"),
    row("C055","In a finance article, what meaning of 'yield' is intended by the surrounding sentence?",family="fallback",control="FALLBACK"),

    row("C056","Several records concern the same event, but the request never identifies whether evidence, correction, provenance, causality, chronology, or replacement is the relevant connection. Which relational operator is licensed?",family="defer",control="DEFER"),
    row("C057","One note points to another, but the text does not establish the direction or type of the relationship. What relation can safely be executed?",family="defer",control="DEFER"),
    row("C058","The request asks for 'the linked record' while multiple relation families are present. Which one should the system commit to?",family="defer",control="DEFER"),
    row("C059","The two documents are associated, but neither relation type nor endpoint direction is specified. What relational action is justified?",family="defer",control="DEFER"),
    row("C060","A connection probably matters, yet the wording leaves causal, evidential, temporal, corrective, provenance, and replacement interpretations all plausible. Which operator should be selected?",family="defer",control="DEFER"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if len(ROWS)!=60:
        raise SystemExit(f"challenge row count drift: {len(ROWS)}")
    ids=[r["id"] for r in ROWS]
    if len(ids)!=len(set(ids)):
        raise SystemExit("duplicate challenge ids")
    if any(r["training_allowed"] is not False or r["private_identity_data"] is not False for r in ROWS):
        raise SystemExit("challenge governance drift")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in ROWS),encoding="utf-8")
    print("status=PASS_QSRE_T2_V02_LOCKED_OPERATOR_CHALLENGE")
    print(f"rows={len(ROWS)}")
    print(f"sha256={sha256(output)}")


if __name__=="__main__":
    main()
