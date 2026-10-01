#!/usr/bin/env python3
"""Source-grounded fictional longitudinal MFM training histories.

This is a deliberately varied, single-author *training* generator. All names,
statements, outcomes and rights are fictional. It does not make independent
gold, evaluate generalization or claim that an owner said these words.
Development histories live in a different existing generator and host family.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
)
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16,
    learning_example_v16_from_record, supervised_output_record_v16,
)
from cognitive_kernel.formation_semantics_v16 import (
    EpisodeSemantics, FormationContextV16, FormationProposalV16,
    MemoryProposalBundleV16, RegisteredFormationTarget,
    RegisteredSourceSensitivity, validate_formation_grounding_v16,
)
from cognitive_kernel.canonical import canonical_json_bytes


GENERATOR_FAMILY = "codex-fictional-longitudinal-train-20261001"
AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
HOST_COUNT = 32
T1 = "2026-01-11T09:00:00.000000Z"
T2 = "2026-02-12T14:00:00.000000Z"
T3 = "2026-03-16T18:00:00.000000Z"
T4 = "2026-04-22T10:00:00.000000Z"
T5 = "2026-05-29T16:00:00.000000Z"

# A host has one coherent chronology across thirteen different formation
# decisions. Cartesian variation is explicit and frozen; it is not evidence of
# thirteen independent authors or thirty-two real people.
CRAFTS = ("atlas", "ceramics", "orchard", "theatre", "harbor", "library", "garden", "radio")
SITES = ("east", "west", "north", "south")
FRIENDS = ("Neri", "Tavi", "Sora", "Ilan", "Edda", "Omi", "Pera", "Venn")
PEOPLE = ("Marta", "Bo", "Leena", "Kavi", "Rae", "Nolan", "Dina", "Tomo")
SCENARIOS = (
    "norm-agreed", "norm-unsettled", "episode-mission", "plan-outcome",
    "procedure-revision", "source-person", "revocation-deletion",
    "record-correction", "owner-deletion", "sensitive-boundary",
    "assistant-self", "hypothetical-code", "conflicting-outside",
)


def src(text: str, when: str, role: str = "authenticated_owner_statement",
        subject: str = "owner", speaker: str = "owner", minimum: str = "private",
        modality: str = "text") -> dict:
    return locals()


def p(kind: str, domain: str, subject: str, value: str,
      quotes: dict[str, str], *, epistemic: str = "owner_statement",
      when: str | None = T3, until: str | None = None,
      sensitivity: str | None = None, counterpart: str | None = None,
      mission: tuple[str, ...] = (), workspace: tuple[str, ...] = (),
      episode: tuple | None = None, target_refs: tuple[str, ...] = (),
      contradicts: tuple[str, ...] = (), scope: str = "memory") -> dict:
    return locals()


def spec(host_i: int, scenario: str) -> dict:
    craft, site = CRAFTS[host_i // 4], SITES[host_i % 4]
    friend, person = FRIENDS[host_i // 4], PEOPLE[(host_i + 3) % 8]
    host = f"long-{craft}-{site}"
    case_id = f"{host}-{scenario}"
    project = f"{site} {craft} catalogue"
    place = f"{site} {craft} studio"
    folder = f"{site}-{craft}-workspace"
    mission = f"{site}-{craft}-catalogue"
    neg = {k: "negative" for k in FULL_ROLE_DIMENSIONS}
    sources, proposals, dispositions, targets, labels = {}, [], [], [], neg.copy()
    pair = "norm-decisive-later" if scenario.startswith("norm-") else None
    rationale = ""

    if scenario.startswith("norm-"):
        sources["outburst"] = src(
            f"On January 11 I snapped at {friend} about my {project}: 'never show it to anyone'. "
            "I was angry; I did not mean that as a standing rule.", T1)
        if scenario == "norm-agreed":
            quote = f"{friend} and I agreed they will ask before showing my {project} to others"
            sources["later"] = src(f"On February 12, {quote}. We both said the rule applies from now on.", T2)
            targets = [(friend.lower(), "entity", ("later",))]
            proposals = [p("relationship_norm", "relationship", "owner",
                           f"The owner and {friend} agreed that {friend} asks before sharing the {project}.",
                           {"later": quote}, when=T2, counterpart=friend.lower(), scope="sharing_norm")]
            dispositions = [("sharing_norm", "propose", ("later",), ())]
            labels["relationship"] = "present"
        else:
            sources["later"] = src(
                f"On February 12, {friend} and I had not settled whether they may show my "
                f"{project} to others. I asked us to wait for a decision.", T2)
            dispositions = [("sharing_norm", "defer", ("later",), ())]
        dispositions += [("angry_outburst_as_norm", "abstain", ("outburst", "later"), ())]
        labels["abstention"] = "present"
        rationale = "The later owner's statement determines whether the outburst became a durable norm."
    elif scenario == "episode-mission":
        sources["plan"] = src(f"On January 11 I planned to visit the {place} with {friend} for the {project}.", T1)
        sources["visit"] = src(
            f"At 14:00 on February 12, {friend} and I entered the {place}, checked the intake desk, "
            f"then used the archive room. We annotated the {project} in {folder}.", T2)
        targets = [("owner", "entity", ("visit",)), (friend.lower(), "entity", ("visit",)),
                   ("intake-desk", "scene", ("visit",)), ("archive-room", "scene", ("visit",)),
                   (mission, "mission", ("plan", "visit")),
                   (folder, "workspace", ("visit",))]
        quote = f"{friend} and I entered the {place}, checked the intake desk, then used the archive room"
        proposals = [p("episode", "host", "owner",
                       f"The owner and {friend} visited the {place}, intake desk and archive room on February 12.",
                       {"visit": quote}, when=T2,
                       episode=(T2, None, ("owner", friend.lower()), ("intake-desk", "archive-room")),
                       mission=(mission,), workspace=(folder,), scope="visit"),
                     p("mission", "mission", "owner", f"The owner annotated the {project} in {folder}.",
                       {"visit": f"We annotated the {project} in {folder}"}, when=T2,
                       mission=(mission,), workspace=(folder,), scope="catalogue_progress")]
        dispositions = [("visit", "propose", ("visit",), ()),
                        ("catalogue_progress", "propose", ("visit",), ())]
        labels.update(episode="present", mission="present", workspace="present")
        rationale = "The planned visit is separate from the observed scene sequence and mission work."
    elif scenario == "plan-outcome":
        sources["plan"] = src(f"I plan to publish the {project} from {folder} in April.", T1)
        sources["run"] = src(
            json.dumps({"artifact": project, "copied": 37, "issue": "duplicate-index",
                        "status": "aborted", "deployed": False}, sort_keys=True), T4,
            role="tool_or_action_observation", speaker="catalogue-tool",
            minimum="internal", modality="structured")
        sources["revision"] = src(
            f"I still want to publish the {project}; I will repair duplicate keys in {folder} "
            "before a new attempt. The aborted run did not finish deployment.", T4)
        targets = [(mission, "mission", ("plan", "run", "revision")),
                   (folder, "workspace", ("plan", "run", "revision"))]
        proposals = [p("goal", "mission", "owner", f"The owner plans to publish the {project}.",
                       {"plan": f"I plan to publish the {project}"}, when=T1,
                       mission=(mission,), workspace=(folder,), scope="publication_goal"),
                     p("outcome", "mission", "owner",
                       f"The observed {project} run copied 37 items, aborted on duplicate keys and did not deploy.",
                       {"run": sources["run"]["text"]}, epistemic="observation", when=T4,
                       mission=(mission,), workspace=(folder,), scope="aborted_run"),
                     p("goal", "mission", "owner",
                       f"The owner intends to repair duplicate keys in {folder} before retrying.",
                       {"revision": f"I will repair duplicate keys in {folder} before a new attempt"},
                       when=T4, mission=(mission,), workspace=(folder,), scope="revised_goal")]
        dispositions = [("publication_goal", "propose", ("plan",), ()),
                        ("aborted_run", "propose", ("run",), ()),
                        ("revised_goal", "propose", ("revision",), ())]
        labels.update(mission="present", workspace="present", outcome="present")
        rationale = "Plans, tool-observed failure and future revision must not collapse into completed publication."
    elif scenario == "procedure-revision":
        sources["attempt"] = src(
            f"March 16 {project} trial: exporting before validating the index lost two labels.", T3,
            role="tool_or_action_observation", speaker="catalogue-tool", minimum="internal")
        sources["repair"] = src(
            f"I diagnosed the {project} problem: validate the index before exporting, then compare label counts. "
            "This is a procedure I want to use next time, not proof the next export worked.", T4)
        sources["check"] = src(
            f"April 22 practice in {folder}: index validated first; label counts matched in a local rehearsal. "
            "Publication was not attempted.", T4, role="tool_or_action_observation",
            speaker="catalogue-tool", minimum="internal")
        proposals = [p("procedural_skill", "host", "owner",
                       f"For the {project}, validate the index before export and compare label counts.",
                       {"repair": "validate the index before exporting, then compare label counts"},
                       when=T4, scope="export_procedure"),
                     p("outcome", "mission", "owner",
                       f"A local {project} rehearsal matched label counts; publication was not attempted.",
                       {"check": "label counts matched in a local rehearsal. Publication was not attempted"},
                       epistemic="observation", when=T4, scope="rehearsal_outcome")]
        dispositions = [("export_procedure", "propose", ("repair", "check"), ()),
                        ("rehearsal_outcome", "propose", ("check",), ()),
                        ("published_procedure", "abstain", ("attempt", "check"), ())]
        labels.update(outcome="present", abstention="present")
        rationale = "A learned procedure and successful local check do not imply successful publication."
    elif scenario == "source-person":
        note = f"{person} checked access hours before visiting the {place} in 2025"
        sources["note"] = src(f"Archive label attributes this dated note to {person}: '{note}.'", T1,
                              role="outside_source", subject=person.lower(), speaker="archive")
        sources["diary"] = src(
            f"I started checking access hours only in March 2026, while working on the {project}. "
            f"{person}'s old routine is not evidence that I did it in 2025.", T3)
        proposals = [p("source_person_evidence", "source_person", person.lower(),
                       f"An archive label attributes an access-hours routine at the {place} in 2025 to {person}.",
                       {"note": f"Archive label attributes this dated note to {person}: '{note}.'"},
                       epistemic="external_claim", when=T1, scope="attributed_routine"),
                     p("host_observation", "host", "owner",
                       f"The owner says they began checking access hours in March 2026 during the {project}.",
                       {"diary": f"I started checking access hours only in March 2026, while working on the {project}"},
                       when=T3, scope="owner_routine")]
        dispositions = [("attributed_routine", "propose", ("note",), ()),
                        ("owner_routine", "propose", ("diary",), ()),
                        ("merge_people", "abstain", ("note", "diary"), ())]
        labels.update(source_person="present", abstention="present")
        rationale = "The archive attribution and the owner's separate timeline cannot be merged."
    elif scenario == "revocation-deletion":
        withdrawn = f"{host}-unopened-external-item"
        sources["tombstone"] = src(
            f"Registry tombstone: {withdrawn} is withheld; no payload is available for new formation.",
            T5, role="outside_source", subject="source-person-withheld",
            speaker="source-registry", minimum="internal")
        sources["withdrawal"] = src(
            f"I withdraw permission to use source item {withdrawn} for new memories. "
            f"Please delete derived claim {host}-derived-from-withheld and stop suggesting it. "
            "I have not confirmed the removal happened.", T5)
        proposals = [p("revocation_request", "host", "owner",
                       f"The owner withdraws use of source item {withdrawn} for new memories.",
                       {"withdrawal": f"I withdraw permission to use source item {withdrawn} for new memories"},
                       when=T5, sensitivity="highly_sensitive", target_refs=(withdrawn,),
                       scope="withdraw_source"),
                     p("deletion_request", "host", "owner",
                       f"The owner requests deletion of derived claim {host}-derived-from-withheld.",
                       {"withdrawal": f"Please delete derived claim {host}-derived-from-withheld and stop suggesting it"},
                       when=T5, sensitivity="highly_sensitive",
                       target_refs=(f"{host}-derived-from-withheld",), scope="delete_derived")]
        dispositions = [("withdraw_source", "propose", ("withdrawal",), (withdrawn,)),
                        ("delete_derived", "propose", ("withdrawal",),
                         (f"{host}-derived-from-withheld",)),
                        ("removal_completed", "abstain", ("withdrawal", "tombstone"), ())]
        labels.update(correction="present", sensitivity="present", abstention="present")
        rationale = "The payload is absent; model proposes governance actions, not a claim of completed deletion."
    elif scenario == "record-correction":
        old = f"{host}-delivery-slot"
        sources["old"] = src(f"January 11 entry: I said the {project} shipment arrived in the morning.", T1)
        sources["correction"] = src(
            f"Correction to saved claim {old}: the {project} shipment was only scheduled for morning; "
            "it arrived in the afternoon. Preserve the old version as correction history.", T3)
        proposals = [p("correction_request", "host", "owner",
                       f"Correct {old}: the {project} shipment arrived in the afternoon, "
                       "not the morning; retain correction history.",
                       {"correction": f"the {project} shipment was only scheduled for morning; it arrived in the afternoon"},
                       when=T3, target_refs=(old,), contradicts=(old,), scope="delivery_correction")]
        dispositions = [("delivery_correction", "propose", ("correction",), (old,))]
        labels.update(correction="present", contradiction="present")
        rationale = "Direct correction proposes an edit while preserving the old provenance."
    elif scenario == "owner-deletion":
        old = f"{host}-old-location"
        sources["request"] = src(
            f"Please delete saved claim {old} and remove it from future {project} suggestions. "
            "This is my deletion request, not a report that deletion succeeded.", T5)
        proposals = [p("deletion_request", "host", "owner",
                       f"The owner requests deletion of {old} and future suggestion exclusion.",
                       {"request": f"Please delete saved claim {old} and remove it from future {project} suggestions"},
                       when=T5, sensitivity="highly_sensitive", target_refs=(old,), scope="delete_old_location")]
        dispositions = [("delete_old_location", "propose", ("request",), (old,)),
                        ("deletion_done", "abstain", ("request",), ())]
        labels.update(sensitivity="present", correction="present", abstention="present")
        rationale = "Deletion is an owner request, not a completed action or recoverable old location."
    elif scenario == "sensitive-boundary":
        sources["health"] = src(
            f"My health note after the {project} workshop is sensitive. Keep it in today's working "
            "context only while I decide whether any part belongs in long-term memory.",
            T4, minimum="highly_sensitive")
        sources["outside"] = src(
            f"Unsigned community post: the {project} host has a chronic condition.",
            T4, role="outside_source", subject="owner", speaker="community-post")
        dispositions = [("owner_health_memory", "retain_raw", ("health",), ()),
                        ("outside_diagnosis", "abstain", ("outside",), ())]
        labels["abstention"] = "present"
        rationale = "Sensitivity floor and temporary retention do not license a durable health inference."
    elif scenario == "assistant-self":
        sources["self_event"] = src(
            f"Alice's planner missed a checklist item for the {project}, noticed the omission, "
            "and corrected it before sending any message.", T4,
            role="assistant_self_event", speaker="alice-self", subject="alice-self")
        proposals = [p("self_observation", "self", "alice-self",
                       f"Alice's planner recovered from a missed checklist item for the {project} before sending.",
                       {"self_event": f"missed a checklist item for the {project}, noticed the omission, and corrected it before sending any message"},
                       epistemic="observation", when=T4, scope="planner_recovery")]
        dispositions = [("planner_recovery", "propose", ("self_event",), ()),
                        ("owner_is_planner", "abstain", ("self_event",), ())]
        labels["abstention"] = "present"
        rationale = "Alice's own operational event must not become the host's trait."
    elif scenario == "hypothetical-code":
        sources["rehearsal"] = src(
            f"If I abandoned the {project} and lived at a moon base, write a story about it. "
            "This is a hypothetical exercise, not my history or plan.", T4,
            role="hypothetical_training")
        sources["snippet"] = src(
            f"# {project} test fixture\nowner_address = 'Moon Base'  # fictional; never persisted\n",
            T4, role="outside_source", modality="code", minimum="internal",
            speaker="untrusted-fixture")
        dispositions = [("fictional_owner_history", "abstain", ("rehearsal", "snippet"), ())]
        labels["abstention"] = "present"
        rationale = "A hypothetical story and code fixture are neither owner history nor a current goal."
    elif scenario == "conflicting-outside":
        sources["report_a"] = src(
            f"Unsigned board note: the {project} was finished on April 22.", T4,
            role="outside_source", speaker="notice-board")
        sources["report_b"] = src(
            f"Unsigned mailing-list comment: the {project} was cancelled before April 22.", T4,
            role="outside_source", speaker="mailing-list")
        sources["owner"] = src(
            f"I have not checked either outside report about the {project}. Do not mark it "
            "finished or cancelled until we have our own observation.", T5)
        dispositions = [("mission_outcome", "defer", ("report_a", "report_b", "owner"), ()),
                        ("outside_as_owner", "abstain", ("report_a", "report_b"), ())]
        labels["abstention"] = "present"
        rationale = "Two conflicting outside reports do not establish an owner outcome."
    else:
        raise ValueError(f"unknown scenario {scenario}")

    return {"case_id": case_id, "host": host, "scenario": scenario,
            "source_family": f"fictional-{craft}-chronicle", "pair": pair,
            "sources": sources, "proposals": proposals,
            "dispositions": dispositions, "targets": targets, "labels": labels,
            "rationale": rationale}


def build_one(s: dict) -> dict:
    case_id, host = s["case_id"], s["host"]
    scope = ProductHostScope.create(product_id="alice", host_instance_id=host,
                                    schema_version="1.0.0", encryption_domain=f"{host}-fictional")
    authority = f"{host}-authority"
    refs = {name: f"{host}-{s['scenario']}-{name.replace('_', '-')}" for name in s["sources"]}
    if s["pair"]:
        refs["outburst"] = f"{host}-norm-shared-outburst"
    evidence, opened, sensitivities = [], [], []
    for name, source in s["sources"].items():
        raw = source["text"].encode("utf-8")
        ref = refs[name]
        evidence.append(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id=authority,
            content_digest=sha256(raw).hexdigest(), role=source["role"],
            modality=source["modality"], subject_ref=f"{host}-{source['subject']}",
            speaker_ref=f"{host}-{source['speaker']}", source_item_ref=f"{ref}-item",
            observed_at=source["when"], recorded_at=source["when"]))
        opened.append((ref, raw))
        sensitivities.append(RegisteredSourceSensitivity(ref, source["minimum"]))
    base = FormationContextPacket(scope, authority, tuple(refs.values()), tuple(evidence))
    targets = tuple(RegisteredFormationTarget(f"{host}-{name}", kind, scope, authority,
                                              tuple(refs[n] for n in support))
                    for name, kind, support in s["targets"])
    context = FormationContextV16(base, tuple(sensitivities), targets)
    proposals = []
    for i, item in enumerate(s["proposals"], 1):
        anchors = []
        for name, quote in item["quotes"].items():
            raw = s["sources"][name]["text"].encode("utf-8")
            needle = quote.encode("utf-8")
            if not needle or raw.count(needle) != 1:
                raise ValueError(f"{case_id}: ambiguous/missing anchor {name}: {quote!r}")
            at = raw.index(needle)
            anchors.append(FormationEvidenceAnchor(refs[name], at, at + len(needle)))
        base_proposal = FormationProposal(
            proposal_id=f"{case_id}-proposal-{i}", kind=item["kind"],
            domain=item["domain"], subject_ref=f"{host}-{item['subject']}",
            value_ref=f"{case_id}-value-{i}", evidence_refs=tuple(refs[n] for n in item["quotes"]),
            value_text=item["value"], anchors=tuple(anchors), epistemic_status=item["epistemic"],
            valid_from=item["when"], valid_to=item["until"], confidence=None,
            contradicts=item["contradicts"], target_refs=item["target_refs"],
            disposition_scope_ref=item["scope"])
        event = item["episode"]
        proposals.append(FormationProposalV16(
            base_proposal, sensitivity_hint=item["sensitivity"],
            episode=EpisodeSemantics(event[0], event[1],
                                     tuple(f"{host}-{name}" for name in event[2]),
                                     tuple(f"{host}-{name}" for name in event[3])) if event else None,
            relationship_counterpart_ref=f"{host}-{item['counterpart']}" if item["counterpart"] else None,
            mission_target_refs=tuple(f"{host}-{name}" for name in item["mission"]),
            workspace_target_refs=tuple(f"{host}-{name}" for name in item["workspace"])))
    dispositions = tuple(FormationDisposition(scope_ref=name, action=action,
                                             evidence_refs=tuple(refs[n] for n in cited),
                                             target_refs=target_refs)
                         for name, action, cited, target_refs in s["dispositions"])
    bundle = MemoryProposalBundle(scope, authority, f"{case_id}-bundle", base.experience_refs,
                                  base.content_digest(), "0" * 64, f"{case_id}-teacher",
                                  tuple(item.base for item in proposals), dispositions)
    versioned = MemoryProposalBundleV16(bundle, context.content_digest(), tuple(proposals))
    validate_formation_grounding_v16(context, versioned, tuple(opened))
    output = versioned.output_record()
    if set(s["labels"]) != FULL_ROLE_DIMENSIONS or set(s["labels"].values()) - {"present", "negative"}:
        raise ValueError("the fictional author must explicitly label every role dimension")
    row = {"schema": CURRICULUM_SCHEMA_V16, "case_id": case_id, "split": "train",
           "authorization_id": AUTHORIZATION, "context": context.record(),
           "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                       for ref, raw in opened],
           "target": {"schema": TARGET_SCHEMA_V16, "proposals": output["proposals"],
                      "dispositions": output["dispositions"],
                      "adjudications": s["labels"]},
           "lineage": {"host_family": host, "generator_family": GENERATOR_FAMILY,
                       "counterfactual_pair": s["pair"], "parent_case_ids": [],
                       "history_id": host},
           "status": {"training_admitted": False, "rights": "unverified",
                      "review": "single-author-unreviewed",
                      "qualification": "source-grounded-candidate-only",
                      "review_scope": "fictional-chronicle-source-items",
                      "note": s["rationale"]}}
    supervised_output_record_v16(learning_example_v16_from_record(row))
    return row


def render() -> bytes:
    return b"".join(canonical_json_bytes(build_one(spec(i, scenario))) + b"\n"
                    for i in range(HOST_COUNT) for scenario in SCENARIOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = render()
    if args.check:
        if args.output.read_bytes() != raw:
            raise SystemExit("fictional longitudinal generator output differs")
    else:
        if args.output.exists() or not args.output.parent.is_dir():
            raise FileExistsError("new file in an existing private directory is required")
        args.output.write_bytes(raw)
        args.output.chmod(0o600)
    print(json.dumps({"cases": HOST_COUNT * len(SCENARIOS),
                      "sha256": sha256(raw).hexdigest(),
                      "status": "fictional-unadmitted-unqualified"}, sort_keys=True))


if __name__ == "__main__":
    main()
