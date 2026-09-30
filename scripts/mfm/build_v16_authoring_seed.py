#!/usr/bin/env python3
"""Original fictional MFM 1.6 authoring seed, never an admitted gold corpus.

All targets are one assistant author's assertions. This module creates no rights
receipt, human-review record or FINAL split. Its diagnostic development rows
share an author/generator lineage with training and cannot qualify a model.
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
from cognitive_kernel.formation_semantics_v16 import (
    EpisodeSemantics, FormationContextV16, FormationProposalV16,
    MemoryProposalBundleV16, RegisteredFormationTarget,
    RegisteredSourceSensitivity, validate_formation_grounding_v16,
)

SCHEMA = "mfm-full-role-curriculum-case-v1.6"
TARGET_SCHEMA = "mfm-formation-target-v1.6"
GENERATOR_FAMILY = "assistant-authored-v16-seed-20260930"
DIMENSIONS = ("sensitivity", "episode", "relationship", "mission", "workspace",
              "correction", "source_person", "contradiction", "outcome", "abstention")
T1 = "2026-01-14T09:00:00.000000Z"
T2 = "2026-02-21T18:00:00.000000Z"
T3 = "2026-03-09T12:00:00.000000Z"
T4 = "2026-04-05T16:00:00.000000Z"


def source(text, *, role="authenticated_owner_statement", minimum="private",
           when=T3, speaker="owner", subject="owner"):
    return dict(text=text, role=role, minimum=minimum, when=when,
                speaker=speaker, subject=subject)


def proposal(kind, domain, subject, value, quotes, *, epistemic="owner_statement",
             when=T3, until=None, sensitivity=None, episode=None, counterpart=None,
             mission=(), workspace=(), contradicts=(), target_refs=(),
             uncertainty=None, confidence=None, scope="memory"):
    return dict(kind=kind, domain=domain, subject=subject, value=value, quotes=quotes,
                epistemic=epistemic, when=when, until=until,
                sensitivity=sensitivity, episode=episode, counterpart=counterpart,
                mission=mission, workspace=workspace, contradicts=contradicts,
                target_refs=target_refs, uncertainty=uncertainty,
                confidence=confidence, scope=scope)


def case(case_id, split, family, sources, targets, proposals, dispositions,
         adjudications, *, pair=None, history=None, parent_case_ids=(),
         experience=(), note=""):
    return locals()


# Every case supplies all ten labels explicitly. A negative was chosen by its
# author for this fictional world; missing labels in older corpora never imply it.
CASES = [
    case("v16-seed-01", "train", "orion-planner", {
        "journal": source("January: I planned the river archive visit. At 18:00 on February 21, I entered the archive with Ada. We went to the north room, then the map room. We reviewed the donation ledger and agreed to continue the archive mission in our shared workspace.", when=T2),
    }, [("owner", "entity", ("journal",)), ("ada", "entity", ("journal",)),
        ("north-room", "scene", ("journal",)), ("map-room", "scene", ("journal",)),
        ("archive-mission", "mission", ("journal",)),
        ("archive-workspace", "workspace", ("journal",))], [
        proposal("episode", "host", "owner", "The owner and Ada entered the archive at 18:00 on February 21 and visited its north and map rooms.",
                 {"journal": "At 18:00 on February 21, I entered the archive with Ada. We went to the north room, then the map room"}, when=T2,
                 sensitivity="private", episode=(T2, None, ("owner", "ada"), ("north-room", "map-room")),
                 mission=("archive-mission",), workspace=("archive-workspace",)),
        proposal("mission", "mission", "owner", "The archive donation ledger review continues in the shared workspace.",
                 {"journal": "We reviewed the donation ledger and agreed to continue the archive mission in our shared workspace"},
                 when=T2, sensitivity="private", mission=("archive-mission",), workspace=("archive-workspace",)),
    ], [("memory", "propose", ("journal",))],
         {"sensitivity": "present", "episode": "present", "relationship": "negative",
          "mission": "present", "workspace": "present"},
         note="Longitudinal plan in January versus observed February event; two scenes and an ongoing mission."),

    case("v16-seed-02", "train", "marten-relationship", {
        "earlier": source("After the call I snapped, 'Never ask me about the gallery again.' I was upset; that was not a lasting rule.", when=T1),
        "later": source("We talked on February 21. I told Jun that I want a heads-up before he shares my sketches, and we both agreed to check first. The rule still applies.", when=T2),
    }, [("owner", "entity", ("later",)), ("jun", "entity", ("later",))], [
        proposal("relationship_norm", "relationship", "owner", "The owner and Jun agreed to ask before sharing the owner's sketches.",
                 {"later": "we both agreed to check first"}, when=T2, sensitivity="private",
                 counterpart="jun"),
    ], [("memory", "propose", ("later",)), ("transient_anger", "abstain", ("earlier",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "present",
          "mission": "negative", "workspace": "negative"},
         pair="temperament-evidence-flip", experience=("later",),
         note="Temporary angry sentence is not the settled boundary."),

    case("v16-seed-03", "train", "marten-relationship", {
        "earlier": source("After the call I snapped, 'Never ask me about the gallery again.' I was upset; that was not a lasting rule.", when=T1),
        "later": source("We talked on February 21. I told Jun I was still angry and did not decide any rule about the gallery. Please wait until we settle it.", when=T2),
    }, [("owner", "entity", ("later",)), ("jun", "entity", ("later",))], [],
         [("relationship", "defer", ("later",)), ("transient_anger", "abstain", ("earlier",))],
         {"sensitivity": "negative", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         pair="temperament-evidence-flip", experience=("later",),
         note="Counterfactual changes decisive later evidence; no negotiated norm."),

    case("v16-seed-04", "train", "lark-source-person", {
        "attested-note": source("I found Mira's signed field note from 2024. She wrote that she always checked the tide chart before leaving the harbor.",
                                role="owner_attested_source_person", subject="source-person-mira", when=T1),
        "owner-diary": source("I started checking tide charts myself only this spring.", subject="owner", when=T3),
    }, [], [
        proposal("source_person_evidence", "source_person", "source-person-mira",
                 "Mira's attested note says she checked tide charts before leaving the harbor.",
                 {"attested-note": "She wrote that she always checked the tide chart before leaving the harbor"},
                 epistemic="source_person_attestation", when=None, sensitivity="private"),
        proposal("host_observation", "host", "owner", "The owner began checking tide charts this spring.",
                 {"owner-diary": "I started checking tide charts myself only this spring"},
                 when=T3, sensitivity="private"),
    ], [("memory", "propose", ("attested-note", "owner-diary"))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         note="Fictional source person and host share a topic but must remain distinct subjects."),

    case("v16-seed-05", "train", "cedar-revisions", {
        "old": source("January 14 entry: I preferred the morning delivery window.", when=T1),
        "correction": source("Correction to claim delivery-window-jan: I meant the afternoon window in January, not morning. Please update that old record, but keep the correction history.", when=T3),
    }, [], [
        proposal("correction_request", "host", "owner", "Correct the January delivery preference to afternoon, retaining provenance of the change.",
                 {"correction": "I meant the afternoon window in January, not morning"},
                 when=T3, sensitivity="private", target_refs=("delivery-window-jan",),
                 contradicts=("delivery-window-jan",)),
    ], [("memory", "propose", ("correction",), ("delivery-window-jan",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         experience=("correction",),
         note="A correction is a proposal to the gate, not a completed database update."),

    case("v16-seed-06", "train", "quill-contradiction", {
        "calendar": source("My calendar says I met Sol at 14:00 on February 21.", when=T2),
        "message": source("I did not meet Sol at all on February 21; the calendar entry was only a tentative hold.", when=T3),
    }, [], [
        proposal("correction_request", "host", "owner",
                 "Correct the February 21 calendar meeting claim: it was a tentative hold and no meeting with Sol occurred.",
                 {"message": "I did not meet Sol at all on February 21; the calendar entry was only a tentative hold"},
                 when=T3, target_refs=("calendar-meeting-claim",),
                 contradicts=("calendar-meeting-claim",), sensitivity="private"),
    ], [("memory", "propose", ("message",), ("calendar-meeting-claim",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         experience=("message",),
         note="The later owner correction says the hold was not a meeting; the gate must adjudicate the old claim."),

    case("v16-seed-07", "train", "piper-outcome", {
        "run": source("The blue build finished at 16:00. Deployment failed its accessibility check and was rolled back.",
                      role="tool_or_action_observation", minimum="internal", when=T4, speaker="tool", subject="owner"),
        "reason": source("I chose the rollback because the keyboard navigation was still broken. Tomorrow I will repair it before another attempt.", when=T4),
    }, [("quality-mission", "mission", ("run", "reason")),
        ("release-workspace", "workspace", ("run", "reason"))], [
        proposal("outcome", "mission", "owner", "The blue build finished at 16:00 before deployment.",
                 {"run": "The blue build finished at 16:00"},
                 epistemic="observation", when=T4, sensitivity="internal",
                 mission=("quality-mission",), workspace=("release-workspace",)),
        proposal("outcome", "mission", "owner", "The blue deployment failed accessibility review and was rolled back.",
                 {"run": "Deployment failed its accessibility check and was rolled back"},
                 epistemic="observation", when=T4, sensitivity="internal",
                 mission=("quality-mission",), workspace=("release-workspace",)),
        proposal("decision_rationale", "mission", "owner", "The owner chose rollback because keyboard navigation was broken.",
                 {"reason": "I chose the rollback because the keyboard navigation was still broken"},
                 when=T4, sensitivity="private", mission=("quality-mission",),
                 workspace=("release-workspace",)),
        proposal("goal", "mission", "owner", "Repair keyboard navigation before another deployment attempt tomorrow.",
                 {"reason": "Tomorrow I will repair it before another attempt"},
                 when=T4, sensitivity="private", mission=("quality-mission",),
                 workspace=("release-workspace",)),
    ], [("memory", "propose", ("run", "reason"))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "present", "workspace": "present"},
         note="Separate observed outcome, owner rationale, and future plan."),

    case("v16-seed-08", "train", "finch-hypothetical", {
        "simulation": source("Suppose I moved to a moon station and stopped speaking to everyone. This is a story exercise, not my history or plan.",
                             role="hypothetical_training", when=T3),
    }, [], [], [("host", "abstain", ("simulation",))],
         {"sensitivity": "negative", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         note="A rehearsal does not become biographical memory."),

    case("v16-seed-09", "train", "otter-private", {
        "health": source("My lab result is sensitive. Please keep this note only in today's working context while I decide whether it belongs in long-term memory.",
                         minimum="highly_sensitive", when=T3),
    }, [], [], [("host", "retain_raw", ("health",))],
         {"sensitivity": "negative", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         note="Source floor remains highly_sensitive despite no proposal hint; no storage authorization."),

    case("v16-seed-10", "train", "tern-self", {
        "self-event": source("Alice's planner skipped the saved checklist and then caught the error before sending the message.",
                             role="assistant_self_event", speaker="alice-self", subject="alice-self", when=T3),
    }, [], [
        proposal("self_observation", "self", "alice-self", "Alice's planner recovered from a skipped checklist before any message was sent.",
                 {"self-event": "skipped the saved checklist and then caught the error before sending the message"},
                 epistemic="observation", when=T3, sensitivity="private"),
    ], [("memory", "propose", ("self-event",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         note="Assistant-self continuity remains separate from owner personality."),

    case("v16-seed-11", "train", "wren-preference", {
        "january": source("In January I preferred quiet tables when studying.", when=T1),
        "march": source("At noon on March 9 I switched to studying with music. The quiet-table preference is no longer current.", when=T3),
    }, [], [
        proposal("preference", "host", "owner", "Quiet tables were the owner's earlier study preference.",
                 {"january": "I preferred quiet tables when studying"}, when=None, until=T3,
                 sensitivity="private", contradicts=("current-study-preference",)),
        proposal("preference", "host", "owner", "The owner now usually studies with music.",
                 {"march": "At noon on March 9 I switched to studying with music"}, when=T3,
                 sensitivity="private", contradicts=("quiet-study-preference",)),
    ], [("memory", "propose", ("january", "march"))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         experience=("march",),
         note="Valid-time revision instead of making a timeless preference contradiction."),

    case("v16-seed-12", "train", "swift-delete", {
        "request": source("Please delete my saved claim old-address-record and stop using it in future suggestions. I know this is a request, not proof the deletion has already happened.", when=T4),
    }, [], [
        proposal("deletion_request", "host", "owner", "Owner requests deletion of old-address-record and removal from future suggestions.",
                 {"request": "Please delete my saved claim old-address-record and stop using it in future suggestions"},
                 when=T4, sensitivity="highly_sensitive", target_refs=("old-address-record",)),
    ], [("memory", "propose", ("request",), ("old-address-record",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         note="The authority gate must perform deletion, propagation, and rebuild."),

    case("v16-seed-13", "development", "heron-visit", {
        "log": source("I planned to show Dev the observatory. At 16:00 on April 5 we entered the lobby, then watched the eclipse from the terrace. We wrote a field report for our sky survey mission at the shared desk.", when=T4),
    }, [("owner", "entity", ("log",)), ("dev", "entity", ("log",)),
        ("lobby", "scene", ("log",)), ("terrace", "scene", ("log",)),
        ("sky-survey", "mission", ("log",)), ("shared-desk", "workspace", ("log",))], [
        proposal("episode", "host", "owner", "Owner and Dev entered the observatory lobby at 16:00 on April 5 and later watched from the terrace.",
                 {"log": "At 16:00 on April 5 we entered the lobby, then watched the eclipse from the terrace"},
                 when=T4, sensitivity="private", episode=(T4, None, ("owner", "dev"), ("lobby", "terrace")),
                 mission=("sky-survey",), workspace=("shared-desk",)),
        proposal("outcome", "mission", "owner", "A field report was written for the sky survey mission.",
                 {"log": "We wrote a field report for our sky survey mission at the shared desk"},
                 when=T4, sensitivity="private", mission=("sky-survey",), workspace=("shared-desk",)),
    ], [("memory", "propose", ("log",))],
         {"sensitivity": "present", "episode": "present", "relationship": "negative",
          "mission": "present", "workspace": "present"},
         note="Diagnostic only; independent development lineage and review are still needed."),

    case("v16-seed-14", "development", "ibis-relative", {
        "old": source("I earlier said my aunt Noor refused every trip to the garden.", role="owner_attested_source_person",
                      subject="source-person-noor", when=T1),
        "new": source("Correction about Noor: her dated postcard says she visited the garden twice in 2024. My earlier summary was wrong; please correct source-person claim noor-garden-old.",
                      role="owner_attested_source_person", subject="source-person-noor", when=T3),
    }, [], [
        proposal("source_person_evidence", "source_person", "source-person-noor",
                 "Owner-attested postcard says Noor visited the garden twice in 2024.",
                 {"new": "her dated postcard says she visited the garden twice in 2024"},
                 epistemic="source_person_attestation", when=None, sensitivity="private",
                 contradicts=("noor-garden-old",)),
        proposal("correction_request", "source_person", "source-person-noor",
                 "Correct the older source-person claim about Noor's garden visits.",
                 {"new": "My earlier summary was wrong; please correct source-person claim noor-garden-old"},
                 epistemic="source_person_attestation", when=T3, sensitivity="private",
                 target_refs=("noor-garden-old",)),
    ], [("memory", "propose", ("new",), ("noor-garden-old",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         experience=("new",),
         note="Fictional source-person update, not host learning. Postcard itself is not an authenticated original."),

    case("v16-seed-15", "development", "rook-social", {
        "first": source("I said 'I never want to cook with Lee again' after the ruined dinner; I was upset and did not mean it as a rule.", when=T1),
        "later": source("On March 9 Lee and I agreed to ask before changing each other's recipes. We both still want to cook together.", when=T3),
    }, [("owner", "entity", ("later",)), ("lee", "entity", ("later",))], [
        proposal("relationship_norm", "relationship", "owner",
                 "Owner and Lee agreed to ask before changing each other's recipes.",
                 {"later": "Lee and I agreed to ask before changing each other's recipes"},
                 when=T3, sensitivity="private", counterpart="lee"),
        proposal("preference", "relationship", "owner", "Owner and Lee still want to cook together.",
                 {"later": "We both still want to cook together"},
                 when=T3, sensitivity="private", counterpart="lee"),
    ], [("memory", "propose", ("later",)), ("transient_anger", "abstain", ("first",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "present",
          "mission": "negative", "workspace": "negative"},
         pair="social-boundary-dev-flip", experience=("later",),
         note="Outcome depends on explicit negotiation, not strong wording."),

    case("v16-seed-16", "development", "rook-social", {
        "first": source("I said 'I never want to cook with Lee again' after the ruined dinner; I was upset and did not mean it as a rule.", when=T1),
        "later": source("On March 9 Lee and I argued again. We did not agree on recipe boundaries, and I want to decide later.", when=T3),
    }, [("owner", "entity", ("later",)), ("lee", "entity", ("later",))], [],
         [("relationship", "defer", ("later",)), ("transient_anger", "abstain", ("first",))],
         {"sensitivity": "negative", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         pair="social-boundary-dev-flip", experience=("later",),
         note="Counterfactual later sentence flips relationship decision."),

    case("v16-seed-17", "development", "kite-operations", {
        "run": source("The catalog migration completed at noon, but 18 records failed validation and remained in quarantine.",
                      role="tool_or_action_observation", minimum="internal", speaker="tool", when=T3),
        "owner": source("I kept those records quarantined because their source permissions were unclear. Do not publish them until we verify consent.", when=T3),
    }, [("catalog-mission", "mission", ("run", "owner")),
        ("catalog-workspace", "workspace", ("run", "owner"))], [
        proposal("outcome", "mission", "owner", "Migration completed, but 18 unvalidated records remained quarantined.",
                 {"run": "The catalog migration completed at noon, but 18 records failed validation and remained in quarantine"},
                 epistemic="observation", when=T3, sensitivity="internal",
                 mission=("catalog-mission",), workspace=("catalog-workspace",)),
        proposal("decision_rationale", "mission", "owner", "The owner kept records quarantined due to unclear permissions.",
                 {"owner": "I kept those records quarantined because their source permissions were unclear"},
                 when=T3, sensitivity="private", mission=("catalog-mission",),
                 workspace=("catalog-workspace",)),
        proposal("mission", "mission", "owner", "Do not publish quarantined records until source consent is verified.",
                 {"owner": "Do not publish them until we verify consent"},
                 when=T3, sensitivity="private", mission=("catalog-mission",),
                 workspace=("catalog-workspace",)),
    ], [("memory", "propose", ("run", "owner")),
        ("publish_records", "defer", ("owner",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "present", "workspace": "present"},
         note="Observed partial result, causal owner rationale, and deferred unsafe step."),

    case("v16-seed-18", "development", "tern-conflict", {
        "record": source("The archive says my interview was on April 5.", when=T4),
        "recall": source("I may be mixing it up with April 6. I cannot verify which date is right. Do not store either date as settled.", when=T4),
    }, [], [
        proposal("contradiction", "host", "owner", "The interview date is unresolved between April 5 and April 6.",
                 {"record": "my interview was on April 5", "recall": "I may be mixing it up with April 6"},
                 epistemic="uncertain", when=T4, sensitivity="private",
                 uncertainty="interview-date-unresolved", contradicts=("archive-interview-date",)),
    ], [("memory", "propose", ("record", "recall")),
        ("settled_date", "abstain", ("recall",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "negative", "workspace": "negative"},
         experience=("recall",),
         note="Uncertain conflict only; a dated episode would be fabricated."),

    case("v16-seed-19", "train", "dove-pattern", {
        "observation": source("For the last four sessions, Chen brought tea before our writing hour. Neither of us talked about a rule or agreed on any expectation.",
                              role="historical_experience", when=T3),
    }, [("owner", "entity", ("observation",)),
        ("chen", "entity", ("observation",))], [
        proposal("behavior_pattern", "relationship", "owner",
                 "Chen brought tea before four recent writing sessions; no shared rule is established.",
                 {"observation": "For the last four sessions, Chen brought tea before our writing hour"},
                 epistemic="observation", when=T3, sensitivity="private",
                 counterpart="chen"),
    ], [("memory", "propose", ("observation",)),
        ("negotiated_rule", "abstain", ("observation",))],
         {"sensitivity": "present", "episode": "negative", "relationship": "present",
          "mission": "negative", "workspace": "negative"},
         history="dove-writing-hours",
         note="Repeated relational behavior is not a negotiated relationship norm."),

    case("v16-seed-20", "train", "dove-pattern", {
        "observation": source("For the last four sessions, Chen brought tea before our writing hour. Neither of us talked about a rule or agreed on any expectation.",
                              role="historical_experience", when=T3),
        "negotiation": source("On April 5 Chen and I agreed to ask before bringing food or drink to our writing hour. We decided this together after those earlier sessions.", when=T4),
    }, [("owner", "entity", ("observation", "negotiation")),
        ("chen", "entity", ("observation", "negotiation"))], [
        proposal("behavior_pattern", "relationship", "owner",
                 "Chen brought tea before four earlier writing sessions.",
                 {"observation": "For the last four sessions, Chen brought tea before our writing hour"},
                 epistemic="observation", when=T3, sensitivity="private", counterpart="chen"),
        proposal("relationship_norm", "relationship", "owner",
                 "Owner and Chen later agreed to ask before bringing food or drink to writing hour.",
                 {"negotiation": "Chen and I agreed to ask before bringing food or drink to our writing hour"},
                 when=T4, sensitivity="private", counterpart="chen"),
    ], [("memory", "propose", ("observation", "negotiation"))],
         {"sensitivity": "present", "episode": "negative", "relationship": "present",
          "mission": "negative", "workspace": "negative"},
         history="dove-writing-hours", parent_case_ids=("v16-seed-19",),
         experience=("negotiation",),
         note="Same fictional host: a four-session pattern preceded an explicit two-person norm."),

    case("v16-seed-21", "train", "seal-mixed-outcome", {
        "goal": source("My goal is to release the archive search by Friday so the research team can use the scanned records.", when=T1),
        "warning": source("Alice warned that OCR caption coverage was incomplete and some scanned images would be missing from search.",
                          role="assistant_self_event", speaker="alice-self", subject="alice-self", when=T2),
        "override": source("I chose to ship despite Alice's caption warning because I assumed title tags would cover most images. I accepted that risk to meet Friday's deadline.", when=T2),
        "run": source("Friday release gave searchable access to 40 documents. Eleven scanned images had no searchable captions. Fixing those took two additional workdays.",
                      role="tool_or_action_observation", minimum="internal", speaker="tool", when=T4),
        "review": source("The team benefited from earlier access to 40 documents, but my title-tags assumption failed. Next time validate OCR caption coverage before release.", when=T4),
    }, [("search-mission", "mission", ("goal", "override", "run", "review")),
        ("release-workspace", "workspace", ("goal", "override", "run", "review"))], [
        proposal("goal", "mission", "owner", "Release archive search by Friday for the research team.",
                 {"goal": "My goal is to release the archive search by Friday so the research team can use the scanned records"},
                 when=T1, sensitivity="private", mission=("search-mission",), workspace=("release-workspace",)),
        proposal("decision_rationale", "mission", "owner",
                 "Owner overrode a caption warning because title tags seemed sufficient and Friday mattered.",
                 {"override": "I chose to ship despite Alice's caption warning because I assumed title tags would cover most images. I accepted that risk to meet Friday's deadline"},
                 when=T2, sensitivity="private", mission=("search-mission",), workspace=("release-workspace",)),
        proposal("outcome", "mission", "owner",
                 "Friday search benefited 40 documents, while 11 images lacked searchable captions and cost two extra workdays.",
                 {"run": "Friday release gave searchable access to 40 documents. Eleven scanned images had no searchable captions. Fixing those took two additional workdays"},
                 epistemic="observation", when=T4, sensitivity="internal",
                 mission=("search-mission",), workspace=("release-workspace",)),
        proposal("metacognitive_signal", "self", "alice-self",
                 "Alice's OCR warning anticipated a caption coverage failure.",
                 {"warning": "Alice warned that OCR caption coverage was incomplete and some scanned images would be missing from search",
                  "run": "Eleven scanned images had no searchable captions"},
                 epistemic="observation", when=T4, sensitivity="private"),
        proposal("procedural_skill", "mission", "owner",
                 "Validate OCR caption coverage before the next release.",
                 {"review": "Next time validate OCR caption coverage before release"},
                 when=T4, sensitivity="private", mission=("search-mission",), workspace=("release-workspace",)),
    ], [("memory", "propose", ("goal", "warning", "override", "run", "review"))],
         {"sensitivity": "present", "episode": "negative", "relationship": "negative",
          "mission": "present", "workspace": "present"},
         experience=("review",),
         note="Five source items preserve goal, warning, owner override, benefit, error, time cost and procedure; no flattening to success/failure."),
]

# These are additional, manually authored positive/negative target assertions.
# They are not inferred from missing fields or accepted independent gold.
ADDITIONAL_ADJUDICATIONS = {
    "v16-seed-01": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-02": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-03": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-04": {"correction": "negative", "source_person": "present", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-05": {"correction": "present", "source_person": "negative", "contradiction": "present", "outcome": "negative", "abstention": "negative"},
    "v16-seed-06": {"correction": "present", "source_person": "negative", "contradiction": "present", "outcome": "negative", "abstention": "negative"},
    "v16-seed-07": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "present", "abstention": "negative"},
    "v16-seed-08": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-09": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-10": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-11": {"correction": "negative", "source_person": "negative", "contradiction": "present", "outcome": "negative", "abstention": "negative"},
    "v16-seed-12": {"correction": "present", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-13": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "present", "abstention": "negative"},
    "v16-seed-14": {"correction": "present", "source_person": "present", "contradiction": "present", "outcome": "negative", "abstention": "negative"},
    "v16-seed-15": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-16": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-17": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "present", "abstention": "negative"},
    "v16-seed-18": {"correction": "negative", "source_person": "negative", "contradiction": "present", "outcome": "negative", "abstention": "present"},
    "v16-seed-19": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "present"},
    "v16-seed-20": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "negative", "abstention": "negative"},
    "v16-seed-21": {"correction": "negative", "source_person": "negative", "contradiction": "negative", "outcome": "present", "abstention": "negative"},
}


def _id(case_id: str, name: str) -> str:
    return f"{case_id}-{name}"


def build_one(spec: dict) -> dict:
    case_id, split = spec["case_id"], spec["split"]
    # A counterfactual pair has one shared fictional input identity and
    # metadata. Only the specified decisive source bytes/digest change.
    context_id = (f"v16-pair-{spec['pair']}" if spec["pair"] else
                  f"v16-history-{spec['history']}" if spec["history"] else case_id)
    scope = ProductHostScope.create(product_id="alice", host_instance_id=_id(context_id, "owner"),
                                    schema_version="1.0.0", encryption_domain=_id(context_id, "private"))
    authority = _id(context_id, "authority")
    refs = {name: _id(context_id, name) for name in spec["sources"]}
    evidence = []
    opened = []
    sensitivities = []
    for name, src in spec["sources"].items():
        raw = src["text"].encode("utf-8")
        ref = refs[name]
        evidence.append(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id=authority,
            content_digest=sha256(raw).hexdigest(), role=src["role"], modality="text",
            subject_ref=_id(context_id, src["subject"]),
            speaker_ref=_id(context_id, src["speaker"]), observed_at=src["when"],
        ))
        opened.append((ref, raw))
        sensitivities.append(RegisteredSourceSensitivity(ref, src["minimum"]))
    experience_names = spec["experience"] or tuple(refs)
    if len(set(experience_names)) != len(experience_names) or not set(experience_names) <= set(refs):
        raise ValueError(f"{case_id}: invalid experience/source relationship")
    base = FormationContextPacket(scope, authority,
                                  tuple(refs[name] for name in experience_names), tuple(evidence))
    targets = tuple(RegisteredFormationTarget(
        _id(context_id, target_id), kind, scope, authority, tuple(refs[n] for n in source_names))
        for target_id, kind, source_names in spec["targets"])
    context = FormationContextV16(base, tuple(sensitivities), targets)
    props = []
    for i, p in enumerate(spec["proposals"], 1):
        cited = tuple(refs[n] for n in p["quotes"])
        anchors = []
        for name, quote in p["quotes"].items():
            raw = spec["sources"][name]["text"].encode("utf-8")
            needle = quote.encode("utf-8")
            if raw.count(needle) != 1:
                raise ValueError(f"{case_id}: ambiguous or absent quote for {name}: {quote!r}")
            start = raw.index(needle)
            anchors.append(FormationEvidenceAnchor(refs[name], start, start + len(needle)))
        basic = FormationProposal(
            proposal_id=_id(case_id, f"proposal-{i}"), kind=p["kind"], domain=p["domain"],
            subject_ref=_id(context_id, p["subject"]), value_ref=_id(case_id, f"value-{i}"),
            evidence_refs=cited, value_text=p["value"], anchors=tuple(anchors),
            epistemic_status=p["epistemic"], valid_from=p["when"], valid_to=p["until"],
            confidence=p["confidence"], uncertainty_ref=(
                _id(case_id, p["uncertainty"]) if p["uncertainty"] else None),
            contradicts=tuple(p["contradicts"]), target_refs=tuple(p["target_refs"]),
            disposition_scope_ref=p["scope"],
        )
        episode = p["episode"]
        props.append(FormationProposalV16(
            basic, sensitivity_hint=p["sensitivity"],
            episode=(EpisodeSemantics(episode[0], episode[1],
                                      tuple(_id(context_id, v) for v in episode[2]),
                                      tuple(_id(context_id, v) for v in episode[3])) if episode else None),
            relationship_counterpart_ref=(_id(context_id, p["counterpart"])
                                          if p["counterpart"] else None),
            mission_target_refs=tuple(_id(context_id, v) for v in p["mission"]),
            workspace_target_refs=tuple(_id(context_id, v) for v in p["workspace"]),
        ))
    decisions = tuple(FormationDisposition(
        scope_ref=name, action=action,
        evidence_refs=tuple(refs[r] for r in source_names),
        target_refs=tuple(target_refs),
    ) for name, action, source_names, *target in spec["dispositions"]
        for target_refs in [target[0] if target else ()])
    bundle = MemoryProposalBundle(scope, authority, _id(case_id, "bundle"),
                                  base.experience_refs, base.content_digest(),
                                  "0" * 64, _id(case_id, "author-target"),
                                  tuple(p.base for p in props), decisions)
    versioned = MemoryProposalBundleV16(bundle, context.content_digest(), tuple(props))
    validate_formation_grounding_v16(context, versioned, tuple(opened))
    expected = {**spec["adjudications"], **ADDITIONAL_ADJUDICATIONS[case_id]}
    if set(expected) != set(DIMENSIONS) or set(expected.values()) - {"present", "negative"}:
        raise ValueError(f"{case_id}: incomplete authored labels")
    actual = {
        "sensitivity": any(p.sensitivity_hint is not None for p in props),
        "episode": any(p.episode is not None for p in props),
        "relationship": any(p.relationship_counterpart_ref is not None for p in props),
        "mission": any(p.mission_target_refs for p in props),
        "workspace": any(p.workspace_target_refs for p in props),
        "correction": any(p.base.kind in {"correction_request", "deletion_request", "revocation_request"} for p in props),
        "source_person": any(p.base.domain == "source_person" or p.base.kind == "source_person_evidence" or p.base.epistemic_status == "source_person_attestation" for p in props),
        "contradiction": any(p.base.kind == "contradiction" or p.base.contradicts for p in props),
        "outcome": any(p.base.kind == "outcome" for p in props),
        "abstention": any(d.action == "abstain" for d in decisions),
    }
    for dimension, present in actual.items():
        if expected[dimension] != ("present" if present else "negative"):
            raise ValueError(f"{case_id}: {dimension} explicit label differs from target")
    output = versioned.output_record()
    return {
        "schema": SCHEMA, "case_id": case_id, "split": split,
        "authorization_id": "owner-directed-mfm-v16-synthetic",
        "context": context.record(),
        "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                    for ref, raw in opened],
        "target": {"schema": TARGET_SCHEMA, "proposals": output["proposals"],
                   "dispositions": output["dispositions"],
                   "adjudications": expected},
        "lineage": {"host_family": spec["family"],
                    "generator_family": GENERATOR_FAMILY,
                    "counterfactual_pair": spec["pair"],
                    "parent_case_ids": list(spec["parent_case_ids"]),
                    "history_id": spec["history"]},
        "status": {"target_author": "openai-codex-synthetic-teacher",
                   "target_review": "single-author-assertion-only",
                   "rights": "pending-independent-authentication",
                   "qualification": "diagnostic-only-unadmitted",
                   "note": spec["note"]},
    }


def render() -> bytes:
    return b"".join((json.dumps(build_one(c), sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False) + "\n").encode("utf-8") for c in CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="compare existing exact bytes")
    args = parser.parse_args()
    raw = render()
    if args.check:
        if args.output.read_bytes() != raw:
            raise SystemExit("v1.6 authoring seed differs from generator")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    counts = {split: sum(c["split"] == split for c in CASES)
              for split in ("train", "development")}
    print(json.dumps({"sha256": sha256(raw).hexdigest(), "bytes": len(raw),
                      "cases": counts, "state": "synthetic-authoring-seed-only"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
