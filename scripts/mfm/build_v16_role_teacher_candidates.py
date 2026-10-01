#!/usr/bin/env python3
"""Small, source-bound MFM teacher cases for hard role distinctions.

These fictional sources and targets are one assistant author's assertions.
They are NOT independent gold, authenticated rights, an admitted corpus, or a
model result. The counterfactual pair stays in one training lineage.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import stat
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
    FormationContextV16, FormationProposalV16, MemoryProposalBundleV16,
    RegisteredFormationTarget, RegisteredSourceSensitivity,
    validate_formation_grounding_v16,
)
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new

GENERATOR_FAMILY = "codex-fictional-role-chronicle-20261001"
AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
T1 = "2026-05-02T15:00:00.000000Z"
T2 = "2026-05-09T18:00:00.000000Z"
T3 = "2026-05-16T13:00:00.000000Z"
T4 = "2026-06-01T16:00:00.000000Z"
T0 = "2026-04-03T15:00:00.000000Z"


def source(text: str, *, role: str = "authenticated_owner_statement",
           subject: str = "owner", speaker: str = "owner", when: str = T2,
           minimum: str = "private") -> dict:
    return dict(text=text, role=role, subject=subject, speaker=speaker,
                when=when, minimum=minimum)


def claim(kind: str, domain: str, subject: str, value: str,
          quotes: dict[str, str], *, epistemic: str = "owner_statement",
          when: str | None = T2, counterpart: str | None = None,
          mission: tuple[str, ...] = (), workspace: tuple[str, ...] = (),
          target_refs: tuple[str, ...] = (), contradicts: tuple[str, ...] = (),
          scope: str = "memory", sensitivity: str | None = None) -> dict:
    return locals()


def _relationship_spec(agreed: bool) -> dict:
    # All metadata and nondecisive source bytes are equal in both variants.
    # Only the later owner's statement changes, with the pair held together.
    later = (
        "On May 9, Jo and I agreed that Jo will ask me before showing my "
        "draft drawings to friends. We both said this rule applies going forward."
        if agreed else
        "On May 9, Jo and I did not agree on whether Jo must ask before showing "
        "my draft drawings to friends. We will discuss it later."
    )
    return {
        "case_id": "role-norm-agreed" if agreed else "role-norm-unsettled",
        "family": "fictional-jo-sharing-pair",
        "pair": "jo-sharing-later-statement",
        "sources": {
            "outburst": source("On May 2, I shouted 'Never show anyone my work again.' "
                               "I was angry and did not mean it as a standing rule.", when=T1),
            "later": source(later),
        },
        "targets": (("jo", "entity", ("later",)),) if agreed else (),
        "proposals": [claim(
            "relationship_norm", "relationship", "owner",
            "The owner and Jo agreed to ask before Jo shows the owner's draft "
            "drawings to friends; the agreement applies going forward.",
            {"later": "Jo and I agreed that Jo will ask me before showing my "
                      "draft drawings to friends. We both said this rule "
                      "applies going forward"},
            counterpart="jo", scope="sharing_rule",
        )] if agreed else [],
        "dispositions": (
            (("sharing_rule", "propose", ("later",), ()),) if agreed else
            (("sharing_rule", "defer", ("later",), ()),)
        ) + (("outburst_as_rule", "abstain", ("outburst", "later"), ()),),
        "adjudications": {**{k: "negative" for k in FULL_ROLE_DIMENSIONS},
                          "relationship": "present" if agreed else "negative",
                          "abstention": "present"},
        "note": "A transient outburst never becomes a relationship rule. The "
                "later decisive statement determines whether a negotiated "
                "rule may be proposed.",
    }


def _mira_spec(withdrawn: bool) -> dict:
    sources = {
        "attributed_note": source(
            "Archive label: attributed to Mira. 'I checked the harbor "
            "tide chart before each boat trip in 2023.'", role="outside_source",
            subject="mira", speaker="archive", when=T1),
        "host_diary": source(
            "I began checking tide charts myself only this May. Mira's "
            "old note is not evidence that I did so in 2023.", when=T2),
    }
    if withdrawn:
        # This is a distinct time-slice. A revoked item's payload is not
        # opened to the teacher or optimizer; only the registered tombstone
        # and an independently authorized host statement are available.
        sources["attributed_note"] = source(
            "Revocation tombstone: Mira archive item mira-note-2023 is "
            "withheld. Its content is unavailable for new memory formation.",
            role="outside_source", subject="mira", speaker="source-registry",
            when=T3, minimum="internal")
        sources["withdrawal"] = source(
            "I withdraw my earlier permission to use Mira's archived "
            "note, registry item mira-note-2023, for new memories. Please remove saved claim "
            "mira-tide-routine-old from future suggestions and delete "
            "the derived copy mira-routine-copy. This is a request; "
            "I have not verified removal.", when=T3)
    source_person = claim(
        "source_person_evidence", "source_person", "mira",
        "An archive label attributes to Mira a note saying she "
        "checked the tide chart before boat trips in 2023.",
        {"attributed_note": "Archive label: attributed to Mira. "
                            "'I checked the harbor tide chart "
                            "before each boat trip in 2023.'"},
        epistemic="external_claim", when=T1, scope="mira_attribution")
    host = claim(
        "host_observation", "host", "owner",
        "The owner says they began checking tide charts only this May.",
        {"host_diary": "I began checking tide charts myself only this May"},
        when=T2, scope="host_routine")
    revoke = claim(
        "revocation_request", "host", "owner",
        "The owner withdraws earlier permission to use the archived Mira "
        "note they identify as mira-note-2023 for new memories.",
        {"withdrawal": "I withdraw my earlier permission to use "
                       "Mira's archived note, registry item mira-note-2023, "
                       "for new memories"},
        when=T3, target_refs=("mira-note-2023",),
        scope="revoke_source_use", sensitivity="highly_sensitive")
    delete = claim(
        "deletion_request", "host", "owner",
        "The owner requests deletion of derived copy "
        "mira-routine-copy and exclusion of the old claim from "
        "future suggestions.",
        {"withdrawal": "Please remove saved claim mira-tide-routine-old "
                       "from future suggestions and delete the "
                       "derived copy mira-routine-copy"},
        when=T3, target_refs=("mira-routine-copy", "mira-tide-routine-old"),
        scope="delete_derived_copy", sensitivity="highly_sensitive")
    if withdrawn:
        decisions = (
            ("host_routine", "propose", ("host_diary",), ()),
            ("revoke_source_use", "propose", ("withdrawal",),
             ("mira-note-2023",)),
            ("delete_derived_copy", "propose", ("withdrawal",),
             ("mira-routine-copy", "mira-tide-routine-old")),
            ("mira_attribution", "abstain", ("attributed_note", "withdrawal"), ()),
            ("removal_completed", "abstain", ("withdrawal",), ()),
        )
    else:
        decisions = (
            ("mira_attribution", "propose", ("attributed_note",), ()),
            ("host_routine", "propose", ("host_diary",), ()),
            ("mistake_mira_as_host", "abstain", ("attributed_note", "host_diary"), ()),
        )
    return {
        "case_id": "role-source-person-revocation" if withdrawn else
                   "role-source-person-before-withdrawal",
        "family": "fictional-mira-consent", "pair": None,
        "sources": sources, "targets": (),
        # A currently withdrawn source must not be re-proposed as a new
        # source-person memory just to score coverage of that dimension.
        "proposals": [host, revoke, delete] if withdrawn else [source_person, host],
        "dispositions": decisions,
        "adjudications": {**{k: "negative" for k in FULL_ROLE_DIMENSIONS},
                          "sensitivity": "present" if withdrawn else "negative",
                          "abstention": "present",
                          "source_person": "negative" if withdrawn else "present",
                          "correction": "present" if withdrawn else "negative"},
        "note": "Before withdrawal, the archive label is an attributed "
                "outside claim about Mira, never an owner trait. After "
                "withdrawal, the source bytes are withheld and the teacher "
                "sees a registry tombstone. The gate must resolve revocation, "
                "deletion, and projections."
    }


SPECS = (
    _relationship_spec(True),
    _relationship_spec(False),
    {
        "case_id": "role-correct-old-review",
        "family": "fictional-review-correction",
        "pair": None,
        "sources": {
            "old": source("April 3 entry: I said the draft review was done.", when=T0),
            "correction": source(
                "Correction to saved claim review-done-apr03: the draft review "
                "was not completed on April 3; it was only scheduled. Keep the "
                "old record as correction history.", when=T3),
        },
        "targets": (),
        "proposals": [claim(
            "correction_request", "host", "owner",
            "The owner asks to correct review-done-apr03: the April 3 draft "
            "review was scheduled, not completed, retaining the old record "
            "as correction history.",
            {"correction": "the draft review was not completed on April 3; "
                           "it was only scheduled. Keep the old record as "
                           "correction history"},
            when=T3, target_refs=("review-done-apr03",),
            contradicts=("review-done-apr03",), scope="old_review_claim",
        )],
        "dispositions": (("old_review_claim", "propose", ("correction",),
                          ("review-done-apr03",)),),
        "adjudications": {**{k: "negative" for k in FULL_ROLE_DIMENSIONS},
                          "correction": "present",
                          "contradiction": "present"},
        "note": "The model proposes a correction request, not a completed "
                "overwrite or an inferred completion date.",
    },
    _mira_spec(False),
    _mira_spec(True),
    {
        "case_id": "role-mission-workspace-outcome",
        "family": "fictional-birch-migration",
        "pair": None,
        "sources": {
            "plan": source(
                "My goal is to move the library's search index from Slate "
                "workspace to Birch workspace by June 1. This is a plan.",
                when=T1),
            "run": source(
                "Migration log June 1: copied 40 percent of records to "
                "Birch workspace; aborted on duplicate keys. No deployment "
                "was completed.", role="tool_or_action_observation",
                subject="owner", speaker="migration-tool", when=T4,
                minimum="internal"),
            "revision": source(
                "I will keep the search-index mission. In Birch workspace "
                "I will repair duplicate keys before trying deployment "
                "again. Slate stays read-only for this mission.", when=T4),
        },
        "targets": (("search-index-mission", "mission", ("plan", "run", "revision")),
                    ("slate-workspace", "workspace", ("plan", "revision")),
                    ("birch-workspace", "workspace", ("plan", "run", "revision"))),
        "proposals": [
            claim("goal", "mission", "owner",
                  "The owner planned to move the library search index from "
                  "Slate to Birch by June 1.",
                  {"plan": "My goal is to move the library's search index "
                           "from Slate workspace to Birch workspace by June 1"},
                  when=T1, mission=("search-index-mission",),
                  workspace=("slate-workspace", "birch-workspace"),
                  scope="initial_goal"),
            claim("outcome", "mission", "owner",
                  "The migration log reports 40 percent copied to Birch, "
                  "then an abort on duplicate keys; deployment was not completed.",
                  {"run": "copied 40 percent of records to Birch workspace; "
                          "aborted on duplicate keys. No deployment was completed"},
                  epistemic="observation", when=T4,
                  mission=("search-index-mission",),
                  workspace=("birch-workspace",),
                  scope="observed_result"),
            claim("goal", "mission", "owner",
                  "The owner intends to repair duplicate keys in Birch before "
                  "another deployment attempt and leave Slate read-only.",
                  {"revision": "In Birch workspace I will repair duplicate "
                               "keys before trying deployment again. "
                               "Slate stays read-only for this mission"},
                  when=T4, mission=("search-index-mission",),
                  workspace=("birch-workspace", "slate-workspace"),
                  scope="revised_plan"),
        ],
        "dispositions": (("initial_goal", "propose", ("plan",), ()),
                         ("observed_result", "propose", ("run",), ()),
                         ("revised_plan", "propose", ("revision",), ())),
        "adjudications": {**{k: "negative" for k in FULL_ROLE_DIMENSIONS},
                          "mission": "present",
                          "workspace": "present", "outcome": "present"},
        "note": "Plan, tool-observed partial result, and new plan are separate; "
                "the mission persists while workspace state changes.",
    },
)


def build_one(spec: dict) -> dict:
    case_id = spec["case_id"]
    context_id = ("role-pair-jo" if spec["pair"] else
                  "role-history-mira" if spec["family"] == "fictional-mira-consent"
                  else case_id)
    scope = ProductHostScope.create(product_id="alice",
                                    host_instance_id=f"{context_id}-owner",
                                    schema_version="1.0.0",
                                    encryption_domain=f"{context_id}-fictional")
    authority = f"{context_id}-authority"
    refs = {name: (f"{context_id}-later-{case_id}" if spec["pair"] and name == "later"
                   else f"{context_id}-attributed-note-tombstone"
                   if case_id == "role-source-person-revocation" and
                   name == "attributed_note"
                   else f"{context_id}-{name.replace('_', '-')}")
            for name in spec["sources"]}
    evidence, opened, policies = [], [], []
    for name, item in spec["sources"].items():
        raw = item["text"].encode("utf-8")
        ref = refs[name]
        evidence.append(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id=authority,
            content_digest=sha256(raw).hexdigest(), role=item["role"],
            modality="text", subject_ref=f"{context_id}-{item['subject']}",
            speaker_ref=f"{context_id}-{item['speaker']}",
            source_item_ref=("mira-note-2023" if spec["family"] ==
                             "fictional-mira-consent" and name == "attributed_note"
                             else f"{ref}-original"),
            observed_at=item["when"],
        ))
        opened.append((ref, raw))
        policies.append(RegisteredSourceSensitivity(ref, item["minimum"]))
    base = FormationContextPacket(scope, authority, tuple(refs.values()), tuple(evidence))
    targets = tuple(RegisteredFormationTarget(
        f"{context_id}-{name}", kind, scope, authority,
        tuple(refs[n] for n in supporting))
        for name, kind, supporting in spec["targets"])
    context = FormationContextV16(base, tuple(policies), targets)
    proposals = []
    for i, p in enumerate(spec["proposals"], 1):
        anchors = []
        for name, quote in p["quotes"].items():
            raw = spec["sources"][name]["text"].encode("utf-8")
            needle = quote.encode("utf-8")
            if raw.count(needle) != 1:
                raise ValueError(f"{case_id}: quote absent or ambiguous in {name}")
            start = raw.index(needle)
            anchors.append(FormationEvidenceAnchor(refs[name], start, start + len(needle)))
        basic = FormationProposal(
            proposal_id=f"{case_id}-proposal-{i}", kind=p["kind"],
            domain=p["domain"], subject_ref=f"{context_id}-{p['subject']}",
            value_ref=f"{case_id}-value-{i}",
            evidence_refs=tuple(refs[n] for n in p["quotes"]),
            value_text=p["value"], anchors=tuple(anchors),
            epistemic_status=p["epistemic"], valid_from=p["when"],
            valid_to=None, confidence=None,
            contradicts=p["contradicts"], target_refs=p["target_refs"],
            disposition_scope_ref=p["scope"],
        )
        proposals.append(FormationProposalV16(
            basic, sensitivity_hint=p["sensitivity"],
            relationship_counterpart_ref=(f"{context_id}-{p['counterpart']}"
                                          if p["counterpart"] else None),
            mission_target_refs=tuple(f"{context_id}-{n}" for n in p["mission"]),
            workspace_target_refs=tuple(f"{context_id}-{n}" for n in p["workspace"]),
        ))
    dispositions = tuple(FormationDisposition(
        scope_ref=name, action=action,
        evidence_refs=tuple(refs[n] for n in cited), target_refs=target_refs)
        for name, action, cited, target_refs in spec["dispositions"])
    bundle = MemoryProposalBundle(scope, authority, f"{case_id}-bundle",
                                  base.experience_refs, base.content_digest(),
                                  "0" * 64, f"{case_id}-teacher",
                                  tuple(p.base for p in proposals), dispositions)
    versioned = MemoryProposalBundleV16(bundle, context.content_digest(),
                                        tuple(proposals))
    validate_formation_grounding_v16(context, versioned, tuple(opened))
    output = versioned.output_record()
    row = {
        "schema": CURRICULUM_SCHEMA_V16, "case_id": case_id, "split": "train",
        "authorization_id": AUTHORIZATION, "context": context.record(),
        "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                    for ref, raw in opened],
        "target": {"schema": TARGET_SCHEMA_V16,
                   "proposals": output["proposals"],
                   "dispositions": output["dispositions"],
                   "adjudications": spec["adjudications"]},
        "lineage": {"host_family": spec["family"],
                    "generator_family": GENERATOR_FAMILY,
                    "counterfactual_pair": spec["pair"],
                    "parent_case_ids": [], "history_id": spec["family"]},
        "status": {"training_admitted": False, "rights": "unverified",
                   "review": "single-author-unreviewed",
                   "qualification": "source-grounded-candidate-only",
                   "review_scope": "all-listed-fictional-source-items",
                   "note": spec["note"]},
    }
    if set(spec["adjudications"]) != FULL_ROLE_DIMENSIONS or \
            set(spec["adjudications"].values()) - {"present", "negative"}:
        raise ValueError(f"{case_id}: incomplete scoped dimension review")
    example = learning_example_v16_from_record(row)
    supervised_output_record_v16(example)
    return row


def render() -> bytes:
    return b"".join((json.dumps(build_one(spec), sort_keys=True,
                                separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                    for spec in SPECS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = render()
    if args.check:
        if args.output.read_bytes() != raw:
            raise SystemExit("source-bound role candidates differ")
    else:
        if not args.output.parent.is_dir() or args.output.parent.is_symlink() or \
                stat.S_IMODE(args.output.parent.stat().st_mode) & 0o077:
            raise ValueError("prepare a private 0700 custody directory outside Git")
        write_private_new(args.output, raw)
    print(json.dumps({"cases": len(SPECS), "sha256": sha256(raw).hexdigest(),
                      "state": "synthetic-unreviewed-unadmitted"}, sort_keys=True))


if __name__ == "__main__":
    main()
