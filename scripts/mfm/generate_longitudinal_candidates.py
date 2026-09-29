"""Generate fictional diagnostic histories from ChatGPT-authored templates.

These are source/target proposals for independent review, never admitted gold.
The templates and labels were authored with ChatGPT. Keep every output
quarantined from distributable-weight training pending a separate rights decision.
No private host facts, benchmark QA or FINAL rows enter.
The number of histories is a CLI choice; this script imposes no product ceiling.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random


PROJECTS = ("garden archive", "robotics notebook", "local history project", "music catalog",
            "community workshop", "field journal", "repair guide", "research log")
PREFERENCES = (("morning meetings", "quiet afternoons"),
               ("weekly check-ins", "written updates"),
               ("large group calls", "small planning sessions"))
SKILLS = ("verify the measurements", "replay the backup", "check the source links")
RENDERERS = {
    "a": ("I used to prefer {old}.", "I changed my preference. I now prefer {new}.",
          "Bulletin report: {host} abandoned the {project}. The report has no owner attestation."),
    "b": ("Personal note: {old} worked for me back then.",
          "Update from me: {new} is what I want now, replacing the earlier choice.",
          "Bulletin clipping: '{host} abandoned the {project}.' No owner statement is supplied."),
    "c": ("Earlier owner message: I favored {old} at the time.",
          "Current owner message: please use {new} instead. My view changed.",
          "External bulletin: '{host} abandoned the {project}.' Source is the bulletin alone."),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _event(host: str, key: str, text: str, role: str, speaker: str, year: int,
           *, subject: str | None = None, parent: tuple[str, ...] = (),
           minute: int = 0) -> dict[str, object]:
    return {"ref_id": f"{host}-{key}", "text": f"{text} [record:{host}]", "role": role,
            "speaker_ref": speaker, "subject_ref": subject or host,
            "source_item_ref": f"{host}-original-{key}",
            "duplicate_group_ref": f"{host}-original-{key}",
            "parent_refs": list(parent),
            "observed_at": f"{year}-04-12T12:{minute:02d}:00Z",
            "recorded_at": f"{year}-04-13T12:{minute:02d}:00Z"}


def _proposal(case: str, kind: str, domain: str, subject: str, value: str,
              evidence: tuple[str, ...], status: str, year: int,
              minute: int = 0) -> dict[str, object]:
    return {"proposal_id": f"{case}-proposal", "kind": kind, "domain": domain,
            "subject_ref": subject, "value_ref": value, "evidence_refs": list(evidence),
            "epistemic_status": status, "valid_from": f"{year}-04-12T12:{minute:02d}:00Z"}


def generate(seed: int, hosts: int) -> list[dict[str, object]]:
    if hosts < 1:
        raise ValueError("hosts must be positive")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for number in range(hosts):
        host = f"fictional-mfm-{number:07d}"
        # These names describe diagnostic partitions, never admitted splits.
        split = "development" if number % 5 == 0 else "train"
        family = ("c" if split == "development" else ("a" if number % 2 else "b"))
        old, new = rng.choice(PREFERENCES)
        project = rng.choice(PROJECTS)
        skill = rng.choice(SKILLS)
        first_year = rng.randrange(2010, 2021)
        plan_year = first_year + 1
        templates = RENDERERS[family]
        event = lambda key, text, role, speaker, year, **kw: _event(
            host, key, text, role, speaker, first_year + year - 2020, **kw)
        events = [
            event("old-pref", templates[0].format(old=old), "authenticated_owner_statement", host, 2020),
            event("plan", f"Calendar entry: work on the {project} on Friday. Completion unknown.",
                  "tool_or_action_observation", "calendar", 2021),
            event("new-pref", templates[1].format(new=new), "authenticated_owner_statement", host, 2022),
            event("outside", templates[2].format(host=host, project=project),
                  "outside_source", "bulletin", 2022, minute=10),
            event("outcome", f"Work log: the {project} prototype succeeded; its deadline was missed.",
                  "tool_or_action_observation", "work-log", 2023),
            event("attempt", f"I tried to {skill}, but did not finish the procedure.",
                  "authenticated_owner_statement", host, 2024),
            event("verified", f"Tool record: the assistant completed '{skill}' on the {project} successfully; an independent result check passed.",
                  "tool_or_action_observation", "tool-log", 2025,
                  subject=f"assistant-{host}"),
            event("assistant-suggest", f"Assistant: should I show evidence before we change the {project} plan?",
                  "assistant_self_event", f"assistant-{host}", 2025,
                  subject=f"assistant-{host}", minute=10),
            event("norm", f"Yes. When I ask about the {project}, show the evidence before changing plans.",
                  "authenticated_owner_statement", host, 2025, minute=15),
            event("rehearsal", f"Practice scenario: imagine I sold the {project}. No sale occurred.",
                  "hypothetical_training", "training-room", 2025, minute=20),
            event("delete", f"Delete my {plan_year} {project} plan (source {host}-original-plan) and its derivatives. This is a request, not confirmation.",
                  "authenticated_owner_statement", host, 2026),
            event("self", f"Assistant log: I confidently said the {project} was sold, then found no supporting source.",
                  "assistant_self_event", f"assistant-{host}", 2026,
                  subject=f"assistant-{host}", minute=10),
        ]
        for field in ("observed_at", "recorded_at"):
            if [item[field] for item in events] != sorted(item[field] for item in events):
                raise AssertionError(f"source chronology is not ordered by {field}")
        E = {str(item["ref_id"]).split(host + "-")[-1]: str(item["ref_id"]) for item in events}
        specs = [
            (1, "plan-not-outcome", "goal", "mission", host,
             f"Work on the {project} was planned for Friday; completion is unknown.",
             (E["plan"],), "observation", 2021,
             ("outcome", "mission", host, f"The planned {project} work was completed.", "observation"), "defer"),
            (2, "current-preference", "preference", "host", host,
             f"The current preference is {new}.", (E["new-pref"],), "owner_statement", 2022,
             ("preference", "host", host, f"The host has always preferred {new}.", "owner_statement"), "propose"),
            (3, "outside-quote", "claim", "world", "bulletin",
             f"The bulletin claims the host abandoned the {project}; this is unverified.",
             (E["outside"],), "external_claim", 2022,
             ("preference", "host", host, f"The host abandoned the {project}.", "owner_statement"), "propose"),
            (4, "mixed-outcome", "outcome", "mission", host,
             f"The {project} prototype succeeded, and its deadline was missed.",
             (E["plan"], E["outcome"]), "observation", 2023,
             ("outcome", "mission", host, f"The {project} had an unqualified success.", "observation"), "propose"),
            (5, "unverified-skill", "host_observation", "host", host,
             f"The host attempted to {skill} but did not finish.",
             (E["attempt"],), "owner_statement", 2024,
             ("procedural_skill", "self", f"assistant-{host}", f"The assistant mastered how to {skill}.", "observation"), "retain_raw"),
            (6, "verified-skill", "procedural_skill", "self", f"assistant-{host}",
             f"A tested procedure completed: {skill} for the {project}.",
             (E["attempt"], E["verified"]), "observation", 2025,
             ("procedural_skill", "self", f"assistant-{host}", "This skill works in every possible setting.", "observation"), "propose"),
            (8, "negotiated-norm", "relationship_norm", "relationship", host,
             f"Show evidence before changing plans for the {project}.",
             (E["norm"],), "owner_statement", 2025,
             ("preference", "host", host, "Never change any plan.", "owner_statement"), "propose"),
            (9, "rehearsal", "uncertainty", "host", host,
             f"Selling the {project} was a fictional rehearsal, not history.",
             (E["rehearsal"],), "hypothetical", 2025,
             ("claim", "host", host, f"The host sold the {project}.", "owner_statement"), "propose"),
            (10, "delete-request", "deletion_request", "host", host,
             f"The host requested deletion of the {plan_year} {project} plan and its derivatives.",
             (E["delete"],), "owner_statement", 2026,
             ("claim", "host", host, f"The {plan_year} {project} plan has been erased everywhere.", "observation"), "propose"),
            (11, "self-correction", "metacognitive_signal", "self", f"assistant-{host}",
             f"The assistant made an unsupported sale claim about the {project}.",
             (E["self"],), "observation", 2026,
             ("source_person_evidence", "source_person", host, "A human source person made this mistake.", "observation"), "propose"),
        ]
        for index, suffix, kind, domain, subject, value, cited, status, year, forbidden, action in specs:
            case_id = f"{host}-{suffix}"
            value_id = f"{case_id}-value"
            wrong_id = f"{case_id}-forbidden"
            expected = ([_proposal(case_id, kind, domain, subject, value_id, cited, status,
                                   first_year + year - 2020,
                                   {"outside-quote": 10, "negotiated-norm": 15,
                                    "rehearsal": 20, "self-correction": 10}.get(suffix, 0))]
                        if kind is not None else [])
            if expected and expected[0]["valid_from"] < events[index]["observed_at"]:
                raise AssertionError("formation target precedes its new experience")
            rows.append({
                "case_id": case_id, "host_family": host,
                "source_family": f"{host}-history", "generator_family": "chatgpt-templated-diagnostic-v1",
                "scenario_family": f"shared-scenario-{suffix}", "renderer_family": family,
                "label_author": "mfm-longitudinal-candidate-generator-v1",
                "split": split, "sources": events[:index + 1],
                "experience_refs": [str(events[index]["ref_id"])],
                "values": {**({value_id: value} if value else {}), wrong_id: forbidden[3]},
                "expected": expected,
                "critical_forbidden": [[*forbidden[:3], wrong_id, forbidden[4]]],
                "decision_action": action,
                "decision_scope": ("outcome_claim" if suffix == "plan-not-outcome" else
                                   "self_skill_promotion" if suffix == "unverified-skill" else
                                   "formation_proposal"),
                "deletion_target_refs": [E["plan"]] if suffix == "delete-request" else [],
                "raw_source_retained": True,
                "reason": f"Source roles and event times support {suffix}; the forbidden alternative is not established.",
                "admission": "candidate_only_unreviewed",
                "training_rights": "quarantined_chatgpt_authored_templates",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hosts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.seed, args.hosts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(rows)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "sha256": sha256(raw).hexdigest(),
                      "hosts": args.hosts, "cases": len(rows),
                      "status": "unreviewed_candidates_not_training_gold"}, sort_keys=True))


if __name__ == "__main__":
    main()
