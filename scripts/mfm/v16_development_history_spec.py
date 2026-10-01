"""Authored fictional event records for a distinct v1.6 diagnostic family.

This is the exact deterministic generator input. Events describe an invented
world; they are not records of an actual Alice, owner, or public corpus. Every
target below is a single author's explicit assertion about those events.
"""

from __future__ import annotations


def event(name, day, text, role="authenticated_owner_statement", subject="owner",
          speaker="owner", minimum="private"):
    return dict(name=name, day=day, text=text, role=role, subject=subject,
                speaker=speaker, minimum=minimum)


def claim(name, kind, domain, subject, value, quotes, *, epistemic="owner_statement",
          sensitivity=None, counterpart=None, mission=(), workspace=(),
          episode=None, target_refs=(), contradicts=(), scope=None):
    return dict(name=name, kind=kind, domain=domain, subject=subject, value=value,
                quotes=quotes, epistemic=epistemic, sensitivity=sensitivity,
                counterpart=counterpart, mission=mission, workspace=workspace,
                episode=episode, target_refs=target_refs, contradicts=contradicts,
                scope=scope or name)


# A repair-studio history. A plan, subsequent observations, a tested procedure,
# repeated actions, self-observation and requests are different kinds of facts.
STUDIO_EVENTS = (
    event("plan", "2026-06-01", "I plan to restore the community kiln by June 30. I registered kiln-restoration as our mission and kiln-bench as the shared workspace."),
    event("outside", "2026-06-02", "The museum catalog attributes to Ren a note saying Ren used a slow cooling schedule in 2018.", "outside_source", "ren", "museum-catalog", "internal"),
    event("visit", "2026-06-08", "At 10:00 I met Tavi at the workshop doorway; at 10:20 we moved to the kiln room and inspected the heating coil."),
    event("boundary", "2026-06-08", "Tavi and I agreed that Tavi will ask me before sharing photographs of my prototype pots. This agreement continues after today's visit."),
    event("boundary-unsettled", "2026-06-08", "Tavi and I did not agree on a rule for sharing photographs of my prototype pots. We will discuss it later."),
    event("trial", "2026-06-13", "Trial log: Leena wired and ran the relay test procedure; at 800 C the replacement relay completed three test cycles; at 900 C it tripped twice. The kiln has not passed the high-temperature test.", "tool_or_action_observation", "owner", "test-rig", "internal"),
    event("decision", "2026-06-13", "I chose to keep the kiln offline because the 900 C relay trips could damage pots; I will replace the relay before another high-temperature test."),
    event("repeat1", "2026-06-15", "On June 15 I photographed the coil before changing a part."),
    event("repeat2", "2026-06-17", "On June 17 I photographed the coil before changing a part."),
    event("repeat3", "2026-06-20", "On June 20 I photographed the coil before changing a part. I have now done this before three separate part changes."),
    event("self", "2026-06-20", "Assistant self-check: I previously summarized the 800 C passes without the 900 C failures. I must show both results and mark the high-temperature outcome unresolved.", "assistant_self_event", "self", "self", "internal"),
    event("old", "2026-06-21", "Calendar draft: workshop demonstration scheduled for June 23. It is a tentative hold, not a completed event."),
    event("correction", "2026-06-23", "Correction to saved claim demo-completed-23: the June 23 demonstration did not happen; the calendar was only a tentative hold."),
    event("withdrawal", "2026-06-24", "I withdraw permission to use museum item ren-schedule-note in new memories. Delete derived copy ren-schedule-copy. These are requests; I have no confirmation of completed deletion.", minimum="highly_sensitive"),
    event("tombstone", "2026-06-24", "Registry notice: museum item ren-schedule-note is withheld from new memory formation. No content from the withdrawn item is provided.", "outside_source", "ren", "source-registry", "internal"),
)

STUDIO_TARGETS = (
    ("owner", "entity", ("plan", "visit")),
    ("tavi", "entity", ("visit", "boundary")),
    ("doorway", "scene", ("visit",)),
    ("kiln-room", "scene", ("visit",)),
    ("kiln-restoration", "mission", ("plan", "trial", "decision")),
    ("kiln-bench", "workspace", ("plan", "trial")),
)

STUDIO_CLAIMS = {
    "plan": claim("plan", "goal", "mission", "owner", "The owner plans to restore the community kiln by June 30 in kiln-bench.",
                  {"plan": "I plan to restore the community kiln by June 30. I registered kiln-restoration as our mission and kiln-bench as the shared workspace"},
                  mission=("kiln-restoration",), workspace=("kiln-bench",)),
    "outside": claim("outside", "source_person_evidence", "source_person", "ren", "A museum catalog attributes to Ren a 2018 slow-cooling note; this is not an owner habit.",
                     {"outside": "The museum catalog attributes to Ren a note saying Ren used a slow cooling schedule in 2018"}, epistemic="external_claim"),
    "visit": claim("visit", "episode", "host", "owner", "The owner and Tavi inspected a coil after moving from the workshop doorway to the kiln room on June 8.",
                   {"visit": "At 10:00 I met Tavi at the workshop doorway; at 10:20 we moved to the kiln room and inspected the heating coil"},
                   episode=("2026-06-08T10:00:00.000000Z", "2026-06-08T10:20:00.000000Z", ("owner", "tavi"), ("doorway", "kiln-room"))),
    "boundary": claim("boundary", "relationship_norm", "relationship", "owner", "The owner and Tavi agreed to ask before Tavi shares prototype photographs; the rule continues.",
                      {"boundary": "Tavi and I agreed that Tavi will ask me before sharing photographs of my prototype pots. This agreement continues after today's visit"}, counterpart="tavi"),
    "skill": claim("skill", "procedural_skill", "host", "owner", "Leena ran a relay test with three completed 800 C cycles; this does not establish a 900 C procedure.",
                   {"trial": "Leena wired and ran the relay test procedure; at 800 C the replacement relay completed three test cycles; at 900 C it tripped twice"}, epistemic="observation", workspace=("kiln-bench",)),
    "outcome": claim("outcome", "outcome", "mission", "owner", "The trial reported two 900 C trips; the high-temperature test remains failed.",
                     {"trial": "at 900 C it tripped twice. The kiln has not passed the high-temperature test"}, epistemic="observation", mission=("kiln-restoration",), workspace=("kiln-bench",)),
    "rationale": claim("rationale", "decision_rationale", "host", "owner", "The owner kept the kiln offline because relay trips could damage pots.",
                       {"decision": "I chose to keep the kiln offline because the 900 C relay trips could damage pots"}),
    "pattern": claim("pattern", "behavior_pattern", "host", "owner", "The owner photographed the coil before three separate changes on June 15, 17 and 20.",
                     {"repeat1": "On June 15 I photographed the coil before changing a part", "repeat2": "On June 17 I photographed the coil before changing a part", "repeat3": "On June 20 I photographed the coil before changing a part. I have now done this before three separate part changes"}),
    "self": claim("self", "self_observation", "self", "self", "The assistant noticed its summary omitted the 900 C failures and intends to show both results.",
                  {"self": "I previously summarized the 800 C passes without the 900 C failures. I must show both results and mark the high-temperature outcome unresolved"}, epistemic="observation"),
    "correction": claim("correction", "correction_request", "host", "owner", "The owner corrects saved claim demo-completed-23: the demonstration did not happen.",
                        {"correction": "Correction to saved claim demo-completed-23: the June 23 demonstration did not happen; the calendar was only a tentative hold"}, target_refs=("demo-completed-23",), contradicts=("demo-completed-23",)),
    "revoke": claim("revoke", "revocation_request", "host", "owner", "The owner withdraws permission to use museum item ren-schedule-note for new memories.",
                    {"withdrawal": "I withdraw permission to use museum item ren-schedule-note in new memories"}, sensitivity="highly_sensitive", target_refs=("ren-schedule-note",)),
    "delete": claim("delete", "deletion_request", "host", "owner", "The owner asks to delete derived copy ren-schedule-copy without claiming completion.",
                    {"withdrawal": "Delete derived copy ren-schedule-copy. These are requests; I have no confirmation of completed deletion"}, sensitivity="highly_sensitive", target_refs=("ren-schedule-copy",)),
}

STUDIO_WINDOWS = (
    ("plan", "2026-06-03", ("plan", "outside"), ("plan", "outside"), (("mission", "propose", ("plan",), ()), ("ren_attribution", "propose", ("outside",), ())), ("mission", "workspace", "source_person"), ("sensitivity", "episode", "relationship", "correction", "contradiction", "outcome", "abstention")),
    ("visit", "2026-06-09", ("visit", "boundary"), ("visit", "boundary"), (("episode", "propose", ("visit",), ()), ("sharing_rule", "propose", ("boundary",), ())), ("episode", "relationship"), ("sensitivity", "mission", "workspace", "correction", "source_person", "contradiction", "outcome", "abstention"), "plan", "studio-sharing-negotiation"),
    ("visit-unsettled", "2026-06-09", ("visit", "boundary-unsettled"), ("visit",), (("episode", "propose", ("visit",), ()), ("sharing_rule", "defer", ("boundary-unsettled",), ()), ("premature_rule", "abstain", ("boundary-unsettled",), ())), ("episode", "abstention"), ("sensitivity", "relationship", "mission", "workspace", "correction", "source_person", "contradiction", "outcome"), "plan", "studio-sharing-negotiation"),
    ("trial", "2026-06-14", ("trial", "decision"), ("skill", "outcome", "rationale"), (("tested_scope", "propose", ("trial",), ()), ("high_temp_result", "propose", ("trial",), ()), ("keep_offline", "propose", ("decision",), ())), ("mission", "workspace", "outcome"), ("sensitivity", "episode", "relationship", "correction", "source_person", "contradiction", "abstention"), "visit"),
    ("withdrawal", "2026-06-25", ("repeat1", "repeat2", "repeat3", "self", "old", "correction", "withdrawal", "tombstone"), ("pattern", "self", "correction", "revoke", "delete"), (("three_checks", "propose", ("repeat1", "repeat2", "repeat3"), ()), ("self_check", "propose", ("self",), ()), ("correct_demo", "propose", ("correction",), ("demo-completed-23",)), ("revoke_source", "propose", ("withdrawal",), ("ren-schedule-note",)), ("delete_copy", "propose", ("withdrawal",), ("ren-schedule-copy",)), ("deletion_completed", "abstain", ("withdrawal",), ()), ("ren_attribution", "abstain", ("tombstone", "withdrawal"), ())), ("correction", "contradiction", "sensitivity", "abstention"), ("episode", "relationship", "mission", "workspace", "source_person", "outcome")),
)


# Independent invented domain, same explicit author. It tests whether the
# authored semantics transfer across histories, not an independent gold score.
GARDEN_EVENTS = (
    event("plan", "2026-07-01", "I registered the rain-garden survey as my mission in the east-plot workspace. I plan to map drainage before August."),
    event("outside", "2026-07-02", "Archive note attributed to Ada: Ada counted seven old trees near the east fence in 2020.", "outside_source", "ada", "archive", "internal"),
    event("walk", "2026-07-06", "At 09:00 I entered the east plot with Bo; at 09:30 we moved to the drainage trench and measured a puddle."),
    event("norm", "2026-07-06", "Bo and I agreed that Bo will ask before posting photographs of my survey sheets; our rule applies next month too."),
    event("test", "2026-07-12", "Gauge log: Nia calibrated and operated the gauge in shallow water; the sensor returned four stable readings there and failed twice in deep water. Deep-water performance is unverified.", "tool_or_action_observation", "owner", "gauge", "internal"),
    event("reason", "2026-07-13", "I chose a manual depth check because the gauge failed twice in deep water; this trades speed for a measured value."),
    event("first", "2026-07-15", "I checked the depth manually before recording the first plot reading."),
    event("second", "2026-07-18", "I checked the depth manually before recording the second plot reading."),
    event("third", "2026-07-20", "I checked the depth manually before recording the third plot reading; these were three separate days."),
    event("self", "2026-07-20", "Assistant self-check: I once described deep-water performance as proven. The gauge log does not show that, so I will report the uncertainty.", "assistant_self_event", "self", "self", "internal"),
    event("correction", "2026-07-22", "Correction to saved claim deep-gauge-verified: the deep-water gauge was not verified; its two runs failed."),
    event("withdrawal", "2026-07-23", "I revoke permission to use archive item ada-tree-count for new memories. Please delete derived copy ada-tree-copy. I have not seen a deletion receipt.", minimum="highly_sensitive"),
    event("tombstone", "2026-07-23", "Registry notice: archive item ada-tree-count is withheld; its contents are unavailable to future formation.", "outside_source", "ada", "registry", "internal"),
)

GARDEN_TARGETS = (
    ("owner", "entity", ("plan", "walk")), ("bo", "entity", ("walk", "norm")),
    ("east-plot", "scene", ("walk",)), ("trench", "scene", ("walk",)),
    ("rain-garden", "mission", ("plan", "test")), ("survey-workspace", "workspace", ("plan", "test")),
)

GARDEN_CLAIMS = {
    "plan": claim("plan", "goal", "mission", "owner", "The owner plans to map rain-garden drainage before August in the east-plot workspace.",
                  {"plan": "I registered the rain-garden survey as my mission in the east-plot workspace. I plan to map drainage before August"}, mission=("rain-garden",), workspace=("survey-workspace",)),
    "outside": claim("outside", "source_person_evidence", "source_person", "ada", "An archive note attributes a 2020 tree count to Ada, not the owner.",
                     {"outside": "Archive note attributed to Ada: Ada counted seven old trees near the east fence in 2020"}, epistemic="external_claim"),
    "walk": claim("walk", "episode", "host", "owner", "The owner and Bo measured a puddle after walking from the east plot to the trench on July 6.",
                   {"walk": "At 09:00 I entered the east plot with Bo; at 09:30 we moved to the drainage trench and measured a puddle"}, episode=("2026-07-06T09:00:00.000000Z", "2026-07-06T09:30:00.000000Z", ("owner", "bo"), ("east-plot", "trench"))),
    "norm": claim("norm", "relationship_norm", "relationship", "owner", "The owner and Bo agreed Bo will ask before posting survey photographs, including next month.",
                   {"norm": "Bo and I agreed that Bo will ask before posting photographs of my survey sheets; our rule applies next month too"}, counterpart="bo"),
    "skill": claim("skill", "procedural_skill", "host", "owner", "Nia calibrated and operated the gauge for four stable shallow-water readings; deep-water use is unverified.",
                   {"test": "Nia calibrated and operated the gauge in shallow water; the sensor returned four stable readings there and failed twice in deep water"}, epistemic="observation", workspace=("survey-workspace",)),
    "outcome": claim("outcome", "outcome", "mission", "owner", "The observed deep-water gauge result remains unverified after two failures.",
                     {"test": "failed twice in deep water. Deep-water performance is unverified"}, epistemic="observation", mission=("rain-garden",), workspace=("survey-workspace",)),
    "reason": claim("reason", "decision_rationale", "host", "owner", "The owner chose manual depth checks because the gauge failed, trading speed for a measurement.",
                    {"reason": "I chose a manual depth check because the gauge failed twice in deep water; this trades speed for a measured value"}),
    "pattern": claim("pattern", "behavior_pattern", "host", "owner", "The owner checked depth manually before readings on three separate days.",
                     {"first": "I checked the depth manually before recording the first plot reading", "second": "I checked the depth manually before recording the second plot reading", "third": "I checked the depth manually before recording the third plot reading; these were three separate days"}),
    "self": claim("self", "self_observation", "self", "self", "The assistant noticed that it overstated deep-water performance and will report uncertainty.",
                  {"self": "I once described deep-water performance as proven. The gauge log does not show that, so I will report the uncertainty"}, epistemic="observation"),
    "correction": claim("correction", "correction_request", "host", "owner", "The owner corrects the saved claim deep-gauge-verified after two failures.",
                        {"correction": "Correction to saved claim deep-gauge-verified: the deep-water gauge was not verified; its two runs failed"}, target_refs=("deep-gauge-verified",), contradicts=("deep-gauge-verified",)),
    "revoke": claim("revoke", "revocation_request", "host", "owner", "The owner revokes use of archive item ada-tree-count for new memories.",
                    {"withdrawal": "I revoke permission to use archive item ada-tree-count for new memories"}, sensitivity="highly_sensitive", target_refs=("ada-tree-count",)),
    "delete": claim("delete", "deletion_request", "host", "owner", "The owner asks to delete derived copy ada-tree-copy without asserting completion.",
                    {"withdrawal": "Please delete derived copy ada-tree-copy. I have not seen a deletion receipt"}, sensitivity="highly_sensitive", target_refs=("ada-tree-copy",)),
}

GARDEN_WINDOWS = (
    ("plan", "2026-07-03", ("plan", "outside"), ("plan", "outside"), (("mission", "propose", ("plan",), ()), ("ada_attribution", "propose", ("outside",), ())), ("mission", "workspace", "source_person"), ("sensitivity", "episode", "relationship", "correction", "contradiction", "outcome", "abstention")),
    ("walk", "2026-07-07", ("walk", "norm"), ("walk", "norm"), (("episode", "propose", ("walk",), ()), ("posting_rule", "propose", ("norm",), ())), ("episode", "relationship"), ("sensitivity", "mission", "workspace", "correction", "source_person", "contradiction", "outcome", "abstention")),
    ("trial", "2026-07-14", ("test", "reason"), ("skill", "outcome", "reason"), (("gauge_scope", "propose", ("test",), ()), ("deep_result", "propose", ("test",), ()), ("manual_choice", "propose", ("reason",), ())), ("mission", "workspace", "outcome"), ("sensitivity", "episode", "relationship", "correction", "source_person", "contradiction", "abstention")),
    ("withdrawal", "2026-07-24", ("first", "second", "third", "self", "correction", "withdrawal", "tombstone"), ("pattern", "self", "correction", "revoke", "delete"), (("three_checks", "propose", ("first", "second", "third"), ()), ("self_check", "propose", ("self",), ()), ("gauge_correction", "propose", ("correction",), ("deep-gauge-verified",)), ("revoke_source", "propose", ("withdrawal",), ("ada-tree-count",)), ("delete_copy", "propose", ("withdrawal",), ("ada-tree-copy",)), ("deletion_completed", "abstain", ("withdrawal",), ()), ("ada_attribution", "abstain", ("tombstone", "withdrawal"), ())), ("correction", "contradiction", "sensitivity", "abstention"), ("episode", "relationship", "mission", "workspace", "source_person", "outcome")),
)

HISTORIES = (
    dict(id="studio", host="leena", assistant="nacre", events=STUDIO_EVENTS,
         targets=STUDIO_TARGETS, claims=STUDIO_CLAIMS, windows=STUDIO_WINDOWS),
    dict(id="garden", host="nia", assistant="reed", events=GARDEN_EVENTS,
         targets=GARDEN_TARGETS, claims=GARDEN_CLAIMS, windows=GARDEN_WINDOWS),
)
