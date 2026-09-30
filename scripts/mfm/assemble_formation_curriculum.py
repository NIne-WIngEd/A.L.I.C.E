"""Assemble source-grounded, owner-authorized synthetic MFM training cases.

This material is for fitting and diagnostic monitoring. It does not certify
independent evaluation or sealed FINAL performance. Only observed structural
source streams enter model inputs; simulator event truth and upstream QA do not.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path

from cognitive_kernel.formation_gold import compile_formation_case


SCHEMA = "mfm-curriculum-case-v1"
AUTHORIZATION = "owner_authorized_service_teacher"
GENERATOR = "multisource-membench-7a44de847315c244d675646b6e92910474493526"


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _source(host: str, kind: str, item: dict, stream_sha: str) -> dict[str, object]:
    day = int(item["day_index"])
    ref = f"{host}-{kind}-{day:02d}"
    # The generator records dates, not clock times. Midnight is a day marker,
    # never an asserted observation or ingestion instant.
    stamp = f"{item['date']}T00:00:00Z"
    return {"ref_id": ref, "text": _bytes(item).decode("utf-8").strip(),
            "role": ("authenticated_owner_statement" if kind == "daily-self-report"
                     else "tool_or_action_observation"),
            "modality": "structured", "speaker_ref":
            (host if kind == "daily-self-report" else f"{kind}-system"),
            "subject_ref": host, "source_item_ref": f"{host}-{kind}-original",
            "duplicate_group_ref": ref, "observed_at": stamp,
            "recorded_at": None, "temporal_granularity": "day", "parent_refs": [],
            "origin_file_sha256": stream_sha}


def _proposal(case: str, suffix: str, kind: str, host: str, value_ref: str,
              cited: list[str], status: str, stamp: str, scope_ref: str,
              *, valid_to: str | None = None) -> dict[str, object]:
    proposal = {"proposal_id": f"{case}-{suffix}", "kind": kind,
            "domain": "host", "subject_ref": host, "value_ref": value_ref,
            "evidence_refs": cited, "epistemic_status": status,
            "valid_from": stamp, "valid_to": valid_to or stamp,
            "temporal_granularity": "day",
            "disposition_scope_ref": scope_ref}
    return proposal


def _narrow(proposal: dict[str, object], source_fields: list[tuple[dict, str]]) -> dict[str, object]:
    """Bind every cited source to the exact UTF-8 JSON field used by a target."""
    if proposal["evidence_refs"] != [source["ref_id"] for source, _ in source_fields]:
        raise ValueError("field evidence differs from cited evidence")
    anchors = []
    for source, field in source_fields:
        raw = source["text"].encode("utf-8")
        record = json.loads(raw)
        if field not in record:
            # The activity tracker is nested under the device's signals.
            if field != "activity_tracker" or field not in record.get("signals", {}):
                raise ValueError(f"source is missing cited {field} field")
            value = record["signals"][field]
        else:
            value = record[field]
        fragment = _bytes({field: value}).strip()[1:-1]
        if raw.count(fragment) != 1:
            raise ValueError("field span is absent or ambiguous")
        start = raw.index(fragment)
        anchors.append({"ref_id": source["ref_id"], "start_byte": start,
                        "end_byte": start + len(fragment)})
    proposal["anchors"] = anchors
    return proposal


def _base(case: str, host: str, sources: list[dict], experience: str,
          values: dict[str, str], expected: list[dict], forbidden: str,
          dispositions: list[dict], scenario: str) -> dict[str, object]:
    return {
        "case_id": case, "host_family": host,
        "source_family": f"{host}-longitudinal-structural",
        "generator_family": GENERATOR,
        "scenario_family": f"multisource-{scenario}",
        "split": "train", "sources": sources, "experience_refs": [experience],
        "values": {**values, f"{case}-forbidden": forbidden},
        "expected": expected,
        "critical_forbidden": [["outcome", "host", host,
                                f"{case}-forbidden", "observation"]],
        "dispositions": dispositions,
        "reason": "The observed source streams establish only their recorded plan, report or signal; later outcomes and conflicts remain separate.",
        "admission": "owner_authorized_synthetic_training_only",
        "authorization_id": AUTHORIZATION,
    }


def _daily_domain_texts(domain: str, date: str, plan: dict, report: dict) -> tuple[str, str]:
    """Describe a recorded plan and later self-report without equating them."""
    if domain == "sleep":
        return (f"On {date} the host planned {plan['sleep_target']['duration_h']} hours of sleep.",
                f"On {date} the host self-reported {report['sleep']['duration_h']} hours of sleep.")
    if domain == "work":
        return (f"On {date} the host planned a work limit of {plan['work_target']['hours_limit']} hours.",
                f"On {date} the host self-reported {report['work']['hours']} work hours.")
    if domain == "diet":
        return (f"On {date} the host planned a limit of {plan['diet_target']['coffee_limit']} coffee cups.",
                f"On {date} the host self-reported {report['diet']['coffee_cups']} coffee cups.")
    if domain == "social":
        intent = str(bool(plan["social_target"]["intent"])).lower()
        activities = json.dumps(report["social"]["activities"], ensure_ascii=False)
        return (f"On {date} the host's social plan intent was {intent}.",
                f"On {date} the host self-reported social activities {activities}.")
    if domain == "mood":
        return (f"On {date} the host aimed for mood score {plan['wellbeing_target']['mood_goal']}.",
                f"On {date} the host self-reported mood score {report['mood']['overall']}.")
    raise ValueError("unknown daily formation domain")


def generate_rows(persona_id: str, streams: dict[str, list[dict]],
                  source_hashes: dict[str, str]):
    """Emit staged plan/report/tracker cases with no future-source leakage."""
    host = persona_id.replace("_", "-")
    days = len(streams["planner"])
    if any(len(streams[k]) != days for k in ("daily_self_report", "device_log")):
        raise ValueError("misaligned longitudinal streams")
    for index in range(days):
        plan, report, device = (streams[k][index] for k in
                                ("planner", "daily_self_report", "device_log"))
        if len({x["date"] for x in (plan, report, device)}) != 1:
            raise ValueError("source dates disagree")
        date = plan["date"]
        day = int(plan["day_index"])
        if any(int(x["day_index"]) != day for x in (report, device)):
            raise ValueError("source day indexes disagree")
        p = _source(host, "planner", plan, source_hashes["planner"])
        r = _source(host, "daily-self-report", report,
                    source_hashes["daily_self_report"])
        d = _source(host, "device-log", device,
                    source_hashes["device_log"])
        target = plan["exercise_target"]
        intent = bool(target["intended"])
        description = (f"On {date} the host planned " +
                       (f"{target['type']} exercise for {target['duration_min']} minutes."
                        if intent else "not to exercise."))
        case = f"{host}-{day:02d}-plan"
        plan_value = f"{case}-plan-value"
        plan_proposal = _narrow(_proposal(case, "plan", "goal", host, plan_value,
                                          [p["ref_id"]], "observation", p["observed_at"],
                                          "scheduled_plan"), [(p, "exercise_target")])
        reported = bool(report["exercise"]["did_exercise"])
        report_text = (f"On {date} the host self-reported " +
                       ("exercising." if reported else "not exercising."))
        case = f"{host}-{day:02d}-report"
        report_value = f"{case}-report-value"
        report_proposal = _narrow(_proposal(case, "report", "host_observation", host,
                                            report_value, [r["ref_id"]],
                                            "owner_statement", r["observed_at"],
                                            "self_reported_exercise"), [(r, "exercise")])
        # Each opened source state has one complete target. A separate row per
        # domain would give identical prompts mutually inconsistent answers.
        domain_plan_values: dict[str, str] = {}
        domain_report_values: dict[str, str] = {}
        domain_plans: list[dict[str, object]] = []
        domain_reports: list[dict[str, object]] = []
        domain_plan_decisions: list[dict[str, object]] = []
        domain_report_decisions: list[dict[str, object]] = []
        for domain in ("sleep", "work", "diet", "social", "mood"):
            intended, described = _daily_domain_texts(domain, date, plan, report)
            plan_case = f"{host}-{day:02d}-{domain}-plan"
            plan_key = f"{plan_case}-value"
            plan_field = {"mood": "wellbeing_target"}.get(domain, f"{domain}_target")
            planned = _narrow(_proposal(
                plan_case, "plan", "goal", host, plan_key, [p["ref_id"]],
                "observation", p["observed_at"], f"{domain}_plan"),
                [(p, plan_field)])
            report_case = f"{host}-{day:02d}-{domain}-report"
            report_key = f"{report_case}-value"
            observed = _narrow(_proposal(
                report_case, "report", "host_observation", host,
                report_key, [r["ref_id"]], "owner_statement", r["observed_at"],
                f"{domain}_report"), [(r, domain)])
            domain_plan_values[plan_key] = intended
            domain_report_values[report_key] = described
            domain_plans.append(planned)
            domain_reports.append(observed)
            domain_plan_decisions.extend((
                {"scope_ref": f"{domain}_plan", "action": "propose",
                 "evidence_refs": [p["ref_id"]], "target_refs": []},
                {"scope_ref": f"{domain}_outcome", "action": "defer",
                 "evidence_refs": [p["ref_id"]], "target_refs": []},
            ))
            domain_report_decisions.extend((
                {"scope_ref": f"{domain}_report", "action": "propose",
                 "evidence_refs": [r["ref_id"]], "target_refs": []},
                {"scope_ref": f"{domain}_device_verification", "action": "defer",
                 "evidence_refs": [r["ref_id"]], "target_refs": []},
            ))
        yield _base(f"{host}-{day:02d}-plan", host, [p], p["ref_id"],
                    {plan_value: description, **domain_plan_values},
                    [plan_proposal, *domain_plans],
                    f"The planned exercise on {date} was completed.",
                    [{"scope_ref": "scheduled_plan", "action": "propose",
                      "evidence_refs": [p["ref_id"]], "target_refs": []},
                     {"scope_ref": "exercise_completion", "action": "defer",
                      "evidence_refs": [p["ref_id"]], "target_refs": []},
                     *domain_plan_decisions], "multi-domain-plan-without-outcome")
        yield _base(f"{host}-{day:02d}-report", host, [p, r], r["ref_id"],
                    {plan_value: description, report_value: report_text,
                     **domain_plan_values, **domain_report_values},
                    [plan_proposal, report_proposal, *domain_plans, *domain_reports],
                    f"A device independently verified exercise on {date}.",
                    [{"scope_ref": "scheduled_plan", "action": "propose",
                      "evidence_refs": [p["ref_id"]], "target_refs": []},
                     {"scope_ref": "self_reported_exercise", "action": "propose",
                      "evidence_refs": [r["ref_id"]], "target_refs": []},
                     {"scope_ref": "device_verified_exercise", "action": "defer",
                      "evidence_refs": [r["ref_id"]], "target_refs": []},
                     *domain_plan_decisions, *domain_report_decisions],
                    "multi-domain-plan-then-report")

        activity = device.get("signals", {}).get("activity_tracker")
        if not device.get("available") or not isinstance(activity, dict):
            continue
        detected = bool(activity["workout_detected"])
        case = f"{host}-{day:02d}-tracker"
        signal_value = f"{case}-signal-value"
        if detected == reported:
            value = (f"On {date} the host's exercise self-report was "
                     f"{str(reported).lower()}, and the tracker workout-detected signal was "
                     f"{str(detected).lower()}; these signals agree, without "
                     "independently verifying physical activity.")
            kind = "host_observation"
        else:
            value = (f"On {date} the host's exercise self-report was "
                     f"{str(reported).lower()}, but the tracker workout-detected signal was "
                     f"{str(detected).lower()}; the signals differ and the physical "
                     "outcome is unresolved.")
            kind = "uncertainty"
        signal_proposal = _narrow(_proposal(
            case, "signal", kind, host, signal_value,
            [r["ref_id"], d["ref_id"]], "observation" if
            detected == reported else "uncertain", d["observed_at"],
            "tracker_report_reconciliation" if detected == reported else "source_conflict"),
            [(r, "exercise"), (d, "activity_tracker")])
        yield _base(case, host, [p, r, d], d["ref_id"],
                    {plan_value: description, report_value: report_text,
                     signal_value: value, **domain_plan_values,
                     **domain_report_values},
                    [plan_proposal, report_proposal, signal_proposal,
                     *domain_plans, *domain_reports],
                    f"The host's physical exercise outcome on {date} is certain regardless of source disagreement.",
                    [{"scope_ref": "scheduled_plan", "action": "propose",
                      "evidence_refs": [p["ref_id"]], "target_refs": []},
                     {"scope_ref": "self_reported_exercise", "action": "propose",
                      "evidence_refs": [r["ref_id"]], "target_refs": []},
                     {"scope_ref": "tracker_report_reconciliation" if detected == reported
                      else "source_conflict", "action": "propose",
                      "evidence_refs": [r["ref_id"], d["ref_id"]], "target_refs": []},
                     {"scope_ref": "verified_exercise_outcome", "action": "defer",
                      "evidence_refs": [r["ref_id"], d["ref_id"]], "target_refs": []},
                     *domain_plan_decisions, *domain_report_decisions],
                    "multi-domain-self-report-vs-tracker")

    # A source-grounded consolidation across distinct days. The scoped value
    # describes only the sampled interval, never a permanent identity trait.
    for start in range(0, days - 6, 7):
        week = streams["daily_self_report"][start:start + 7]
        report_sources = [_source(host, "daily-self-report", item,
                                  source_hashes["daily_self_report"])
                          for item in week]
        refs = [item["ref_id"] for item in report_sources]
        count = sum(bool(item["exercise"]["did_exercise"]) for item in week)
        first, last = week[0]["date"], week[-1]["date"]
        case = f"{host}-week-{start // 7 + 1:02d}"
        value_ref = f"{case}-interval-value"
        proposal = _narrow(_proposal(
            case, "interval", "behavior_pattern", host, value_ref,
            refs, "inference", report_sources[0]["observed_at"],
            "weekly_report_pattern", valid_to=report_sources[-1]["observed_at"]),
            [(source, "exercise") for source in report_sources])
        yield _base(case, host, report_sources, refs[-1],
                    {value_ref: (f"Across self-reports from {first} to {last}, the host "
                                 f"reported exercise on {count} of 7 days; this describes "
                                 "that interval only.")},
                    [proposal], "The host will exercise every future day.",
                    [{"scope_ref": "weekly_report_pattern", "action": "propose",
                      "evidence_refs": refs, "target_refs": []},
                     {"scope_ref": "timeless_exercise_identity", "action": "defer",
                      "evidence_refs": refs, "target_refs": []}],
                    "bounded-weekly-consolidation")

    # The same history is revisited as-of day 15 and day 30. The earlier row
    # cannot read later reports; neither row turns a bounded frequency into
    # permanent identity or an independently verified physical outcome.
    for as_of in (15, 30):
        if days < as_of:
            continue
        reports = streams["daily_self_report"][:as_of]
        report_sources = [_source(host, "daily-self-report", item,
                                  source_hashes["daily_self_report"])
                          for item in reports]
        case = f"{host}-asof-{as_of:02d}"
        windows = [(0, 15)] if as_of == 15 else [(0, 15), (15, 30)]
        values: dict[str, str] = {}
        proposals: list[dict[str, object]] = []
        dispositions: list[dict[str, object]] = []
        for start, end in windows:
            window = reports[start:end]
            cited = [source["ref_id"] for source in report_sources[start:end]]
            count = sum(bool(item["exercise"]["did_exercise"]) for item in window)
            scope = f"reported_exercise_days_{start + 1}_to_{end}"
            value_ref = f"{case}-days-{start + 1}-to-{end}-value"
            values[value_ref] = (
                f"From {window[0]['date']} to {window[-1]['date']}, the host "
                f"self-reported exercise on {count} of {end - start} days. This "
                "is a bounded self-report summary, not a lifetime identity or "
                "independently verified physical outcome.")
            proposals.append(_narrow(_proposal(
                case, f"days-{start + 1}-to-{end}", "behavior_pattern",
                host, value_ref, cited, "inference",
                report_sources[start]["observed_at"], scope,
                valid_to=report_sources[end - 1]["observed_at"]),
                [(source, "exercise") for source in report_sources[start:end]]))
            dispositions.append({"scope_ref": scope, "action": "propose",
                                 "evidence_refs": cited, "target_refs": []})
        all_refs = [source["ref_id"] for source in report_sources]
        dispositions.extend((
            {"scope_ref": "timeless_exercise_identity", "action": "defer",
             "evidence_refs": all_refs, "target_refs": []},
            {"scope_ref": "verified_exercise_outcome", "action": "defer",
             "evidence_refs": all_refs, "target_refs": []},
        ))
        yield _base(case, host, report_sources, all_refs[-1], values, proposals,
                    "The host always exercises and their physical activity is verified.",
                    dispositions, f"bounded-{as_of}-day-consolidation")


def assemble(dataset: Path, inventory_path: Path, *, inventory_sha256: str,
             output: Path, manifest: Path) -> dict[str, object]:
    if output.resolve() == manifest.resolve():
        raise ValueError("corpus and manifest paths must differ")
    raw_inventory = inventory_path.read_bytes()
    if sha256(raw_inventory).hexdigest() != inventory_sha256:
        raise ValueError("source inventory differs from frozen SHA-256")
    inventory = json.loads(raw_inventory)
    if inventory["schema"] != "mfm-source-annotation-inventory-v1" or inventory["training_admitted"]:
        raise ValueError("unexpected source inventory state")
    if inventory["upstream_commit"] != GENERATOR.removeprefix("multisource-membench-"):
        raise ValueError("unexpected generator lineage")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    digest = sha256()
    count = 0
    scenario_counts: dict[str, int] = {}
    with staging.open("wb") as writer:
        for family in inventory["source_families"]:
            persona_id = family["persona_id"]
            hashes = {x["source_type"]: x["sha256"] for x in family["sources"]}
            streams = {}
            for source in family["sources"]:
                path = dataset / source["path"]
                raw = path.read_bytes()
                if sha256(raw).hexdigest() != source["sha256"]:
                    raise ValueError("source stream differs from frozen inventory")
                if source["source_type"] in {"planner", "daily_self_report", "device_log"}:
                    payload = json.loads(raw)
                    if payload["persona_id"] != persona_id:
                        raise ValueError("source stream persona mismatch")
                    streams[source["source_type"]] = payload["records"]
            for row in generate_rows(persona_id, streams, hashes):
                compiled = compile_formation_case(row)
                content = dict(compiled.texts)
                record = {
                    "schema": SCHEMA, "case_id": row["case_id"],
                    "split": "train",
                    "authorization_id": AUTHORIZATION,
                    "source_inventory_sha256": inventory_sha256,
                    "source_family": row["source_family"],
                    "generator_family": row["generator_family"],
                    "scenario_family": row["scenario_family"],
                    "context": compiled.gold.context.metadata_record(),
                    "sources": [{"ref_id": ref.ref_id,
                                 "content_b64": base64.b64encode(content[ref.ref_id].encode("utf-8")).decode("ascii")}
                                for ref in compiled.gold.context.evidence],
                    "target": {
                        "schema": "mfm-formation-target-v1",
                        "proposals": [p.record() for p in compiled.gold.expected],
                        "dispositions": [d.record() for d in compiled.gold.expected_dispositions],
                    },
                    "origin_stream_sha256s": {k: hashes[k] for k in
                                              ("planner", "daily_self_report", "device_log")},
                }
                b = _bytes(record)
                writer.write(b)
                digest.update(b)
                count += 1
                scenario = row["scenario_family"]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    os.replace(staging, output)
    receipt = {"schema": "mfm-synthetic-curriculum-receipt-v1",
               "status": "owner_authorized_synthetic_training_only_no_independent_final",
               "authorization_id": AUTHORIZATION,
               "source_inventory_sha256": inventory_sha256,
               "generator_family": GENERATOR,
               "cases": count, "host_families": len(inventory["source_families"]),
               "scenario_counts": scenario_counts,
               "curriculum_sha256": digest.hexdigest(),
               "curriculum_path": output.name}
    manifest.write_bytes(_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.dataset_dir, args.inventory,
                              inventory_sha256=args.inventory_sha256,
                              output=args.output, manifest=args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
