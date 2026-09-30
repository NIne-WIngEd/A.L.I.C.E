"""Make unreviewed, source-only history windows from a frozen Multi-Source inventory.

The JSONL is an annotation intake. It contains no targets, reviewer decisions,
rights attestations, simulator event tables, upstream QA, or split assignments.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path


DAILY_TYPES = ("planner", "daily_self_report", "objective_log", "device_log")
SOURCE_TYPES = frozenset((*DAILY_TYPES, "profile_ltm"))
DAILY_FIELDS = {
    "planner": {"date", "day_index", "sleep_target", "exercise_target",
                "work_target", "diet_target", "social_target", "wellbeing_target",
                "persona_context"},
    "daily_self_report": {"date", "day_index", "sleep", "exercise", "diet",
                          "work", "social", "mood"},
    "objective_log": {"date", "day_index", "available", "signals"},
    "device_log": {"date", "day_index", "available", "signals"},
}
FORBIDDEN_KEYS = {"event_table", "ground_truth", "generation_metadata",
                  "source_knobs", "difficulty_type", "question", "answer",
                  "qa", "target", "expected_output", "adjudication"}
DEFAULT_DAYS = (1, 7, 15, 30)


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(raw: bytes) -> dict:
    result = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _check_view(value: object) -> None:
    if isinstance(value, dict):
        if any(key.casefold() in FORBIDDEN_KEYS for key in value):
            raise ValueError("forbidden simulator or label field in source view")
        for child in value.values():
            _check_view(child)
    elif isinstance(value, list):
        for child in value:
            _check_view(child)


def _source_path(root: Path, host: str, source: dict) -> Path:
    kind = source.get("source_type")
    expected = f"{host}/structural_sources/{kind}.json"
    if kind not in SOURCE_TYPES or source.get("path") != expected:
        raise ValueError("inventory source is outside the five structural streams")
    path = (root / expected).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path escapes dataset root")
    return path


def _checked_sources(root: Path, family: dict) -> tuple[dict, int]:
    host = family["host_family"]
    if (not isinstance(host, str) or not host.startswith("bench_") or
            host in ("bench_", ".", "..") or "/" in host or "\\" in host or
            family.get("persona_id") != host):
        raise ValueError("invalid host family")
    items = family.get("sources")
    if not isinstance(items, list) or len(items) != len(SOURCE_TYPES):
        raise ValueError("expected five inventoried source streams")
    by_type = {item.get("source_type"): item for item in items}
    if set(by_type) != SOURCE_TYPES:
        raise ValueError("missing or duplicate source type")

    opened = {}
    dates = None
    total = None
    for kind in (*DAILY_TYPES, "profile_ltm"):
        source = by_type[kind]
        raw = _source_path(root, host, source).read_bytes()
        if sha256(raw).hexdigest() != source.get("sha256"):
            raise ValueError(f"source SHA-256 differs: {host}/{kind}")
        doc = _load(raw)
        if (doc.get("schema_version") != "v2.source_projection" or
                doc.get("persona_id") != host or doc.get("source_type") != kind):
            raise ValueError("source schema, host or type differs")
        units = doc.get("facts") if kind == "profile_ltm" else doc.get("records")
        if not isinstance(units, dict if kind == "profile_ltm" else list) or \
                len(units) != source.get("units"):
            raise ValueError("source units differ from inventory")
        if kind == "profile_ltm":
            anchor = doc.get("anchor_window")
            if (set(units) != {"identity", "traits", "routine_snapshot"} or
                    not isinstance(anchor, dict) or
                    set(anchor) != {"start_day_index", "end_day_index",
                                    "total_days", "staleness_days"} or
                    any(type(v) is not int for v in anchor.values()) or
                    anchor["start_day_index"] != 1 or
                    not 1 <= anchor["end_day_index"] <= anchor["total_days"] or
                    anchor["staleness_days"] != anchor["total_days"] - anchor["end_day_index"]):
                raise ValueError("invalid profile anchor window")
            view = {"facts": units, "anchor_window": anchor}
            _check_view(view)
            opened[kind] = (source, view)
            continue

        if not units:
            raise ValueError("empty daily source")
        seen_dates = []
        for index, record in enumerate(units, start=1):
            if not isinstance(record, dict) or set(record) != DAILY_FIELDS[kind] or \
                    type(record["day_index"]) is not int or record["day_index"] != index:
                raise ValueError("daily source shape or day order differs")
            try:
                day = date.fromisoformat(record["date"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid daily source date") from exc
            if day.isoformat() != record["date"]:
                raise ValueError("noncanonical daily source date")
            _check_view(record)
            seen_dates.append(day)
        if any(later - earlier != timedelta(days=1)
               for earlier, later in zip(seen_dates, seen_dates[1:])):
            raise ValueError("nonconsecutive daily source dates")
        if dates is not None and seen_dates != dates:
            raise ValueError("daily source dates differ across streams")
        dates = seen_dates
        if total is not None and total != len(units):
            raise ValueError("daily source lengths differ")
        total = len(units)
        opened[kind] = (source, units)
    if opened["profile_ltm"][1]["anchor_window"]["total_days"] != total:
        raise ValueError("profile horizon differs from daily streams")
    return opened, total


def build_packets(dataset: Path, inventory_path: Path, *,
                  expected_inventory_sha256: str, days: tuple[int, ...] = DEFAULT_DAYS,
                  hosts: tuple[str, ...] = ()) -> bytes:
    """Return deterministic JSONL; each row is one cumulative host/day view."""
    root = dataset.resolve()
    raw = inventory_path.read_bytes()
    inventory_sha = sha256(raw).hexdigest()
    if inventory_sha != expected_inventory_sha256:
        raise ValueError("inventory differs from expected SHA-256")
    inventory = _load(raw)
    if (inventory.get("schema") != "mfm-source-annotation-inventory-v1" or
            inventory.get("status") != "source_candidate_only_no_formation_labels_or_review" or
            inventory.get("training_admitted") is not False):
        raise ValueError("expected unreviewed source inventory")
    if not days or any(type(day) is not int or day < 1 for day in days) or \
            len(set(days)) != len(days):
        raise ValueError("days must be unique positive indices")
    families = inventory.get("source_families")
    if not isinstance(families, list) or not families:
        raise ValueError("empty source inventory")
    family_by_host = {family["host_family"]: family for family in families}
    if len(family_by_host) != len(families):
        raise ValueError("duplicate host family")
    requested = set(hosts)
    if len(requested) != len(hosts) or requested - family_by_host.keys():
        raise ValueError("requested host is duplicated or absent from inventory")

    rows = []
    for host in sorted(requested or family_by_host):
        family = family_by_host[host]
        opened, total = _checked_sources(root, family)
        if max(days) > total:
            raise ValueError(f"requested day exceeds source horizon: {host}")
        generator = family["generator_family"]
        if not isinstance(generator, str) or not generator:
            raise ValueError("missing generator family")
        for through_day in sorted(days):
            evidence = []
            for day in range(1, through_day + 1):
                for kind in DAILY_TYPES:
                    source, records = opened[kind]
                    record = records[day - 1]
                    evidence.append({
                        "source_id": f"{host}/{kind}/day-{day:02d}",
                        "source_type": kind, "source_path": source["path"],
                        "source_file_sha256": source["sha256"],
                        "source_locator": f"/records/{day - 1}",
                        "unit_sha256": sha256(_canonical(record)).hexdigest(),
                        "observed_day_index": day,
                        "observed_date": record["date"],
                        "record": record,
                    })
            profile_source, profile = opened["profile_ltm"]
            # The profile has whole-run fields, even when its anchor ended early.
            if through_day >= profile["anchor_window"]["total_days"]:
                evidence.append({
                    "source_id": f"{host}/profile_ltm",
                    "source_type": "profile_ltm", "source_path": profile_source["path"],
                    "source_file_sha256": profile_source["sha256"],
                    "source_locator": "/facts",
                    "anchor_locator": "/anchor_window",
                    "unit_sha256": sha256(_canonical(profile)).hexdigest(),
                    "record": profile,
                })
            row = {
                "schema": "mfm-v16-source-review-packet-v1",
                "status": "candidate_unreviewed", "training_admitted": False,
                "packet_id": f"{inventory_sha[:16]}/{host}/through-day-{through_day:02d}",
                "inventory_sha256": inventory_sha,
                "window": {"start_day_index": 1, "end_day_index": through_day},
                "lineage": {"history_id": host, "host_family": host,
                            "generator_family": generator,
                            "lineage_group": f"{generator}/{host}",
                            "generator_seed": inventory["seed"],
                            "upstream_commit": inventory["upstream_commit"]},
                "sources": evidence,
            }
            rows.append(_canonical(row) + b"\n")
    return b"".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--host-family", action="append", default=[])
    parser.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = build_packets(args.dataset_dir, args.inventory,
                        expected_inventory_sha256=args.expected_inventory_sha256,
                        days=tuple(args.days), hosts=tuple(args.host_family))
    if args.output.exists() and args.output.read_bytes() != raw:
        parser.error("existing output differs; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"output_sha256": sha256(raw).hexdigest(),
                      "packets": raw.count(b"\n"), "status": "candidate_unreviewed"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
