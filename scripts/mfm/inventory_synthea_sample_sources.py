#!/usr/bin/env python3
"""Inventory an exact public Synthea sample archive as source candidates only.

The inventory contains metadata and digests, never patient payloads, formation
targets, permission attestations, or an alleged independently held-out split.
The original ZIP remains in private/local source custody outside this repo.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile


SCHEMA = "mfm-synthea-fhir-source-inventory-v1"
ARCHIVE_REL = "downloads/synthea_sample_data_fhir_r4_nov2021.zip"
SAMPLE_REPO = "https://github.com/synthetichealth/synthea-sample-data"
GENERATOR_REPO = "https://github.com/synthetichealth/synthea"
DATE = re.compile(r"^\d{4}-\d\d-\d\d(?:T\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:\d\d)?)?$")
DATE_IN_TEXT = re.compile(r"(?<!\d)(\d{4}-\d\d-\d\d)(?!\d)")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


def _git_revision(repo: Path, expected: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{40}", expected):
        raise ValueError("a full 40-character commit pin is required")
    revision = _git(repo, "rev-parse", "HEAD")
    if revision != expected or _git(repo, "status", "--porcelain"):
        raise ValueError("pinned upstream checkout is wrong or dirty")
    return revision


def _event_date(resource: dict) -> list[str]:
    """A descriptive source inventory, never a clinical or memory judgment."""
    found = []
    for key in ("effectiveDateTime", "authoredOn", "occurrenceDateTime",
                "recordedDate", "onsetDateTime", "performedDateTime",
                "issued", "date"):
        value = resource.get(key)
        if isinstance(value, str) and DATE.fullmatch(value):
            found.append(value[:10])
    for key in ("effectivePeriod", "occurrencePeriod", "onsetPeriod",
                "performedPeriod", "period"):
        period = resource.get(key)
        if isinstance(period, dict):
            for end in ("start", "end"):
                value = period.get(end)
                if isinstance(value, str) and DATE.fullmatch(value):
                    found.append(value[:10])
    return found


def _dated_strings(value: object) -> list[str]:
    """Conservatively reject ISO date strings even inside narrative fields."""
    if isinstance(value, str):
        return DATE_IN_TEXT.findall(value)
    if isinstance(value, list):
        return [day for child in value for day in _dated_strings(child)]
    if isinstance(value, dict):
        return [day for child in value.values() for day in _dated_strings(child)]
    return []


def _entry_span(raw: bytes, ordinal: int, entries: list) -> tuple[int, int]:
    """Locate an exact original-byte JSON entry, without reserializing FHIR."""
    if type(ordinal) is not int or not 0 <= ordinal < len(entries):
        raise ValueError("entry ordinal outside pinned Bundle")
    # The pinned FHIR exporter has the Bundle's entry array in its root header.
    # Restrict the search to that header so nested keys cannot be substituted.
    match = re.search(rb'"entry"\s*:\s*\[', raw[:256])
    if match is None:
        raise ValueError("entry array absent from FHIR Bundle header")
    pos = match.end()
    for index in range(ordinal + 1):
        while raw[pos] in b" \t\r\n":
            pos += 1
        start = pos
        if raw[pos] != ord("{"):
            raise ValueError("FHIR entry must be a JSON object")
        depth = 0
        quoted = escaped = False
        while pos < len(raw):
            byte = raw[pos]
            pos += 1
            if quoted:
                if escaped:
                    escaped = False
                elif byte == ord("\\"):
                    escaped = True
                elif byte == ord('"'):
                    quoted = False
            elif byte == ord('"'):
                quoted = True
            elif byte == ord("{"):
                depth += 1
            elif byte == ord("}"):
                depth -= 1
                if depth == 0:
                    break
        if depth != 0 or quoted:
            raise ValueError("unbalanced original FHIR entry")
        if json.loads(raw[start:pos]) != entries[index]:
            raise ValueError("FHIR entry slice differs from parsed original")
        if index == ordinal:
            return start, pos
        while raw[pos] in b" \t\r\n":
            pos += 1
        if raw[pos] != ord(","):
            raise ValueError("FHIR entry separator differs")
        pos += 1
    raise AssertionError("unreachable")


def isolated_resource(archive: Path, inventory_path: Path, *,
                      expected_inventory_sha256: str, host_family: str,
                      entry_ordinal: int, through_date: str) -> tuple[dict, bytes]:
    """Return one as-of source entry with its original-byte provenance.

    This narrow source-only read is not a model target. It refuses patient
    profiles, undated entries and any date-looking value after the window.
    It makes no promise that clinical content is accurate or rights-cleared.
    """
    raw_inventory = inventory_path.read_bytes()
    if _sha(raw_inventory) != expected_inventory_sha256:
        raise ValueError("source inventory differs from external digest pin")
    inventory = json.loads(raw_inventory)
    if inventory.get("schema") != SCHEMA or inventory.get("training_admitted") is not False:
        raise ValueError("source inventory does not carry unadmitted provenance")
    if _file_sha(archive) != inventory.get("archive_sha256"):
        raise ValueError("Synthea ZIP differs from inventoried source")
    if not re.fullmatch(r"\d{4}-\d\d-\d\d", through_date):
        raise ValueError("through date must be an ISO calendar date")
    matches = [host for host in inventory["source_families"]
               if host["host_family"] == host_family]
    if len(matches) != 1:
        raise ValueError("host absent or duplicated in inventory")
    host = matches[0]
    with ZipFile(archive) as z:
        raw = z.read(host["member"])
    if _sha(raw) != host["member_sha256"]:
        raise ValueError("ZIP member differs from inventoried exact bytes")
    document = json.loads(raw)
    if document.get("resourceType") != "Bundle" or not isinstance(document.get("entry"), list):
        raise ValueError("invalid source Bundle")
    entries = document["entry"]
    start, end = _entry_span(raw, entry_ordinal, entries)
    resource = entries[entry_ordinal]["resource"]
    kind = resource["resourceType"]
    if kind not in ("Encounter", "Observation"):
        raise ValueError("only dated encounter/observation entries are opened")
    dates = _dated_strings(entries[entry_ordinal])
    if not dates or any(day > through_date for day in dates):
        raise ValueError("undated or future-bearing source entry")
    fragment = raw[start:end]
    receipt = {"schema": "mfm-synthea-source-slice-v1",
               "status": "source-only-unreviewed-unadmitted",
               "training_admitted": False, "rights_status": "unverified",
               "inventory_sha256": expected_inventory_sha256,
               "host_family": host_family, "source_member": host["member"],
               "parent_member_sha256": host["member_sha256"],
               "original_byte_anchor": {"start": start, "end": end},
               "entry_ordinal": entry_ordinal, "resource_type": kind,
               "through_date": through_date, "observed_dates": sorted(set(dates)),
               "source_sha256": _sha(fragment)}
    return receipt, fragment


def inventory_archive(archive: Path, *, expected_archive_sha256: str,
                      sample_commit: str, generator_commit: str,
                      code_license_sha256: str, notice_sha256: str) -> dict:
    if _file_sha(archive) != expected_archive_sha256:
        raise ValueError("Synthea sample ZIP differs from external digest pin")
    hosts = []
    totals: Counter[str] = Counter()
    resource_dates = []
    excluded = []
    seen_ids = set()
    with ZipFile(archive) as z:
        infos = z.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("duplicate ZIP member name")
        for info in sorted(infos, key=lambda item: item.filename):
            name = info.filename
            if name.endswith("/"):
                continue
            if (not name.startswith("fhir/") or not name.endswith(".json") or
                    ".." in Path(name).parts or info.file_size > 100_000_000):
                raise ValueError("unrecognized or oversized ZIP member")
            raw = z.read(info)
            if len(raw) != info.file_size:
                raise ValueError("ZIP member size mismatch")
            document = json.loads(raw)
            if (not isinstance(document, dict) or
                    document.get("resourceType") != "Bundle" or
                    not isinstance(document.get("entry"), list)):
                raise ValueError("expected FHIR transaction Bundle")
            entries = document["entry"]
            types = Counter(entry.get("resource", {}).get("resourceType")
                            for entry in entries)
            if None in types:
                raise ValueError("FHIR entry lacks resource type")
            # The two global metadata bundles are not personal histories.
            patient_entries = [entry["resource"] for entry in entries
                               if entry["resource"]["resourceType"] == "Patient"]
            if not patient_entries:
                excluded.append({"member": name, "sha256": _sha(raw),
                                 "reason": "no Patient resource"})
                continue
            if len(patient_entries) != 1:
                raise ValueError("expected exactly one patient per member")
            patient_id = patient_entries[0].get("id")
            if not isinstance(patient_id, str) or not patient_id or patient_id in seen_ids:
                raise ValueError("missing or repeated patient ID")
            seen_ids.add(patient_id)
            dates = [day for entry in entries for day in _event_date(entry["resource"])]
            resource_dates.extend(dates)
            totals.update(types)
            hosts.append({"host_family": "synthea-patient-" + _sha(patient_id.encode())[:24],
                          "generator_family": "synthea-nov2021-fhir-r4-sample",
                          "member": name, "member_sha256": _sha(raw),
                          "member_uncompressed_bytes": len(raw),
                          "resource_types": dict(sorted(types.items())),
                          "event_date_min": min(dates) if dates else None,
                          "event_date_max": max(dates) if dates else None})
    return {
        "schema": SCHEMA, "status": "public_source_candidate_not_admitted",
        "training_admitted": False, "formation_targets": False,
        "independent_development_or_final": False,
        "upstream_sample_repository": SAMPLE_REPO,
        "upstream_sample_commit": sample_commit,
        "archive_path_in_upstream": ARCHIVE_REL,
        "archive_sha256": expected_archive_sha256,
        "upstream_generator_repository": GENERATOR_REPO,
        "upstream_generator_source_commit_inspected": generator_commit,
        "generator_revision_used_for_archived_sample": "unverified",
        "upstream_generator_apache2_license_sha256": code_license_sha256,
        "upstream_generator_notice_sha256": notice_sha256,
        "archive_rights": "requires_steward_review_of_FHIR_and_terminologies",
        "rights_evidence_urls": [
            "https://synthetichealth.github.io/downloads.html",
            "https://github.com/synthetichealth/synthea/discussions/1167"],
        "source_modality": "structured_text_json_only",
        "patient_bundles": len(hosts), "excluded_nonpatient_bundles": excluded,
        "resource_type_counts": dict(sorted(totals.items())),
        "dated_resource_marker_count": len(resource_dates),
        "event_date_min": min(resource_dates) if resource_dates else None,
        "event_date_max": max(resource_dates) if resource_dates else None,
        "source_families": hosts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-repo", required=True, type=Path)
    parser.add_argument("--sample-commit", required=True)
    parser.add_argument("--generator-repo", required=True, type=Path)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sample = args.sample_repo.resolve()
    generator = args.generator_repo.resolve()
    _git_revision(sample, args.sample_commit)
    _git_revision(generator, args.generator_commit)
    if not args.output.parent.exists() or args.output.exists():
        raise ValueError("write into an existing source-custody directory at a new path")
    result = inventory_archive(
        sample / ARCHIVE_REL, expected_archive_sha256=args.archive_sha256,
        sample_commit=args.sample_commit, generator_commit=args.generator_commit,
        code_license_sha256=_file_sha(generator / "LICENSE"),
        notice_sha256=_file_sha(generator / "NOTICE"))
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    print(json.dumps({"inventory_sha256": _sha(raw),
                      "patient_bundles": result["patient_bundles"],
                      "resource_type_counts": result["resource_type_counts"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
