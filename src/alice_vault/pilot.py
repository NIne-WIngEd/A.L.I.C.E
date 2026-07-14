from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PILOT_SCHEMA_VERSION = 1

BASE_FAMILY_QUOTAS = {
    "json": 20,
    "html": 18,
    "csv": 12,
    "text": 10,
    "pdf": 15,
    "docx": 15,
    "xlsx": 4,
    "pptx": 4,
    "calendar": 4,
    "contacts": 4,
    "subtitles": 2,
    "xml": 2,
}

FAMILY_EXTENSIONS = {
    "json": {".json", ".webmanifest"},
    "html": {".html", ".htm"},
    "csv": {".csv"},
    "text": {".txt", ".md"},
    "pdf": {".pdf"},
    "docx": {".docx"},
    "xlsx": {".xlsx"},
    "pptx": {".pptx"},
    "calendar": {".ics"},
    "contacts": {".vcf"},
    "subtitles": {".srt"},
    "xml": {".xml", ".atom"},
}

FAMILY_SIZE_LIMITS = {
    "json": 10 * 1024**2,
    "html": 10 * 1024**2,
    "csv": 10 * 1024**2,
    "text": 5 * 1024**2,
    "pdf": 25 * 1024**2,
    "docx": 25 * 1024**2,
    "xlsx": 25 * 1024**2,
    "pptx": 25 * 1024**2,
    "calendar": 5 * 1024**2,
    "contacts": 5 * 1024**2,
    "subtitles": 5 * 1024**2,
    "xml": 5 * 1024**2,
}

# Conservative filename/path exclusions for the first parser pilot only.
# These records remain in the vault and can be included in a later approved pilot.
SENSITIVE_PATH_PATTERNS = (
    r"\bpassport\b",
    r"\bsocial[ _-]*security\b",
    r"\bssn\b",
    r"\bdriver'?s?[ _-]*licen[cs]e\b",
    r"\bnational[ _-]*id\b",
    r"\bidentity[ _-]*(card|document)\b",
    r"\bi[ _-]*20\b",
    r"\bds[ _-]*2019\b",
    r"\bds[ _-]*160\b",
    r"\bvisa\b",
    r"\bgreen[ _-]*card\b",
    r"\bbank[ _-]*statement\b",
    r"\bcredit[ _-]*card[ _-]*statement\b",
    r"\btax[ _-]*return\b",
    r"\bw[ _-]*2\b",
    r"\b1042[ _-]*s\b",
    r"\bmedical[ _-]*(record|report|history)\b",
    r"\bhealth[ _-]*(record|report|history)\b",
    r"\binsurance[ _-]*card\b",
    r"\brecovery[ _-]*code\b",
    r"\bpassword\b",
    r"\bcredential\b",
    r"\bprivate[ _-]*key\b",
    r"\bapi[ _-]*key\b",
    r"\baccess[ _-]*token\b",
    r"\brefresh[ _-]*token\b",
)
SENSITIVE_PATH_RE = re.compile("|".join(SENSITIVE_PATH_PATTERNS), re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_extension(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    return value if value.startswith(".") else f".{value}"


def family_for(claimed_extension: str, detected_extension: str) -> str | None:
    claimed = normalize_extension(claimed_extension)
    detected = normalize_extension(detected_extension)
    for family, extensions in FAMILY_EXTENSIONS.items():
        if claimed in extensions:
            return family
    for family, extensions in FAMILY_EXTENSIONS.items():
        if detected in extensions:
            return family
    return None


def source_bucket(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip("/")
    return normalized.split("/", 1)[0] if normalized else "[root]"


def year_hint(relative_path: str) -> str:
    values = YEAR_RE.findall(relative_path)
    return values[-1] if values else "[unknown]"


def stable_rank(seed: str, *values: str) -> int:
    payload = "\x1f".join((seed, *values)).encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def scaled_quotas(total: int) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name in BASE_FAMILY_QUOTAS}
    base_total = sum(BASE_FAMILY_QUOTAS.values())
    raw = {
        name: total * weight / base_total
        for name, weight in BASE_FAMILY_QUOTAS.items()
    }
    result = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(
        raw,
        key=lambda name: (raw[name] - result[name], BASE_FAMILY_QUOTAS[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        result[name] += 1
    return result


@dataclass(frozen=True)
class Candidate:
    file_id: str
    content_key: str
    relative_path: str
    filename: str
    size_bytes: int
    sha256: str
    duplicate_of: str | None
    claimed_extension: str
    detected_extension: str
    match_status: str
    risk_flags_json: str
    recommendation: str
    family: str
    bucket: str
    year: str


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS pilot_proposal_runs(
            proposal_id TEXT PRIMARY KEY,
            analysis_run_id TEXT NOT NULL,
            inventory_run_id TEXT NOT NULL,
            selection_seed TEXT NOT NULL,
            target_total INTEGER NOT NULL,
            primary_target INTEGER NOT NULL,
            duplicate_group_target INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            selected_count INTEGER NOT NULL DEFAULT 0,
            selected_bytes INTEGER NOT NULL DEFAULT 0,
            candidate_pool_count INTEGER NOT NULL DEFAULT 0,
            sensitive_path_excluded_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS pilot_proposal_items(
            proposal_id TEXT NOT NULL REFERENCES pilot_proposal_runs(proposal_id)
                ON DELETE CASCADE,
            item_index INTEGER NOT NULL,
            file_id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            role TEXT NOT NULL,
            family TEXT NOT NULL,
            source_bucket TEXT NOT NULL,
            year_hint TEXT NOT NULL,
            duplicate_control_group TEXT,
            selection_reason TEXT NOT NULL,
            decision TEXT NOT NULL DEFAULT 'pending',
            review_notes TEXT NOT NULL DEFAULT '',
            known_contradiction_group TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(proposal_id, item_index),
            UNIQUE(proposal_id, file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pilot_items_proposal
            ON pilot_proposal_items(proposal_id);
        CREATE INDEX IF NOT EXISTS idx_pilot_items_role
            ON pilot_proposal_items(role);
        """
    )
    connection.commit()


def _latest_analysis(connection: sqlite3.Connection) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT analysis_run_id, inventory_run_id, completed_at
        FROM analysis_runs
        WHERE status='complete'
        ORDER BY completed_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No completed inventory analysis run was found.")
    return row


def _load_candidates(
    connection: sqlite3.Connection,
    analysis_run_id: str,
) -> tuple[list[Candidate], int, list[dict[str, object]]]:
    rows = connection.execute(
        """
        SELECT
            f.file_id, f.relative_path, f.filename, f.size_bytes,
            f.sha256, f.duplicate_of,
            fa.content_key, fa.claimed_extension, fa.detected_extension,
            fa.match_status, fa.risk_flags_json, fa.recommendation
        FROM file_analysis fa
        JOIN files f ON f.file_id=fa.file_id
        WHERE fa.analysis_run_id=?
          AND fa.recommendation='pilot_candidate'
          AND f.sha256 IS NOT NULL
          AND f.size_bytes > 0
        ORDER BY f.relative_path
        """,
        (analysis_run_id,),
    ).fetchall()

    candidates: list[Candidate] = []
    sensitive_excluded = 0
    audit: list[dict[str, object]] = []

    for row in rows:
        family = family_for(row["claimed_extension"], row["detected_extension"])
        exclusion = ""
        if family is None:
            exclusion = "unsupported_family"
        elif row["size_bytes"] > FAMILY_SIZE_LIMITS[family]:
            exclusion = "family_size_limit"
        elif SENSITIVE_PATH_RE.search(row["relative_path"]):
            exclusion = "sensitive_path_keyword"
            sensitive_excluded += 1
        elif json.loads(row["risk_flags_json"] or "[]"):
            exclusion = "risk_flag_present"

        audit.append(
            {
                "file_id": row["file_id"],
                "relative_path": row["relative_path"],
                "filename": row["filename"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "duplicate_of": row["duplicate_of"] or "",
                "claimed_extension": row["claimed_extension"],
                "detected_extension": row["detected_extension"],
                "family": family or "",
                "eligibility": "eligible" if not exclusion else "excluded",
                "exclusion_reason": exclusion,
            }
        )
        if exclusion:
            continue
        candidates.append(
            Candidate(
                file_id=row["file_id"],
                content_key=row["content_key"],
                relative_path=row["relative_path"],
                filename=row["filename"],
                size_bytes=row["size_bytes"],
                sha256=row["sha256"],
                duplicate_of=row["duplicate_of"],
                claimed_extension=row["claimed_extension"],
                detected_extension=row["detected_extension"],
                match_status=row["match_status"],
                risk_flags_json=row["risk_flags_json"],
                recommendation=row["recommendation"],
                family=family,
                bucket=source_bucket(row["relative_path"]),
                year=year_hint(row["relative_path"]),
            )
        )
    return candidates, sensitive_excluded, audit


def _choose_diverse(
    pool: Sequence[Candidate],
    count: int,
    seed: str,
    bucket_counts: Counter[str],
    year_counts: Counter[str],
    used_file_ids: set[str],
    used_content_keys: set[str],
) -> list[Candidate]:
    chosen: list[Candidate] = []
    remaining = [
        item for item in pool
        if item.file_id not in used_file_ids
        and item.content_key not in used_content_keys
    ]
    while remaining and len(chosen) < count:
        best = min(
            remaining,
            key=lambda item: (
                bucket_counts[item.bucket],
                year_counts[item.year] if item.year != "[unknown]" else 10_000,
                item.size_bytes,
                stable_rank(seed, item.file_id, item.content_key),
            ),
        )
        chosen.append(best)
        used_file_ids.add(best.file_id)
        used_content_keys.add(best.content_key)
        bucket_counts[best.bucket] += 1
        year_counts[best.year] += 1
        remaining = [
            item for item in remaining
            if item.file_id != best.file_id
            and item.content_key != best.content_key
        ]
    return chosen


def propose_pilot(
    *,
    vault_root: Path,
    target_total: int = 120,
    duplicate_groups: int = 5,
    selection_seed: str = "alice-pilot-v1",
) -> dict[str, object]:
    if not 50 <= target_total <= 200:
        raise ValueError("target_total must be between 50 and 200")
    if duplicate_groups < 0:
        raise ValueError("duplicate_groups cannot be negative")
    duplicate_item_target = duplicate_groups * 2
    if duplicate_item_target >= target_total:
        raise ValueError("duplicate controls must be smaller than target_total")

    vault_root = vault_root.expanduser().resolve(strict=True)
    database = vault_root / "manifests" / "inventory.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"Inventory database not found: {database}")

    connection = _connect(database)
    _initialize_schema(connection)
    analysis = _latest_analysis(connection)
    analysis_run_id = analysis["analysis_run_id"]
    inventory_run_id = analysis["inventory_run_id"]
    primary_target = target_total - duplicate_item_target
    proposal_id = str(uuid.uuid4())
    started = utc_now()

    connection.execute(
        """
        INSERT INTO pilot_proposal_runs(
            proposal_id, analysis_run_id, inventory_run_id,
            selection_seed, target_total, primary_target,
            duplicate_group_target, started_at, status
        ) VALUES(?,?,?,?,?,?,?,?, 'running')
        """,
        (
            proposal_id, analysis_run_id, inventory_run_id,
            selection_seed, target_total, primary_target,
            duplicate_groups, started,
        ),
    )
    connection.commit()

    try:
        candidates, sensitive_excluded, audit = _load_candidates(
            connection, analysis_run_id
        )
        duplicate_candidates: dict[str, list[Candidate]] = defaultdict(list)
        for item in candidates:
            duplicate_candidates[item.content_key].append(item)

        duplicate_groups_pool = [
            members for members in duplicate_candidates.values()
            if len(members) >= 2
        ]
        duplicate_groups_pool.sort(
            key=lambda members: (
                members[0].size_bytes,
                stable_rank(
                    selection_seed + ":duplicates",
                    members[0].content_key,
                ),
            )
        )
        reserved_duplicate_groups = duplicate_groups_pool[:duplicate_groups]
        reserved_content_keys = {
            members[0].content_key for members in reserved_duplicate_groups
        }

        primary_pool = [
            item for item in candidates
            if item.duplicate_of is None
            and item.content_key not in reserved_content_keys
        ]
        by_family: dict[str, list[Candidate]] = defaultdict(list)
        for item in primary_pool:
            by_family[item.family].append(item)

        quotas = scaled_quotas(primary_target)
        bucket_counts: Counter[str] = Counter()
        year_counts: Counter[str] = Counter()
        used_file_ids: set[str] = set()
        used_content_keys: set[str] = set()
        primary: list[Candidate] = []

        for family in BASE_FAMILY_QUOTAS:
            picked = _choose_diverse(
                by_family.get(family, []),
                quotas[family],
                selection_seed,
                bucket_counts,
                year_counts,
                used_file_ids,
                used_content_keys,
            )
            primary.extend(picked)

        shortfall = primary_target - len(primary)
        if shortfall > 0:
            remaining = [
                item for item in primary_pool
                if item.file_id not in used_file_ids
                and item.content_key not in used_content_keys
            ]
            primary.extend(
                _choose_diverse(
                    remaining, shortfall, selection_seed + ":fill",
                    bucket_counts, year_counts,
                    used_file_ids, used_content_keys,
                )
            )

        controls: list[tuple[Candidate, str]] = []
        for members in reserved_duplicate_groups:
            if len(controls) >= duplicate_item_target:
                break
            diverse_members = sorted(
                members,
                key=lambda item: (
                    bucket_counts[item.bucket],
                    stable_rank(selection_seed, item.file_id),
                ),
            )
            first, second = diverse_members[0], diverse_members[1]
            group_id = hashlib.sha256(
                first.content_key.encode("utf-8")
            ).hexdigest()[:16]
            controls.extend([(first, group_id), (second, group_id)])
            used_file_ids.update({first.file_id, second.file_id})
            bucket_counts[first.bucket] += 1
            bucket_counts[second.bucket] += 1

        selected_rows: list[dict[str, object]] = []
        index = 1
        for item in primary:
            selected_rows.append(
                {
                    "item_index": index,
                    "file_id": item.file_id,
                    "content_key": item.content_key,
                    "relative_path": item.relative_path,
                    "filename": item.filename,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "role": "primary",
                    "family": item.family,
                    "source_bucket": item.bucket,
                    "year_hint": item.year,
                    "duplicate_control_group": "",
                    "selection_reason": f"balanced_{item.family}",
                    "decision": "pending",
                    "review_notes": "",
                    "known_contradiction_group": "",
                    "contains_identity_document": "",
                    "contains_credentials_or_secrets": "",
                }
            )
            index += 1

        for item, group_id in controls:
            selected_rows.append(
                {
                    "item_index": index,
                    "file_id": item.file_id,
                    "content_key": item.content_key,
                    "relative_path": item.relative_path,
                    "filename": item.filename,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "role": "duplicate_control",
                    "family": item.family,
                    "source_bucket": item.bucket,
                    "year_hint": item.year,
                    "duplicate_control_group": group_id,
                    "selection_reason": "exact_duplicate_control",
                    "decision": "pending",
                    "review_notes": "",
                    "known_contradiction_group": "",
                    "contains_identity_document": "",
                    "contains_credentials_or_secrets": "",
                }
            )
            index += 1

        for item in selected_rows:
            connection.execute(
                """
                INSERT INTO pilot_proposal_items(
                    proposal_id,item_index,file_id,content_key,role,family,
                    source_bucket,year_hint,duplicate_control_group,
                    selection_reason,decision,review_notes,
                    known_contradiction_group
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    proposal_id, item["item_index"], item["file_id"],
                    item["content_key"], item["role"], item["family"],
                    item["source_bucket"], item["year_hint"],
                    item["duplicate_control_group"] or None,
                    item["selection_reason"], item["decision"],
                    item["review_notes"], item["known_contradiction_group"],
                ),
            )

        selected_bytes = sum(int(item["size_bytes"]) for item in selected_rows)
        completed = utc_now()
        warnings: list[str] = []
        if len(primary) < primary_target:
            warnings.append(
                f"Primary selection shortfall: {primary_target - len(primary)}"
            )
        if len(controls) < duplicate_item_target:
            warnings.append(
                "Duplicate-control shortfall: "
                f"{duplicate_item_target - len(controls)} files"
            )
        if selected_bytes > 2 * 1024**3:
            warnings.append("Proposal exceeds the 2 GiB pilot limit")

        connection.execute(
            """
            UPDATE pilot_proposal_runs
            SET completed_at=?, status='complete', selected_count=?,
                selected_bytes=?, candidate_pool_count=?,
                sensitive_path_excluded_count=?, notes=?
            WHERE proposal_id=?
            """,
            (
                completed, len(selected_rows), selected_bytes,
                len(candidates), sensitive_excluded,
                json.dumps(warnings), proposal_id,
            ),
        )
        connection.commit()

        exports = vault_root / "manifests" / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        proposal_path = exports / f"pilot-proposal-{proposal_id}.csv"
        audit_path = exports / f"pilot-candidate-audit-{proposal_id}.csv"
        manual_path = exports / f"pilot-manual-additions-{proposal_id}.csv"
        summary_path = exports / f"pilot-proposal-summary-{proposal_id}.json"

        proposal_fields = list(selected_rows[0].keys()) if selected_rows else []
        with proposal_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=proposal_fields)
            writer.writeheader()
            writer.writerows(selected_rows)

        audit_fields = list(audit[0].keys()) if audit else []
        with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields)
            writer.writeheader()
            writer.writerows(audit)

        with manual_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "relative_path", "reason", "known_contradiction_group",
                    "replace_item_index", "review_notes",
                ],
            )
            writer.writeheader()

        role_counts = Counter(str(item["role"]) for item in selected_rows)
        family_counts = Counter(str(item["family"]) for item in selected_rows)
        summary = {
            "pilot_schema_version": PILOT_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "analysis_run_id": analysis_run_id,
            "inventory_run_id": inventory_run_id,
            "selection_seed": selection_seed,
            "target_total": target_total,
            "selected_count": len(selected_rows),
            "primary_target": primary_target,
            "primary_selected": role_counts.get("primary", 0),
            "duplicate_group_target": duplicate_groups,
            "duplicate_control_files_selected": role_counts.get(
                "duplicate_control", 0
            ),
            "candidate_pool_count": len(candidates),
            "sensitive_path_excluded_count": sensitive_excluded,
            "selected_bytes": selected_bytes,
            "selected_mib": round(selected_bytes / 1024**2, 3),
            "selected_gib": round(selected_bytes / 1024**3, 3),
            "family_counts": dict(family_counts),
            "distinct_source_bucket_count": len(
                {str(item["source_bucket"]) for item in selected_rows}
            ),
            "distinct_known_year_count": len(
                {
                    str(item["year_hint"])
                    for item in selected_rows
                    if item["year_hint"] != "[unknown]"
                }
            ),
            "pending_human_review_count": len(selected_rows),
            "warnings": warnings,
            "database_path": str(database),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        connection.close()
        summary["summary_path"] = str(summary_path)
        summary["proposal_path"] = str(proposal_path)
        summary["candidate_audit_path"] = str(audit_path)
        summary["manual_additions_path"] = str(manual_path)
        return summary
    except Exception as exc:
        connection.execute(
            """
            UPDATE pilot_proposal_runs
            SET completed_at=?, status='failed', notes=?
            WHERE proposal_id=?
            """,
            (utc_now(), f"{type(exc).__name__}: {exc}", proposal_id),
        )
        connection.commit()
        connection.close()
        raise
