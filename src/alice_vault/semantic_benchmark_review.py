from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .semantic_retrieval import (
    _load_local_model,
    hybrid_search,
    load_semantic_policy,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _preview(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 2].rstrip() + " …"


def _candidate(item: dict[str, Any], limit: int) -> dict[str, Any]:
    provenance = list(item.get("provenance", []))
    return {
        "rank": int(item["rank"]),
        "source_content_sha256": str(item["source_content_sha256"]),
        "chunk_id": str(item.get("chunk_id", "")),
        "family": str(item.get("family", "")),
        "filenames": sorted({
            str(p.get("filename", "")).strip()
            for p in provenance
            if str(p.get("filename", "")).strip()
        }),
        "source_paths": sorted({
            str(p.get("original_relative_path", "")).strip()
            for p in provenance
            if str(p.get("original_relative_path", "")).strip()
        }),
        "snippet": _preview(item.get("snippet", ""), limit),
        "rrf_score": item.get("rrf_score"),
        "lexical_rank": item.get("lexical_rank"),
        "semantic_rank": item.get("semantic_rank"),
        "source_extraction_truncated": bool(
            item.get("source_extraction_truncated", False)
        ),
    }


def _selection(value: str, count: int) -> tuple[str, list[int]]:
    text = value.strip().casefold()
    if text in {"x", "exclude"}:
        return "excluded", []
    if text in {"s", "skip", ""}:
        return "pending", []
    if text in {"q", "quit"}:
        return "quit", []

    chosen: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece.isdigit():
            raise ValueError("Use 1, 1,3, x, s, or q.")
        number = int(piece)
        if not 1 <= number <= count:
            raise ValueError(f"Candidate {number} is outside 1-{count}.")
        if number not in chosen:
            chosen.append(number)
    if not chosen:
        raise ValueError("Select at least one candidate.")
    return "approved", chosen


def review_semantic_benchmark(
    *,
    vault_root: Path,
    benchmark_path: Path,
    pilot_name: str = "pilot-v1",
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    device: str = "auto",
    candidate_limit: int = 5,
    snippet_characters: int = 700,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    search_fn=hybrid_search,
    model_loader=None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = list(benchmark.get("cases", []))
    if not cases:
        raise ValueError("Benchmark contains no cases")

    backup = benchmark_path.with_name(
        f"{benchmark_path.stem}.backup-{_stamp()}.json"
    )
    shutil.copy2(benchmark_path, backup)

    cached_loader = model_loader
    if cached_loader is None and search_fn is hybrid_search:
        policy = load_semantic_policy(semantic_policy_path)
        model, _ = _load_local_model(
            vault_root=vault_root,
            policy=policy,
            device=device,
        )

        def cached_loader(*args, **kwargs):
            return model

    output_fn(
        "Private benchmark reviewer. Content stays local.\n"
        f"Backup: {backup}"
    )

    quit_early = False
    for index, case in enumerate(cases, start=1):
        if str(case.get("status", "pending")).casefold() in {
            "approved", "excluded"
        }:
            continue

        question = str(case.get("question", "")).strip()
        result = search_fn(
            vault_root=vault_root,
            query=question,
            pilot_name=pilot_name,
            semantic_policy_path=semantic_policy_path,
            lexical_policy_path=lexical_policy_path,
            limit=candidate_limit,
            device=device,
            model_loader=cached_loader,
        )
        candidates = [
            _candidate(item, snippet_characters)
            for item in result.get("results", [])
        ]
        case["candidate_sources"] = candidates

        output_fn("\n" + "=" * 78)
        output_fn(f"Case {index}/{len(cases)}: {question}")
        for item in candidates:
            output_fn(
                f"\n[{item['rank']}] "
                f"{', '.join(item['filenames']) or '[unknown]'} "
                f"({item['family'] or 'unknown'})"
            )
            output_fn(f"    SHA: {item['source_content_sha256']}")
            output_fn(
                f"    Path: {' | '.join(item['source_paths']) or '[unknown]'}"
            )
            output_fn(
                f"    Ranks: lexical={item['lexical_rank']} "
                f"semantic={item['semantic_rank']}"
            )
            output_fn(f"    Preview: {item['snippet']}")

        prompt = (
            "\nChoose candidate number(s), x=exclude, s=skip, q=save/quit: "
        )
        answer = input_fn(prompt)
        while True:
            try:
                decision, selected = _selection(answer, len(candidates))
                break
            except ValueError as exc:
                output_fn(f"Invalid selection: {exc}")
                answer = input_fn("Selection: ")

        if decision == "quit":
            quit_early = True
            _atomic_json(benchmark_path, benchmark)
            break
        if decision == "approved":
            case["status"] = "approved"
            case["expected_source_sha256"] = [
                candidates[number - 1]["source_content_sha256"]
                for number in selected
            ]
        elif decision == "excluded":
            case["status"] = "excluded"
            case["expected_source_sha256"] = []
        else:
            case["status"] = "pending"
            case["expected_source_sha256"] = []

        _atomic_json(benchmark_path, benchmark)

    counts = {
        status: sum(
            str(case.get("status", "")).casefold() == status
            for case in benchmark["cases"]
        )
        for status in ("approved", "excluded", "pending")
    }
    return {
        "benchmark_review_schema_version": 1,
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_path": str(benchmark_path),
        "backup_path": str(backup),
        "quit_early": quit_early,
        "final_status_counts": counts,
        "private_text_uploaded": False,
    }
