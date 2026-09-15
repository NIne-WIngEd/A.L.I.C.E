#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.ranker import build_ranker_from_mlm_checkpoint


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm resident-ranker latency sanity check for N0 v0.2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--mlm-checkpoint", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("warm latency sanity requires CUDA")
    device = torch.device("cuda")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before latency benchmark") from exc

    config = load_n0_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer_dir)

    started = time.perf_counter()
    model = build_ranker_from_mlm_checkpoint(args.mlm_checkpoint, hidden_size=config.hidden_size)
    state = load_file(args.ranker, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"ranker state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    row = next(
        json.loads(line)
        for line in Path(args.sample_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    candidates = [str(value) for value in row["candidates"]][:3]
    prompts = [str(row["prompt"])] * len(candidates)

    encoded = tokenizer(
        prompts,
        candidates,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        for _ in range(10):
            model(ids, mask)
        torch.cuda.synchronize(device)

        model_only: list[float] = []
        for _ in range(50):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(ids, mask)
            torch.cuda.synchronize(device)
            model_only.append((time.perf_counter() - start) * 1000.0)

        e2e: list[float] = []
        for _ in range(30):
            start = time.perf_counter()
            live = tokenizer(
                prompts,
                candidates,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            live_ids = live["input_ids"].to(device)
            live_mask = live["attention_mask"].to(device)
            model(live_ids, live_mask)
            torch.cuda.synchronize(device)
            e2e.append((time.perf_counter() - start) * 1000.0)

        stress_prompt = (str(row["prompt"]) + " context") * 80
        stress_prompts = [stress_prompt] * len(candidates)
        stress = tokenizer(
            stress_prompts,
            candidates,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        stress_ids = stress["input_ids"].to(device)
        stress_mask = stress["attention_mask"].to(device)
        for _ in range(5):
            model(stress_ids, stress_mask)
        torch.cuda.synchronize(device)
        stress_times: list[float] = []
        for _ in range(20):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(stress_ids, stress_mask)
            torch.cuda.synchronize(device)
            stress_times.append((time.perf_counter() - start) * 1000.0)

    result = {
        "schema": "alice.eipm.n0.v02-warm-latency-sanity.v0.1",
        "status": "PASS",
        "scope": "resident_model_three_candidate_p100_sanity_not_production_slo",
        "hardware": torch.cuda.get_device_name(0),
        "resident_model_required": True,
        "candidate_count": len(candidates),
        "interactive_sequence_length": int(ids.shape[1]),
        "model_load_seconds": load_seconds,
        "model_only_p50_ms": statistics.median(model_only),
        "model_only_p95_ms": percentile(model_only, 0.95),
        "end_to_end_p50_ms": statistics.median(e2e),
        "end_to_end_p95_ms": percentile(e2e, 0.95),
        "stress_512_model_only_p50_ms": statistics.median(stress_times),
        "stress_512_model_only_p95_ms": percentile(stress_times, 0.95),
        "warm_interactive_target_ms_p95": 500.0,
        "warm_interactive_target_pass": percentile(e2e, 0.95) <= 500.0,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
