from __future__ import annotations

import json
from pathlib import Path

import torch

from alice_personality.n0.ranker import listwise_preference_loss


CURRICULUM = Path("training/eipm/n0/sol_curriculum_seed_v0.1.jsonl")


def test_curriculum_seed_schema_and_coverage() -> None:
    rows = [json.loads(line) for line in CURRICULUM.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 60
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"train", "dev"}
    assert len({row["competency"] for row in rows}) == 43
    for row in rows:
        assert row["task"] == "candidate_ranking"
        assert len(row["candidates"]) >= 2
        assert row["preferred_indices"]
        assert max(row["preferred_indices"]) < len(row["candidates"])
        assert row["source"] == "sol_authored"


def test_listwise_preference_loss_rewards_supported_tie() -> None:
    scores_good = torch.tensor([3.0, 3.0, -1.0])
    scores_bad = torch.tensor([-1.0, -1.0, 3.0])
    preferred = [torch.tensor([True, True, False])]
    good = listwise_preference_loss(scores_good, [3], preferred)
    bad = listwise_preference_loss(scores_bad, [3], preferred)
    assert good < bad
