from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from alice_conversation.final_evaluation import build_expected_observation_fixture
from alice_conversation.final_evaluation_contract import load_conversation_final_evaluation_benchmark, load_conversation_final_evaluation_policy
ROOT=Path(__file__).resolve().parents[2]

def policy(): return load_conversation_final_evaluation_policy(ROOT/"policies/conversation_final_evaluation_policy.json")
def benchmark(): return load_conversation_final_evaluation_benchmark(ROOT/"benchmarks/phase3/conversation_final_evaluation_v1.json",policy=policy())
def passing_submissions(): return build_expected_observation_fixture(benchmark())
def replace_submission(items,index,**changes):
    values=list(items); values[index]=replace(values[index],**changes); return tuple(values)
