"""Optional live Ollama smoke test for the P3.7 local CLI runtime."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alice_conversation.cli_runtime import build_local_conversation_runtime


pytestmark = pytest.mark.skipif(
    os.environ.get("ALICE_RUN_CLI_OLLAMA_INTEGRATION") != "1",
    reason="Set ALICE_RUN_CLI_OLLAMA_INTEGRATION=1 to run the local CLI smoke test.",
)


def test_local_cli_runtime_smoke(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    model = os.environ.get("ALICE_OLLAMA_MODEL", "qwen3:8b")
    runtime = build_local_conversation_runtime(
        repository_root=repository_root,
        vault_root=tmp_path / "private-vault",
        provider="ollama-local",
        model=model,
        retention="session_only",
    )
    try:
        result = runtime.send("Reply with a brief greeting.")
        assert result.content.strip()
        assert result.validation_outcome in {"accepted", "abstained"}
    finally:
        if runtime.has_session:
            runtime.close_session()
