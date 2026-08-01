import importlib.util
import sys
from pathlib import Path


def _load_script_module(root: Path):
    path = root / "scripts/run_phase4_live_research.py"
    spec = importlib.util.spec_from_file_location(
        "p410b_live_research_script_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_private_live_research_script_keeps_runtime_and_output_outside_git():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_phase4_live_research.py").read_text(encoding="utf-8")
    assert "build_phase4_live_research_runtime" in text
    assert "Private runtime and receipt output must remain outside Git" in text
    assert 'mode="research"' in text
    assert 'availability="available"' in text
    assert "Raw query text entered" in text
    assert "Provider credential entered" in text
    assert "substituted the live executor" in text
    assert "validate_operational_boundary" in text
    assert "Source content entered" in text
    assert '"fetch_failures"' in text
    assert "response_validation_hook" not in text  # executor owns P3.6 binding


def test_private_runtime_loader_supports_dataclass_modules(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = _load_script_module(root)
    runtime = tmp_path / "private_runtime.py"
    runtime.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class RuntimeValue:\n"
        "    value: str\n",
        encoding="utf-8",
    )
    module = script._load_module(runtime)
    assert module.RuntimeValue("ok").value == "ok"
