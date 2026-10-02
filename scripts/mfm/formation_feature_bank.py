"""Lossless, source-bound frozen MFM representations; never a memory authority.

The bank is a placement optimization for the existing frozen-base objective.
Every consumer must re-admit its exact train/development corpus. FINAL is never
exported, targets never enter the backbone, and an incomplete bank cannot train.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_learning_v16 import model_input_sha256_v16
from . import train_v16_formation_specialist as training

shared = training.shared
EXPORT_SCHEMA = "mfm-v16-frozen-feature-export-v1"
CASE_SCHEMA = "mfm-v16-frozen-feature-case-v1"
BANK_SCHEMA = "mfm-v16-frozen-feature-bank-v1"


def _within(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise CognitiveKernelContractError("invalid feature-bank member")
    relative = Path(name)
    if relative.is_absolute() or any(p in ("", ".", "..") or ":" in p for p in name.split("/")):
        raise CognitiveKernelContractError("feature-bank path escapes custody")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)) or \
            not path.resolve().is_relative_to(root.resolve()):
        raise CognitiveKernelContractError("feature-bank link or path escapes custody")
    return path


def case_key(example) -> str:
    if example.split not in {"train", "development"}:
        raise CognitiveKernelContractError("feature bank never opens FINAL")
    return model_input_sha256_v16(example)


def feature_binding(preflight: dict, prepared: dict, torch_version: str) -> dict:
    return {
        "preflight_sha256": preflight["record_sha256"],
        "preflight": preflight,
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_weight_sha256": preflight["prepared_base_weight_sha256"],
        "corpus_sha256": preflight["corpus_sha256"],
        "exporter_sha256": shared._digest(Path(__file__)),
        "export_entrypoint_sha256": shared._digest(Path(__file__).with_name(
            "export_v16_frozen_features.py")),
        "torch_version": torch_version,
        "transformers_version": preflight["transformers_version"],
        "backbone_dtype": "bfloat16", "backbone_device": "cpu",
        "representation": "base.model.last_hidden_state",
        "base_frozen": True, "language_head_used": False,
        "target_visible_to_backbone": False, "final_payloads_opened": False,
        "qualified_for_product": False,
    }


def _validate_tensors(tensors: dict, *, width: int | None = None) -> None:
    import torch
    if set(tensors) != {"states", "attention_mask", "input_ids"}:
        raise CognitiveKernelContractError("feature tensor set differs")
    states, mask, ids = (tensors[k] for k in ("states", "attention_mask", "input_ids"))
    if states.ndim != 3 or states.shape[0] != 1 or not states.shape[1] or not states.shape[2] or \
            states.shape[:2] != mask.shape or mask.shape != ids.shape or \
            states.dtype != torch.bfloat16 or mask.dtype != torch.int64 or \
            ids.dtype != torch.int64 or (width is not None and states.shape[2] != width):
        raise CognitiveKernelContractError("feature shapes or dtypes differ")
    if not torch.isfinite(states).all() or not torch.all((mask == 0) | (mask == 1)) or \
            not mask.any() or (ids < 0).any():
        raise CognitiveKernelContractError("feature states or mask are invalid")


def write_case(root: Path, example, encoded: dict, states, export_digest: str) -> dict:
    import torch
    from safetensors.torch import save_file, load_file
    key = case_key(example)
    target = _within(root, "cases/" + key)
    if target.exists():
        return read_case(root, example, export_digest)[0]
    parent = _within(root, "cases")
    parent.mkdir(mode=0o700, exist_ok=True)
    tensors = {
        "states": states.detach().cpu().contiguous(),
        "attention_mask": encoded["attention_mask"].detach().cpu().contiguous(),
        "input_ids": encoded["input_ids"].detach().cpu().contiguous(),
    }
    _validate_tensors(tensors)
    stage = parent / (".staging-" + uuid4().hex)
    stage.mkdir(mode=0o700)
    try:
        path = stage / "features.safetensors"
        save_file(tensors, str(path))
        path.chmod(0o600)
        restored = load_file(str(path), device="cpu")
        if any(not torch.equal(tensors[name], restored[name]) for name in tensors):
            raise CognitiveKernelContractError("feature serialization changed representations")
        receipt = shared._write_new(stage / "case.json", {
            "schema": CASE_SCHEMA, "case_id": example.case_id, "split": example.split,
            "model_input_sha256": key, "export_sha256": export_digest,
            "features_sha256": shared._digest(path),
            "shape": list(tensors["states"].shape),
            "serialized_tensors_exact": True, "target_visible_to_backbone": False,
        })
        (stage / "case.json").chmod(0o600)
        stage.rename(target)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_case(root: Path, example, export_digest: str, *, width: int | None = None):
    from safetensors.torch import load_file
    key = case_key(example)
    selected = _within(root, "cases/" + key)
    receipt_path = _within(root, "cases/" + key + "/case.json")
    payload = _within(root, "cases/" + key + "/features.safetensors")
    receipt = shared._read_sealed(receipt_path, CASE_SCHEMA)
    expected = {"case_id": example.case_id, "split": example.split,
                "model_input_sha256": key, "export_sha256": export_digest,
                "serialized_tensors_exact": True, "target_visible_to_backbone": False}
    if any(receipt.get(k) != v for k, v in expected.items()) or \
            shared._digest(payload) != receipt.get("features_sha256") or \
            {p.name for p in selected.iterdir()} != {"case.json", "features.safetensors"}:
        raise CognitiveKernelContractError("feature case binding or bytes differ")
    tensors = load_file(str(payload), device="cpu")
    _validate_tensors(tensors, width=width)
    if list(tensors["states"].shape) != receipt.get("shape"):
        raise CognitiveKernelContractError("feature receipt shape differs")
    return receipt, tensors


def finalize(root: Path, export: dict, examples, processor_files: list[dict]) -> dict:
    cases = []
    for example in examples:
        receipt, _ = read_case(root, example, export["record_sha256"])
        cases.append({"case_id": example.case_id, "split": example.split,
                      "model_input_sha256": case_key(example),
                      "case_receipt_sha256": receipt["record_sha256"]})
    if len({c["model_input_sha256"] for c in cases}) != len(cases):
        raise CognitiveKernelContractError("feature bank repeats a model input")
    expected_dirs = {c["model_input_sha256"] for c in cases}
    if {p.name for p in (root / "cases").iterdir()} != expected_dirs:
        raise CognitiveKernelContractError("feature bank has unlisted or incomplete cases")
    for row in processor_files:
        if shared._digest(_within(root, "processor/" + row["path"])) != row["sha256"]:
            raise CognitiveKernelContractError("feature processor bytes differ")
    receipt = shared._write_new(root / "bank.json", {
        "schema": BANK_SCHEMA, "export_sha256": export["record_sha256"],
        "corpus_sha256": export["corpus_sha256"],
        "cases": cases, "processor_files": processor_files,
        "complete": True, "final_payloads_opened": False,
        "qualified_for_product": False,
    })
    (root / "bank.json").chmod(0o600)
    return receipt


def verify_bank(root: Path, *, expected_sha256: str, examples, preflight: dict) -> tuple[dict, dict]:
    require_sha256(expected_sha256, "feature_bank_sha256")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise CognitiveKernelContractError("feature-bank custody is a link")
    root = root.resolve(strict=True)
    bank_path = _within(root, "bank.json")
    bank = shared._read_sealed(bank_path, BANK_SCHEMA)
    export = shared._read_sealed(_within(root, "export.json"), EXPORT_SCHEMA)
    if bank["record_sha256"] != expected_sha256 or \
            bank.get("complete") is not True or bank.get("final_payloads_opened") is not False or \
            bank.get("export_sha256") != export["record_sha256"] or \
            bank.get("corpus_sha256") != preflight["corpus_sha256"] or \
            export.get("preflight") != preflight or \
            export.get("preflight_sha256") != preflight["record_sha256"] or \
            export.get("corpus_sha256") != preflight["corpus_sha256"] or \
            export.get("prepared_base_receipt_sha256") != preflight["prepared_base_receipt_sha256"] or \
            export.get("prepared_base_weight_sha256") != preflight["prepared_base_weight_sha256"] or \
            export.get("transformers_version") != preflight["transformers_version"] or \
            export.get("exporter_sha256") != shared._digest(Path(__file__)) or \
            export.get("export_entrypoint_sha256") != shared._digest(Path(__file__).with_name(
                "export_v16_frozen_features.py")) or \
            any(export.get(k) != v for k, v in {
                "base_frozen": True, "language_head_used": False,
                "target_visible_to_backbone": False, "final_payloads_opened": False,
                "backbone_dtype": "bfloat16", "backbone_device": "cpu",
                "representation": "base.model.last_hidden_state"}.items()):
        raise CognitiveKernelContractError("feature bank lineage, source or code differs")
    expected = {(e.case_id, e.split, case_key(e)) for e in examples}
    rows = bank.get("cases", [])
    if len(rows) != len(expected) or {(r["case_id"], r["split"], r["model_input_sha256"]) for r in rows} != expected:
        raise CognitiveKernelContractError("feature bank case/split set differs")
    for example, row in zip(examples, rows):
        if row["case_id"] != example.case_id:
            raise CognitiveKernelContractError("feature bank case order differs")
        receipt, _ = read_case(root, example, export["record_sha256"])
        if receipt["record_sha256"] != row["case_receipt_sha256"]:
            raise CognitiveKernelContractError("feature bank case receipt differs")
    listed = {"bank.json", "export.json"}
    for row in rows:
        listed.update({"cases/" + row["model_input_sha256"] + "/" + p
                       for p in ("case.json", "features.safetensors")})
    files = bank.get("processor_files", [])
    if files != training_prepared_files(preflight):
        raise CognitiveKernelContractError("feature processor file set differs")
    for row in files:
        path = _within(root, "processor/" + row["path"])
        if shared._digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
            raise CognitiveKernelContractError("feature processor bytes differ")
        listed.add("processor/" + row["path"])
    for path in root.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(root).as_posix() not in listed):
            raise CognitiveKernelContractError("feature bank has unlisted payload or link")
    return bank, export


def training_prepared_files(preflight: dict) -> list[dict]:
    # Independently pinned publisher processor files; no weight snapshot is
    # needed on a cache-only trainer. This is artifact transport, not re-cloning.
    from scripts.mfm.verify_gemma4_pretrained import FILES
    return [{"path": name, "size": size, "sha256": digest}
            for name, (size, digest) in sorted(FILES.items())
            if name != "model.safetensors"]
