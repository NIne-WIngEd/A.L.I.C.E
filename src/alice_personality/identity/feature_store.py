"""Caller-owned immutable full-state features; establish private custody first.

No OS isolation or source acceptance is created here. Production callers must
already be inside the owner execution boundary before touching this directory.
"""
from hashlib import sha256
from pathlib import Path
import json
import re

import torch

from .codec import canonical, fingerprint
from .contracts import IdentityError

SCHEMA = "alice-personality-frozen-feature-store-v1"
_SHA = re.compile(r"[0-9a-f]{64}")


def _hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(path):
    value = Path(path)
    if not value.is_absolute() or value != value.resolve() or any(p.is_symlink()
            or (hasattr(p, "is_junction") and p.is_junction()) for p in (value, *value.parents)):
        raise IdentityError("feature store requires an absolute nonsymlink custody path")
    return value


class FrozenFeatureStore:
    @classmethod
    def create(cls, directory, *, source_package_sha256):
        if not isinstance(source_package_sha256, str) or not _SHA.fullmatch(source_package_sha256):
            raise IdentityError("feature store needs an exact original source package binding")
        path = _path(directory)
        path.mkdir(mode=0o700, exist_ok=False)
        result = cls()
        result.path, result.sealed = path, False
        result.manifest = {"schema": SCHEMA, "source_package_sha256": source_package_sha256,
            "source_context": None, "entries": {}, "qualification": "UNQUALIFIED", "acceptance_authority": False}
        return result

    @classmethod
    def open(cls, directory, *, expected_manifest_sha256):
        path = _path(directory)
        manifest_path = _path(path / "manifest.json")
        manifest = json.loads(manifest_path.read_bytes())
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if (manifest.get("schema") != SCHEMA or manifest.get("manifest_sha256") != expected_manifest_sha256
                or sha256(canonical(body)).hexdigest() != expected_manifest_sha256
                or manifest.get("qualification") != "UNQUALIFIED" or manifest.get("acceptance_authority") is not False):
            raise IdentityError("feature store manifest differs from its external custody pin")
        entries = manifest.get("entries")
        if (type(entries) is not dict or not entries or any(not isinstance(k, str) or not _SHA.fullmatch(k) for k in entries)
                or {p.name for p in path.iterdir()} != {"manifest.json", *(k + ".pt" for k in entries)}):
            raise IdentityError("feature store membership differs from its complete entry inventory")
        result = cls()
        result.path, result.sealed, result.manifest = path, True, manifest
        return result

    @property
    def binding_sha256(self):
        if self.manifest["source_context"] is None:
            raise IdentityError("feature store has no bound encoder/tokenizer/source context")
        return sha256(canonical({k: self.manifest[k] for k in ("schema", "source_package_sha256", "source_context")})).hexdigest()

    def bind_context(self, context):
        canonical(context)
        if self.manifest["source_context"] is None and not self.sealed:
            self.manifest["source_context"] = json.loads(canonical(context))
        elif self.manifest["source_context"] != context:
            raise IdentityError("feature store encoder/tokenizer/source/implementation context changed")

    def _read(self, key):
        metadata = self.manifest["entries"].get(key)
        if metadata is None:
            raise IdentityError("complete feature store has no extracted entry for this text")
        path = _path(self.path / (key + ".pt"))
        if _hash(path) != metadata["file_sha256"]:
            raise IdentityError("feature store payload changed")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (set(value) != {"states", "input_ids", "attention_mask"}
                or list(value["states"].shape) != metadata["shape"]
                or fingerprint({k: value[k] for k in ("input_ids", "attention_mask")}) != metadata["inputs_sha256"]):
            raise IdentityError("feature store complete layer/token/mask binding differs")
        return value

    def encoded(self, text):
        key = sha256(text.encode("utf-8")).hexdigest()
        metadata = self.manifest["entries"].get(key)
        if metadata is None:
            raise IdentityError("complete feature store has no extracted entry for this text")
        dtypes = {str(d): d for d in (torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool,
                  torch.float16, torch.bfloat16, torch.float32, torch.float64)}
        value = {k: torch.tensor(metadata[k], dtype=dtypes[metadata["input_dtypes"][k]])
                 for k in ("input_ids", "attention_mask")}
        if fingerprint(value) != metadata["inputs_sha256"]:
            raise IdentityError("feature store token metadata changed")
        return value

    def features(self, text, inputs, compute):
        key = sha256(text.encode("utf-8")).hexdigest()
        inputs_hash = fingerprint(inputs)
        if key in self.manifest["entries"]:
            if self.manifest["entries"][key]["inputs_sha256"] != inputs_hash:
                raise IdentityError("feature store token IDs or attention mask changed for the same text")
            return self._read(key)["states"]
        if self.sealed:
            raise IdentityError("complete immutable feature store has no extracted entry")
        states = compute().detach().cpu()
        payload = {"states": states, **{k: v.detach().cpu() for k, v in inputs.items()}}
        path = _path(self.path / (key + ".pt"))
        with path.open("xb") as stream:
            torch.save(payload, stream)
        self.manifest["entries"][key] = {"file_sha256": _hash(path), "inputs_sha256": inputs_hash,
            "shape": list(states.shape), "bytes": path.stat().st_size,
            "input_dtypes": {k: str(v.dtype) for k, v in inputs.items()},
            **{k: v.detach().cpu().tolist() for k, v in inputs.items()}}
        return states

    def seal(self):
        if self.sealed:
            raise IdentityError("feature store is already sealed")
        if not self.manifest["entries"]:
            raise IdentityError("empty feature store cannot supply construction evidence")
        for key, metadata in self.manifest["entries"].items():
            if _hash(_path(self.path / (key + ".pt"))) != metadata["file_sha256"]:
                raise IdentityError("feature store payload changed before closing custody")
        self.manifest["manifest_sha256"] = sha256(canonical(self.manifest)).hexdigest()
        with _path(self.path / "manifest.json").open("xb") as stream:
            stream.write(canonical(self.manifest) + b"\n")
        self.sealed = True
        return self.summary()

    def summary(self):
        if not self.sealed:
            raise IdentityError("feature custody requires the actual sealed complete inventory")
        return {"schema": SCHEMA, "binding_sha256": self.binding_sha256,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "source_package_sha256": self.manifest["source_package_sha256"],
            "source_context": self.manifest["source_context"],
            "unique_entries": len(self.manifest["entries"]),
            "feature_bytes": sum(v["bytes"] for v in self.manifest["entries"].values()),
            "retention": "CALLER_CUSTODY_COMPLETE_FROZEN_VALUES", "qualification": "UNQUALIFIED", "acceptance_authority": False}


def verify_store_custody(binding, session):
    """Metadata proof is additionally bound to the actual reviewed frame bytes."""
    retained = binding.get("persistent_feature_values_retained")
    if retained is not session.get("persistent_feature_values_retained"):
        raise IdentityError("feature store retention differs from its closed session")
    if binding.get("schema") == "alice-personality-produced-frame-binding-v1":
        if retained is not False or session.get("schema") != "alice-personality-feature-session-v1":
            raise IdentityError("legacy source session cannot claim stored feature custody")
        return
    store = session.get("feature_store")
    if (binding.get("schema") != "alice-personality-produced-frame-binding-v2" or retained is not True
            or session.get("schema") != "alice-personality-feature-session-v2" or type(store) is not dict
            or store.get("schema") != SCHEMA or store.get("qualification") != "UNQUALIFIED"
            or store.get("acceptance_authority") is not False
            or store.get("retention") != "CALLER_CUSTODY_COMPLETE_FROZEN_VALUES"):
        raise IdentityError("complete caller feature custody is required")
    context = store.get("source_context", {})
    if (store.get("binding_sha256") != binding.get("feature_store_binding_sha256")
            or store.get("binding_sha256") != sha256(canonical({"schema": SCHEMA,
                "source_package_sha256": store.get("source_package_sha256"), "source_context": context})).hexdigest()
            or context.get("prepared_receipt_file_sha256") != binding.get("prepared_receipt_file_sha256")
            or context.get("implementation_sha256") != binding.get("implementation_sha256")
            or context.get("source_geometry") != binding.get("source_geometry")
            or type(store.get("unique_entries")) is not int or store["unique_entries"] < 1
            or type(store.get("feature_bytes")) is not int or store["feature_bytes"] < 1
            or not isinstance(store.get("manifest_sha256"), str) or not _SHA.fullmatch(store["manifest_sha256"])):
        raise IdentityError("feature store source/tokenizer/encoder/implementation binding differs")
