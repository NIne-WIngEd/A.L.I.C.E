"""Create-only core/calibration snapshots with no training approval claims."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from .contracts import FRAME_SCHEMA, PACKET_SCHEMA, IdentityError, IdentityModelConfig
from .model import IdentityModel

SNAPSHOT_SCHEMA = "alice-personality-identity-untrained-snapshot-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: _hash(root / name) for name in ("__init__.py", "contracts.py", "model.py", "checkpoints.py")}


def save_untrained_snapshot(model: IdentityModel, destination: str | Path) -> dict:
    """Persist learned parameter tensors separately from calibration state.

    The classification reports absence of validated training proof. Calling
    this API, changing weights, or supplying a source frame cannot self-grant
    identity acceptance. No inputs, source text or source feature values persist.
    """
    path = Path(destination)
    if not path.is_absolute() or path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise IdentityError("snapshot requires a new absolute directory under an existing parent")
    config = asdict(model.config)
    core = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith("n3.")}
    calibration = {name: value.detach().cpu().clone() for name, value in model.n3.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in (*core.values(), *calibration.values())):
        raise IdentityError("nonfinite learned state cannot be exported")
    path.mkdir()
    (path / "config.json").write_bytes(_canonical(config) + b"\n")
    # Plain tensor dictionaries only; loading below uses weights_only=True.
    torch.save(core, path / "core.pt")
    torch.save(calibration, path / "calibration.pt")
    manifest = {"schema": SNAPSHOT_SCHEMA, "training_state": "UNTRAINED", "qualification": "UNQUALIFIED",
                "training_evidence": None, "source_acceptance_authority": False,
                "meaning": "Implemented random/unverified learned-state snapshot; no personality qualification",
                "frame_schema": FRAME_SCHEMA, "packet_schema": PACKET_SCHEMA,
                "config": config, "code_sha256": _code_hashes(), "torch_version": str(torch.__version__),
                "files": {name: _hash(path / name) for name in ("config.json", "core.pt", "calibration.pt")},
                "calibration_separate_from_core": True, "source_values_retained": False}
    manifest["manifest_sha256"] = sha256(_canonical(manifest)).hexdigest()
    (path / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    return manifest


def load_untrained_snapshot(source: str | Path, *, device: str = "cpu") -> IdentityModel:
    path = Path(source)
    expected_files = {"manifest.json", "config.json", "core.pt", "calibration.pt"}
    if (not path.is_absolute() or path.is_symlink() or not path.is_dir()
            or {item.name for item in path.iterdir()} != expected_files
            or any(item.is_symlink() or not item.is_file() for item in path.iterdir())):
        raise IdentityError("snapshot must be an exact nonsymlink implementation package")
    manifest = json.loads((path / "manifest.json").read_bytes())
    digest = manifest.pop("manifest_sha256", None)
    if digest != sha256(_canonical(manifest)).hexdigest():
        raise IdentityError("snapshot manifest hash differs")
    if (manifest.get("schema") != SNAPSHOT_SCHEMA or manifest.get("training_state") != "UNTRAINED"
            or manifest.get("qualification") != "UNQUALIFIED" or manifest.get("training_evidence") is not None
            or manifest.get("source_acceptance_authority") is not False
            or manifest.get("frame_schema") != FRAME_SCHEMA or manifest.get("packet_schema") != PACKET_SCHEMA):
        raise IdentityError("snapshot cannot grant training, acceptance or qualification")
    if manifest.get("code_sha256") != _code_hashes() or manifest.get("torch_version") != str(torch.__version__):
        raise IdentityError("snapshot differs from exact implementation/runtime")
    if set(manifest.get("files", {})) != {"config.json", "core.pt", "calibration.pt"}:
        raise IdentityError("snapshot payload membership differs")
    if any(_hash(path / name) != digest for name, digest in manifest["files"].items()):
        raise IdentityError("snapshot learned-state hash differs")
    config = json.loads((path / "config.json").read_bytes())
    if config != manifest.get("config"):
        raise IdentityError("snapshot config binding differs")
    model = IdentityModel(IdentityModelConfig(**config))
    core = torch.load(path / "core.pt", map_location="cpu", weights_only=True)
    calibration = torch.load(path / "calibration.pt", map_location="cpu", weights_only=True)
    for state in (core, calibration):
        if (type(state) is not dict or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                                         or not bool(torch.isfinite(value).all()) for key, value in state.items())):
            raise IdentityError("snapshot state must contain only finite named tensors")
    expected = model.state_dict()
    if set(core) != {name for name in expected if not name.startswith("n3.")}:
        raise IdentityError("snapshot core must contain exact N1/N2 parameters only")
    model.n3.load_state_dict(calibration, strict=True)
    combined = dict(core)
    combined.update({"n3." + name: value for name, value in calibration.items()})
    model.load_state_dict(combined, strict=True)
    model.to(torch.device(device))
    model.set_phase("inference")
    return model
