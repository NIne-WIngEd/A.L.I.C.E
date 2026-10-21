"""Real native grouped updates survive an explicit experimental restart."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.alice_personality.teaching import TeachingError
from tests.personality import test_paired_learning_experiment as fixtures

try:
    api = importlib.import_module("scripts.eipm.gemma_n0.paired_learning_recovery")
except ModuleNotFoundError as error:
    if error.name != "scripts.eipm.gemma_n0.paired_learning_recovery":
        raise
    api = None


@pytest.fixture
def native():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    value = fixtures.PairedLearningExperimentTests()
    value.setUp()
    yield value
    torch.set_num_threads(previous)


def run(native, directory, **options):
    assert api is not None, "The bounded experimental recovery helper is missing"
    arguments = dict(plan=native.plan, expected_plan_sha256=fixtures.api.plan_sha256(native.plan),
                     batch_loader=native.batches.__getitem__, validate_batch=native.validate,
                     custody_guard=native.guard, progress_directory=directory)
    arguments.update(options)
    return api.run_experiment(native.model, native.objectives, native.recipe, **arguments)


def exact(expected, actual):
    assert type(expected) is type(actual)
    if isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            exact(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            exact(left, right)
    else:
        assert expected == actual


class Interrupted(Exception):
    pass


def interrupted(native, directory, count):
    events = []
    def stop(event):
        events.append(event)
        if len(events) == count:
            raise Interrupted
    with pytest.raises(Interrupted):
        run(native, directory, on_event=stop)
    snapshots = sorted(directory.iterdir())
    assert len(snapshots) == count
    snapshot = snapshots[-1]
    manifest = snapshot / "manifest.json"
    return snapshot, sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_restart_matches_uninterrupted_original_model_adamw_rng_and_events(native, tmp_path, count):
    # Lost AdamW moments, duplicate updates or a changed arm seed break this comparison.
    original = native.run_trial()
    before = deepcopy(native.model.state_dict())
    snapshot, digest = interrupted(native, tmp_path / "attempt", count)
    events = []
    resumed = run(native, tmp_path / "recovery", resume_from=snapshot,
                  expected_resume_manifest_sha256=digest, on_event=events.append,
                  destination=tmp_path / "finished")
    exact(original["arms"], resumed["arms"])
    exact(before, native.model.state_dict())
    assert resumed["total_committed_updates"] == 4
    assert resumed["new_updates_this_invocation"] == len(events) == 4-count
    assert resumed["materializations_this_invocation"] == 4 + 2*(4-count)
    assert resumed["native_checkpoint"] is False and resumed["resumable"] is False
    assert resumed["experimental_recovery_only"] is True
    restored = torch.load(tmp_path / "finished" / "experimental.pt", weights_only=True)
    exact(resumed, restored)
    assert not (tmp_path / "finished" / "resume.pt").exists()


def test_fresh_recovery_matches_original_and_saves_each_group(native, tmp_path):
    original = native.run_trial()
    result = run(native, tmp_path / "progress")
    exact(original["arms"], result["arms"])
    assert result["new_updates_this_invocation"] == result["total_committed_updates"] == 4
    assert result["materializations_this_invocation"] == 12
    assert len(list((tmp_path / "progress").iterdir())) == 4


def rewrite_snapshot(snapshot, change):
    state_path = snapshot / "experimental-progress.pt"
    state = torch.load(state_path, weights_only=True)
    change(state)
    torch.save(state, state_path)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["state_file_sha256"] = sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"))+"\n")
    return sha256(manifest_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["plan", "parameters", "cursor", "events", "occurrences", "frozen", "optimizer", "rng", "arm_gap"])
def test_contradictory_snapshot_refuses_before_any_new_update(native, tmp_path, mutation):
    snapshot, digest = interrupted(native, tmp_path / "attempt", 3)
    def change(state):
        arm = state["arms"]["control"]
        if mutation == "plan":
            state["binding"]["plan"]["control_permutation"] = [0, 1]
        elif mutation == "parameters":
            state["binding"]["optimizer_parameter_names"].reverse()
        elif mutation == "cursor":
            state["cursor"]["next_group"] = 2
        elif mutation == "events":
            arm["events"][0]["batch_sha256"].reverse()
        elif mutation == "occurrences":
            arm["batch_occurrences"] = {}
        elif mutation == "frozen":
            next(value for name, value in arm["model_state"].items() if name.startswith("n1.")).add_(1)
        elif mutation == "optimizer":
            arm["optimizer_state"]["param_groups"][0]["lr"] = .2
        elif mutation == "rng":
            arm["torch_rng"] = torch.tensor([1.])
        else:
            del state["arms"]["matched"]
    digest = rewrite_snapshot(snapshot, change)
    events = []
    with pytest.raises((TeachingError, ValueError, RuntimeError)):
        run(native, tmp_path / "refused", resume_from=snapshot,
            expected_resume_manifest_sha256=digest, on_event=events.append)
    assert events == []


@pytest.mark.parametrize("mutation", ["unsealed", "missing_pin", "changed_pin", "changed_state", "changed_plan"])
def test_missing_or_changed_external_binding_refuses(native, tmp_path, mutation):
    snapshot, digest = interrupted(native, tmp_path / "attempt", 1)
    options = dict(resume_from=snapshot, expected_resume_manifest_sha256=digest)
    if mutation == "unsealed":
        (snapshot / "manifest.json").unlink()
    elif mutation == "missing_pin":
        options.pop("expected_resume_manifest_sha256")
    elif mutation == "changed_pin":
        options["expected_resume_manifest_sha256"] = "0"*64
    elif mutation == "changed_state":
        with (snapshot / "experimental-progress.pt").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        plan = deepcopy(native.plan)
        plan["bindings"]["different"] = True
        options.update(plan=plan, expected_plan_sha256=fixtures.api.plan_sha256(plan))
    with pytest.raises((TeachingError, ValueError, OSError)):
        run(native, tmp_path / "refused", **options)


def test_interrupted_snapshot_write_preserves_prior_complete_generation(native, tmp_path):
    assert api is not None, "The bounded experimental recovery helper is missing"
    actual_save = torch.save
    writes = 0
    def failed_write(value, stream, *args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            stream.write(b"partial")
            raise OSError("injected lost storage during snapshot write")
        return actual_save(value, stream, *args, **kwargs)
    with patch.object(torch, "save", failed_write), pytest.raises(OSError, match="lost storage"):
        run(native, tmp_path / "attempt")
    snapshots = sorted((tmp_path / "attempt").iterdir())
    assert (snapshots[0] / "manifest.json").is_file()
    assert not (snapshots[1] / "manifest.json").exists()
    digest = sha256((snapshots[0] / "manifest.json").read_bytes()).hexdigest()
    resumed = run(native, tmp_path / "recover", resume_from=snapshots[0],
                  expected_resume_manifest_sha256=digest)
    exact(native.run_trial()["arms"], resumed["arms"])


def test_resumption_rechecks_every_admitted_batch_before_new_gradient(native, tmp_path):
    snapshot, digest = interrupted(native, tmp_path / "attempt", 1)
    last = native.digests[-1]
    def validate(batch):
        value = native.validate(batch)
        if value == last:
            raise TeachingError("admitted endpoint changed")
        return value
    events = []
    with pytest.raises(TeachingError, match="endpoint changed"):
        run(native, tmp_path / "refused", resume_from=snapshot,
            expected_resume_manifest_sha256=digest, validate_batch=validate, on_event=events.append)
    assert events == []
