"""Public launcher custody/resource mechanics only; no real Gemma/source proof.

Tiny arbitrary byte banks test preflight, not tensor semantics. Orchestration
tests inject the existing exporter/importer; their source/feature tests remain
the authority for actual closure mechanics. Nothing qualifies personality N0.
"""
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.alice_personality.gemma_n0 import public_feature_handoff as handoff

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/eipm/gemma_n0/close_public_feature_handoff.py"
SBATCH = SCRIPT.with_name("magnolia_close_public_feature_handoff.sbatch")
spec = importlib.util.spec_from_file_location("public_handoff_launcher", SCRIPT)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)


class PublicHandoffLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        original = self.root / "public-semantics-576600"
        evidence = original / "evidence"
        evidence.mkdir(parents=True)
        self.cache = original / "public-feature-cache"
        self.cache.mkdir()
        self.overlay = original / "overlay"
        self.overlay.mkdir()
        (self.overlay / "fixture.py").write_text("# public stdlib-only fixture\n")
        self.snapshot = self.root / "public-fixture-snapshot"
        self.snapshot.mkdir()
        self.runtime = {"backend": "transformers", "dtype": "bfloat16",
            "torch_version": "fixture-only", "transformers_version": "fixture-only"}
        self.code, self.own = {}, {}
        for name in sorted(handoff._PRODUCER_NAMES):
            path = self.root / "code" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# tiny public historical-code fixture " + name)
            self.code[name] = path
        for name in handoff._OWN_CODE:
            path = self.root / "own-code" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# tiny public closure-code fixture " + name)
            self.own[name] = path
        public = self.root / "public-rows.txt"
        public.write_text("Tiny public text only.")
        self.source_path, self.prepared_path, self.clone_path = [self.root / name for name in
            ("source.json", "prepared.json", "clone.json")]
        clone = self.write(self.clone_path, {"schema": "fixture-clone-metadata-only"})
        source = self.write(self.source_path, {"private_identity_data": False,
            "statistics": {"final_payload_opened": False}, "files": [{"path": str(public)}]})
        prepared = self.write(self.prepared_path, {"repository": handoff.MODEL,
            "revision": handoff.REVISION, "role": "personality", "state": "PREPARED_UNQUALIFIED",
            "snapshot_path": str(self.snapshot), "interfaces": {},
            "clone_receipt_path": str(self.clone_path), "clone_receipt_sha256": clone["receipt_sha256"]})
        self.plan_path = original / "precommitted-plan.json"
        self.plan = {"state": "PRECOMMITTED_PUBLIC_READOUT_UNQUALIFIED",
            "private_identity_data": False, "final_payload_opened": False, "n0_approved": False,
            "source_admission_path": str(self.source_path), "preparation_path": str(self.prepared_path),
            "source_admission_file_sha256": handoff._hash(self.source_path),
            "preparation_file_sha256": handoff._hash(self.prepared_path),
            "source_admission_sha256": source["receipt_sha256"], "preparation_sha256": prepared["receipt_sha256"],
            "code": [{"name": name, "sha256": handoff._hash(path)} for name, path in sorted(self.code.items())],
            "runtime": self.runtime, "unique_complete_feature_inputs": 2}
        plan = self.write(self.plan_path, self.plan)
        budget_path = evidence / "feature-budget.json"
        self.write(budget_path, {"estimated_logical_feature_cache_bytes": 256})
        cache_records = []
        for index in range(2):
            key = sha256(str(index).encode()).hexdigest()
            tensor, metadata = self.cache / (key + ".pt"), self.cache / (key + ".json")
            tensor.write_bytes(b"public fixture bytes, not a tensor")
            metadata.write_text('{"scope":"public fixture metadata only"}')
            cache_records.append({"key": key, "tensor_path": str(tensor), "metadata_path": str(metadata),
                "tensor_sha256": handoff._hash(tensor), "metadata_sha256": handoff._hash(metadata),
                "logical_feature_bytes": 128})
        self.experiment_path = evidence / "experiment.json"
        self.experiment = {"state": "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED",
            "private_identity_data": False, "final_payload_opened": False, "n0_approved": False,
            "upstream_gradient": False, "upstream_tensor_mutation": False, "personality_qualified": False,
            "binding": {"plan_sha256": handoff._hash(self.plan_path), "plan_content_sha256": plan["receipt_sha256"]},
            "pre_forward_feature_budget": {"path": str(budget_path), "sha256": handoff._hash(budget_path)},
            "cache": cache_records}
        self.write(self.experiment_path, self.experiment)
        self.run_root = self.root / "fresh-handoff-run"
        self.run_root.mkdir()

    def write(self, path, value):
        sealed = handoff._seal(value)
        path.write_bytes(handoff._canonical(sealed) + b"\n")
        return sealed

    def inspect(self, **overrides):
        args = dict(experiment_path=self.experiment_path, experiment_sha256=handoff._hash(self.experiment_path),
            plan_path=self.plan_path, plan_sha256=handoff._hash(self.plan_path), cache_directory=self.cache,
            compute_root=self.root, overlay_directory=self.overlay,
            overlay_sha256=launch.overlay_inventory(self.overlay)["sha256"], producer_paths=self.code, own_paths=self.own)
        args.update(overrides)
        return launch.inspect_inputs(**args)

    def test_preflight_counts_exact_copy_members_excluding_checkpoint(self):
        (self.snapshot / "large-checkpoint.pt").write_bytes(b"model must never be copied" * 200)
        inputs = self.inspect()
        self.assertEqual(inputs.complete_unique_inputs, 2)
        self.assertEqual(inputs.logical_feature_bytes, 256)
        self.assertEqual(len(inputs.files), 6 + 6 + 2 + 4)
        self.assertTrue(all(self.snapshot not in path.parents for path, _ in inputs.files))
        self.assertEqual(inputs.exact_copied_input_bytes, sum(path.stat().st_size for path, _ in inputs.files))
        free = inputs.exact_copied_input_bytes + launch.METADATA_SLACK_BYTES
        with patch.object(launch.shutil, "disk_usage", return_value=SimpleNamespace(free=free)):
            storage = launch.storage_gate(inputs, self.run_root)
        self.assertEqual(storage["required_free_bytes"], free)
        self.assertEqual(storage["checkpoint_bytes_copied"], 0)

    def test_insufficient_storage_aborts_before_export(self):
        inputs = self.inspect()
        with patch.object(launch.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
             patch.object(handoff, "export_public_features") as exporter:
            with self.assertRaisesRegex(launch.LaunchError, "insufficient"):
                launch.close_handoff(inputs, self.run_root, context={"scope": "public fixture"})
        exporter.assert_not_called()
        self.assertEqual(list(self.run_root.iterdir()), [])

    def test_overwrite_and_external_pin_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "external pin"):
            self.inspect(experiment_sha256="0" * 64)
        inputs = self.inspect()
        (self.run_root / "existing.json").write_text("existing")
        with self.assertRaisesRegex(launch.LaunchError, "fresh empty"):
            launch.storage_gate(inputs, self.run_root)

    def test_historical_code_and_overlay_tamper_rejected(self):
        self.code["backbone.py"].write_text("changed historical producer")
        with self.assertRaisesRegex(launch.LaunchError, "historical producer code"):
            self.inspect()
        old_pin = launch.overlay_inventory(self.overlay)["sha256"]
        (self.overlay / "fixture.py").write_text("changed dependency bytes")
        with self.assertRaisesRegex(launch.LaunchError, "overlay bytes"):
            self.inspect(overlay_sha256=old_pin)

    def test_private_true_or_truthy_flags_cannot_enter_public_closure(self):
        for invalid in (True, "false", 0):
            with self.subTest(invalid=invalid):
                self.experiment["private_identity_data"] = invalid
                self.write(self.experiment_path, self.experiment)
                with self.assertRaisesRegex(launch.LaunchError, "public TRAIN/DEV"):
                    self.inspect()

    def test_missing_extra_or_changed_cache_bytes_rejected(self):
        path = Path(self.experiment["cache"][0]["tensor_path"])
        old = path.read_bytes()
        path.write_bytes(old + b"changed")
        with self.assertRaisesRegex(launch.LaunchError, "cache bytes"):
            self.inspect()
        path.write_bytes(old)
        extra = self.cache / "unexpected.txt"
        extra.write_text("extra")
        with self.assertRaisesRegex(launch.LaunchError, "extra members"):
            self.inspect()
        extra.unlink()
        path.unlink()
        with self.assertRaises(ValueError):
            self.inspect()

    def test_overlay_scan_budget_checked_before_any_byte_hash(self):
        with patch.object(launch, "MAX_OVERLAY_FILES", 0), \
             patch.object(launch, "overlay_file_hash", side_effect=AssertionError("cannot hash oversized scan")):
            with self.assertRaisesRegex(launch.LaunchError, "budget"):
                launch.overlay_inventory(self.overlay)
        with patch.object(launch, "MAX_OVERLAY_BYTES", 1), \
             patch.object(launch, "overlay_file_hash", side_effect=AssertionError("cannot hash oversized scan")):
            with self.assertRaisesRegex(launch.LaunchError, "budget"):
                launch.overlay_inventory(self.overlay)

    def test_overlay_directory_namespace_aliases_rejected_before_hashes(self):
        directory = self.overlay / "pkg"
        directory.mkdir()
        # A mocked directory listing isolates the Linux case-alias condition on
        # Windows too; actual regular-directory checks are retained.
        def regular(path, *, directory=False):
            candidate = Path(path)
            if candidate.as_posix() == (self.overlay / "PKG").as_posix():
                return candidate
            return original(candidate, directory=directory)
        original = launch.overlay_regular
        with patch.object(launch.os, "walk", return_value=iter([(str(self.overlay), ["pkg", "PKG"], [])])), \
             patch.object(launch, "overlay_regular", side_effect=regular), \
             patch.object(launch, "overlay_file_hash", side_effect=AssertionError("cannot hash aliased namespace")):
            with self.assertRaisesRegex(launch.LaunchError, "aliased namespace"):
                launch.overlay_inventory(self.overlay)

    def test_final_source_path_replacement_or_cache_membership_change_refused(self):
        inputs = self.inspect()
        extra = self.cache / "new-member.txt"
        extra.write_text("new public fixture member")
        with self.assertRaisesRegex(launch.LaunchError, "membership changed"):
            launch.verify_inputs_unchanged(inputs)
        extra.unlink()
        original = self.code["backbone.py"]
        target = self.root / "same-byte-replacement.py"
        target.write_bytes(original.read_bytes())
        original.unlink()
        try:
            original.symlink_to(target)
        except OSError:
            # Windows may deny symlink creation. Exercise the regular-path
            # boundary directly; Linux runs the physical replacement above.
            original.write_bytes(target.read_bytes())
            old_regular = handoff._regular
            def changed_path(path, **kwargs):
                if Path(path) == original:
                    raise handoff.HandoffError("simulated persistent same-byte symlink")
                return old_regular(path, **kwargs)
            with patch.object(handoff, "_regular", side_effect=changed_path):
                with self.assertRaisesRegex(ValueError, "same-byte symlink"):
                    launch.verify_inputs_unchanged(inputs)
        else:
            self.assertEqual(handoff._hash(original), handoff._hash(target))
            with self.assertRaisesRegex(ValueError, "links"):
                launch.verify_inputs_unchanged(inputs)

    def test_overlay_pin_command_is_strictly_stdlib(self):
        guard = "import builtins,runpy,sys; original=builtins.__import__; " \
            "exec('def guarded(name,*a,**k):\\n if name.split(\".\")[0] in (\"src\",\"torch\",\"transformers\",\"numpy\"):\\n  raise AssertionError(\"non-stdlib inventory import\")\\n return original(name,*a,**k)'); " \
            "builtins.__import__=guarded; sys.argv=[sys.argv[1],\"overlay-pin\",\"--overlay\",sys.argv[2]]; runpy.run_path(sys.argv[0],run_name=\"__main__\")"
        # Literal newlines are needed in exec's nested Python definition.
        guard = guard.replace("\\\\n", "\\n")
        result = subprocess.run([sys.executable, "-c", guard, str(SCRIPT), str(self.overlay)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["file_count"], 1)

    def test_public_identity_records_guest_kernel_without_private_namespace_claim(self):
        status = self.root / "fixture-proc-status"
        status.write_text("Name:\tfixture\nUid:\t1905\t1905\t1905\t1905\nGid:\t100\t100\t100\t100\n")
        identity = launch.process_identity(1905, 100, status=status)
        self.assertEqual(identity["kernel_status_uid_fields"], [1905] * 4)
        self.assertFalse(identity["private_namespace_claim"])
        self.assertNotIn("uid_map", identity)

    def fake_closed_package(self, **kwargs):
        package = kwargs["output_directory"]
        package.mkdir()
        (package / "manifest.json").write_text("public orchestration fixture")
        return {"manifest_sha256": "a" * 64, "closure_sha256": "b" * 64,
                "status": "CLOSED_SOURCE_VERIFIED", "n0_approved": False}

    def fake_admission(self, package, **kwargs):
        return SimpleNamespace(package_directory=package, binding=handoff._seal({
            "scope": "orchestration fixture only, no source/tensor proof",
            "consumer_retokenized": False, "publisher_checkpoint_loaded": False,
            "n0_approved": False, "mechanics_only": False}))

    def test_orchestration_uses_exact_external_pins_and_records_measured_resources(self):
        inputs = self.inspect()
        resource = SimpleNamespace(RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=1024))
        fixture_modules = {"torch": SimpleNamespace(__version__="fixture-only"),
            "transformers": SimpleNamespace(__version__="fixture-only"), "resource": resource}
        with patch.object(handoff, "export_public_features", side_effect=self.fake_closed_package) as exporter, \
             patch.object(handoff, "import_public_features", side_effect=self.fake_admission) as importer, \
             patch.dict(sys.modules, fixture_modules):
            result = launch.close_handoff(inputs, self.run_root, context={"scope": "public fixture only"})
        self.assertEqual(exporter.call_args.kwargs["expected_experiment_sha256"], inputs.experiment_sha256)
        self.assertEqual(importer.call_args.kwargs,
            {"expected_manifest_sha256": "a" * 64, "expected_closure_sha256": "b" * 64})
        self.assertEqual(result["process_peak_rss_bytes"], 1024 * 1024)
        self.assertFalse(result["model_constructed"])
        self.assertFalse(result["training_run"])
        self.assertFalse(result["n0_approved"])
        self.assertTrue((self.run_root / "import-admission.json").is_file())
        self.assertTrue((self.run_root / "handoff-run.json").is_file())
        with self.assertRaisesRegex(launch.LaunchError, "fresh empty"):
            launch.close_handoff(inputs, self.run_root, context={})

    def test_persistent_source_change_during_import_refuses_admission_receipt(self):
        inputs = self.inspect()
        def tamper(package, **kwargs):
            self.code["backbone.py"].write_text("changed during mocked import")
            return self.fake_admission(package, **kwargs)
        with patch.object(handoff, "export_public_features", side_effect=self.fake_closed_package), \
             patch.object(handoff, "import_public_features", side_effect=tamper), \
             patch.object(handoff, "write_import_receipt") as writer:
            with self.assertRaisesRegex(launch.LaunchError, "changed during closure"):
                launch.close_handoff(inputs, self.run_root, context={"scope": "public fixture"})
        writer.assert_not_called()
        self.assertFalse((self.run_root / "handoff-run.json").exists())

    def test_single_stage_launcher_contract_and_bash_syntax(self):
        text = SBATCH.read_text()
        self.assertEqual(text.count('"$UDOCKER" run '), 1)
        for denied in ("pip install", "udocker setup", "udocker create", "udocker pull", "unshare"):
            self.assertNotIn(denied, text)
        self.assertIn("--mem=16G", text)
        self.assertIn('"$(id -un)" == mxrayan', text)
        self.assertIn('"$(id -u)" != 0', text)
        self.assertIn('"${13}"', text)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
        self.assertIn("12b583048ddf343e5724b92678b43bc7e247b46aa5f016653f9825e7da0d49df", text)
        self.assertIn("6b565c38f969ceee2143f28fc29f760ec84c6f2c8cdc53dc29fd953ac3f93270", text)
        bash = shutil.which("bash")
        bundled_git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if bundled_git_bash.is_file():
            bash = str(bundled_git_bash)
        if bash:
            result = subprocess.run([bash, "-n", str(SBATCH)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
