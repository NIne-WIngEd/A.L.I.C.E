import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

DRIVER = (
    Path(__file__).parents[3]
    / "scripts/eipm/n0/downstream_full_stack_graph_evaluator_v0_1.py"
)
SPEC = importlib.util.spec_from_file_location("full_stack_graph_eval_v01", DRIVER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FullStackGraphEvaluatorBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40

        self.file_roles = sorted(
            MOD.REQUIRED_STACK_ARTIFACTS - MOD.DIRECTORY_ARTIFACTS
        )
        for role in self.file_roles:
            (self.root / f"{role}.bin").write_text(
                f"{role}\n", encoding="utf-8"
            )

        for role in sorted(MOD.DIRECTORY_ARTIFACTS):
            directory = self.root / role
            directory.mkdir()
            (directory / "weights.bin").write_text(
                f"{role}-weights\n", encoding="utf-8"
            )
            nested = directory / "nested"
            nested.mkdir()
            (nested / "metadata.json").write_text(
                json.dumps({"role": role}), encoding="utf-8"
            )

        self.latent = self.root / "latent.safetensors"
        self.latent.write_bytes(b"latent-checkpoint")
        self.evaluation = self.root / "evaluation.jsonl"
        self.evaluation.write_text('{"id":"row-1"}\n', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def artifact_spec(self, role):
        if role in MOD.DIRECTORY_ARTIFACTS:
            path = self.root / role
            digest, _rows = MOD.sha256_directory_tree(path)
            return {
                "path": str(path),
                "artifact_type": "directory_tree",
                "sha256": digest,
            }
        path = self.root / f"{role}.bin"
        return {
            "path": str(path),
            "artifact_type": "file",
            "sha256": MOD.sha256_file(path),
        }

    def manifest(self):
        return {
            "protocol_version": MOD.PROTOCOL_VERSION,
            "source_revision": self.revision,
            "invariants": {
                "diagnostic_only": True,
                "training_enabled": False,
                "ratifies_repair": False,
                "promotion_authorized": False,
                "scale_authorized": False,
                "graph_checkpoint_is_only_arm_variant": True,
                "proxy_graph_transform_used": False,
            },
            "external_inputs": {
                "latent_checkpoint": {"sha256": MOD.sha256_file(self.latent)},
                "evaluation_set": {"sha256": MOD.sha256_file(self.evaluation)},
            },
            "artifacts": {
                role: self.artifact_spec(role)
                for role in sorted(MOD.REQUIRED_STACK_ARTIFACTS)
            },
        }

    def write_manifest(self, manifest=None):
        path = self.root / "stack.json"
        path.write_text(
            json.dumps(self.manifest() if manifest is None else manifest),
            encoding="utf-8",
        )
        return path

    def test_full_stack_manifest_binds_files_and_directory_trees(self):
        bound = MOD.validate_stack_manifest(
            self.write_manifest(),
            observed_source_revision=self.revision,
        )
        self.assertEqual(set(bound["artifacts"]), MOD.REQUIRED_STACK_ARTIFACTS)
        self.assertRegex(bound["stack_binding_fingerprint"], r"^[0-9a-f]{64}$")
        for role in MOD.DIRECTORY_ARTIFACTS:
            self.assertEqual(
                bound["artifacts"][role]["artifact_type"], "directory_tree"
            )

    def test_directory_tree_mutation_is_detected(self):
        manifest_path = self.write_manifest()
        target = self.root / "semantic_checkpoint" / "weights.bin"
        target.write_text("mutated\n", encoding="utf-8")
        with self.assertRaises(MOD.BindingError):
            MOD.validate_stack_manifest(
                manifest_path,
                observed_source_revision=self.revision,
            )

    def test_source_revision_mismatch_fails(self):
        with self.assertRaises(MOD.BindingError):
            MOD.validate_stack_manifest(
                self.write_manifest(),
                observed_source_revision="b" * 40,
            )

    def test_missing_full_stack_artifact_fails(self):
        manifest = self.manifest()
        del manifest["artifacts"]["fusion_ratification"]
        with self.assertRaises(MOD.BindingError):
            MOD.validate_stack_manifest(
                self.write_manifest(manifest),
                observed_source_revision=self.revision,
            )

    def test_graph_cannot_be_smuggled_into_common_stack(self):
        manifest = self.manifest()
        manifest["artifacts"]["evidence_graph"] = self.artifact_spec("latent_config")
        with self.assertRaises(MOD.BindingError):
            MOD.validate_stack_manifest(
                self.write_manifest(manifest),
                observed_source_revision=self.revision,
            )

    def test_wrong_directory_artifact_type_fails(self):
        manifest = self.manifest()
        manifest["artifacts"]["tokenizer_dir"]["artifact_type"] = "file"
        with self.assertRaises(MOD.BindingError):
            MOD.validate_stack_manifest(
                self.write_manifest(manifest),
                observed_source_revision=self.revision,
            )

    def test_external_latent_hash_is_enforced(self):
        bound = MOD.validate_stack_manifest(
            self.write_manifest(),
            observed_source_revision=self.revision,
        )
        self.latent.write_bytes(b"changed")
        with self.assertRaises(MOD.BindingError):
            MOD.verify_external_file(
                role="latent_checkpoint",
                path=self.latent,
                binding=bound,
            )

    def test_symlink_in_directory_tree_is_rejected_when_supported(self):
        directory = self.root / "semantic_checkpoint"
        link = directory / "linked.bin"
        try:
            link.symlink_to(directory / "weights.bin")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this platform")
        with self.assertRaises(MOD.BindingError):
            MOD.sha256_directory_tree(directory)


if __name__ == "__main__":
    unittest.main()
