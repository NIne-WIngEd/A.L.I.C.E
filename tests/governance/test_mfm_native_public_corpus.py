"""Global lineage and byte-transfer checks for public Stage A candidates."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha1, sha256
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from scripts.mfm.native_hf_acquire import (
    fetch_exact_file, freeze_selection, inventory_exact_repo, selection_template,
    verify_complete_acquisitions,
)
from scripts.mfm.native_public_corpus import _row_sha256, reconcile, stage_shard, verify_shard


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "configs/mfm/native_foundation_source_candidates_v1.json"
N0 = ROOT / "configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, record: dict) -> str:
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return digest(path)


class TestPublicCorpus(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.candidate_sha = digest(CANDIDATE)
        self.n0_sha = digest(N0)
        self.sources = {row["source_id"]: row for row in
                        json.loads(N0.read_text("utf-8"))["sources"]}

    def frozen_repo(self, family, api, name):
        tree_file = self.base / f"{name}-tree.json"
        tree = inventory_exact_repo(candidate_file=CANDIDATE,
                                    candidate_sha=self.candidate_sha,
                                    n0_file=N0, n0_sha=self.n0_sha,
                                    candidate_id=family, output=tree_file, api=api)
        selection = selection_template(tree)
        selection["repo_tree_sha256"] = digest(tree_file)
        selection["review_record"] = "fixture source file inventory reviewed"
        selection_file = self.base / f"{name}-selection.json"
        selection_sha = freeze(selection_file, selection)
        frozen_file = self.base / f"{name}-frozen.json"
        result = freeze_selection(tree_file=tree_file, tree_sha=digest(tree_file),
                                  selection_file=selection_file,
                                  selection_sha=selection_sha, output=frozen_file)
        return frozen_file, digest(frozen_file), tree, result

    def source_row(self, family, row_id, text, url, *, license_value=None,
                   provenance="original source provenance"):
        source = self.sources[family]
        return {"id": row_id, "text": text,
                "metadata": {"license": license_value or source["allowed_license_values"][0],
                             "url": url, "provenance": provenance}}

    def acquire_and_stage(self, family, name, rows):
        source = self.sources[family]
        raw_bytes = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                             for row in rows)
        repo_path = f"data/{name}.jsonl"
        acquire_dir = self.base / name

        class API:
            def dataset_info(self, *, repo_id, revision):
                return SimpleNamespace(sha=revision)

            def list_repo_tree(self, **kwargs):
                yield SimpleNamespace(path=".gitattributes", size=8,
                                      blob_id="b" * 40, lfs=None)
                yield SimpleNamespace(path=repo_path, size=len(raw_bytes),
                                      blob_id="a" * 40,
                                      lfs=SimpleNamespace(sha256=sha256(raw_bytes).hexdigest(),
                                                          size=len(raw_bytes)))

            def get_paths_info(self, repo_id, **kwargs):
                if kwargs["paths"] != [repo_path]:
                    raise AssertionError("acquisition did not request the exact file path")
                return [SimpleNamespace(path=repo_path, size=len(raw_bytes), blob_id="a" * 40,
                                        lfs=SimpleNamespace(sha256=sha256(raw_bytes).hexdigest(),
                                                            size=len(raw_bytes)))]

        def download(*, filename, local_dir, **kwargs):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw_bytes)
            return str(path)

        frozen_file, frozen_sha, _, _ = self.frozen_repo(family, API(), name)
        acquired = fetch_exact_file(candidate_file=CANDIDATE,
                                    candidate_sha=self.candidate_sha,
                                    n0_file=N0, n0_sha=self.n0_sha,
                                    candidate_id=family, upstream_repo_path=repo_path,
                                    frozen_inventory_file=frozen_file,
                                    frozen_inventory_sha=frozen_sha,
                                    output=acquire_dir, api=API(), downloader=download)
        receipt_file = acquire_dir / "acquisition_receipt.json"
        raw = acquire_dir / repo_path
        staged = self.base / "staged" / name
        receipt = stage_shard(candidate_file=CANDIDATE,
                              candidate_sha=self.candidate_sha,
                              n0_file=N0, n0_sha=self.n0_sha,
                              candidate_id=family, raw_file=raw,
                              raw_sha=acquired["downloaded_sha256"],
                              upstream_repo_path=repo_path, format_name="jsonl",
                              output=staged, acquisition_receipt=receipt_file,
                              acquisition_sha=digest(receipt_file), part_bytes=200)
        self.assertEqual(receipt, verify_shard(staged / "shard_receipt.json", raw_file=raw))
        self.assertEqual(receipt, stage_shard(candidate_file=CANDIDATE,
                                              candidate_sha=self.candidate_sha,
                                              n0_file=N0, n0_sha=self.n0_sha,
                                              candidate_id=family, raw_file=raw,
                                              raw_sha=acquired["downloaded_sha256"],
                                              upstream_repo_path=repo_path,
                                              format_name="jsonl", output=staged,
                                              acquisition_receipt=receipt_file,
                                              acquisition_sha=digest(receipt_file),
                                              part_bytes=200))
        return staged, receipt, raw

    def list_and_exclusions(self, staged, *, excluded_hashes=()):
        links = [{"path": str((directory / "shard_receipt.json").relative_to(self.base)),
                  "sha256": digest(directory / "shard_receipt.json")}
                 for directory in staged]
        list_file = self.base / "receipt_list.json"
        list_sha = freeze(list_file, {"schema": "mfm-native-public-receipt-list-v1",
                                      "receipts": links})
        excluded = self.base / "exclusions.json"
        excluded_sha = freeze(excluded,
                              {"schema": "mfm-native-public-exclusions-v1",
                               "reserved_evaluations": ["LongMemEval-V2", "LoCoMo", "CareCall"],
                               "candidate_ids": [], "content_sha256s": list(excluded_hashes),
                               "source_urls": [], "upstream_item_refs": []})
        return list_file, list_sha, excluded, excluded_sha

    def test_cross_source_lineage_dedup_and_no_document_truncation(self):
        family_a = "common_pile_wikimedia_filtered"
        family_b = "common_pile_arxiv_abstracts_filtered"
        very_long = "Long document content " * 400
        first, receipt_a, _ = self.acquire_and_stage(family_a, "first", [
            self.source_row(family_a, "a1", very_long, "https://example.org/one"),
            self.source_row(family_a, "a2", "same exact content", "https://example.org/two"),
            self.source_row(family_a, "a3", "unknown license", "https://example.org/three",
                            license_value="unlisted license"),
            self.source_row(family_a, "a4", "missing provenance", "https://example.org/four",
                            provenance=""),
            self.source_row(family_a, "a5", "bad URL", {"opaque": "url"})])
        second, receipt_b, _ = self.acquire_and_stage(family_b, "second", [
            self.source_row(family_b, "b1", "different text, same URL",
                            "https://example.org/one"),
            self.source_row(family_b, "b2", "same exact content",
                            "https://example.org/five"),
            self.source_row(family_b, "b3", "unique content",
                            "https://example.org/six")])
        self.assertEqual(receipt_a["counts"]["seen"], 5)
        self.assertEqual(receipt_a["counts"]["held_license"], 1)
        self.assertEqual(receipt_a["counts"]["held_provenance"], 1)
        self.assertEqual(receipt_a["counts"]["held_invalid_url"], 1)
        self.assertEqual(receipt_a["counts"]["candidates"], 2)
        held = [json.loads(line) for line in (first / "held_rows.jsonl").read_text().splitlines()]
        self.assertEqual({row["reason"] for row in held},
                         {"held_license", "held_provenance", "held_invalid_url"})
        self.assertGreater(len(receipt_a["parts"]), 1)
        part = next(first.glob("candidates-*.jsonl"))
        self.assertEqual(json.loads(part.read_text("utf-8"))["text"], very_long.strip())

        list_file, list_sha, excluded, excluded_sha = self.list_and_exclusions([first, second])
        result = reconcile(receipt_list=list_file, receipt_list_sha=list_sha,
                           exclusions=excluded, exclusions_sha=excluded_sha,
                           output=self.base / "reconciled")
        self.assertEqual(result["totals"]["rows"], 5)
        self.assertEqual(result["totals"]["exact_duplicate_hold"], 1)
        rows = [json.loads(line) for line in
                (self.base / "reconciled/candidate_index.jsonl").read_text("utf-8").splitlines()]
        by_url = {url: [row for row in rows if row["source_url"] == url]
                  for url in {row["source_url"] for row in rows}}
        linked = by_url["https://example.org/one"]
        self.assertEqual({row["lineage_component"] for row in linked},
                         {linked[0]["lineage_component"]})
        self.assertEqual({row["split_candidate"] for row in linked},
                         {linked[0]["split_candidate"]})
        duplicates = [row for row in rows if row["content_sha256"] ==
                      sha256(b"same exact content").hexdigest()]
        self.assertEqual({row["lineage_component"] for row in duplicates},
                         {duplicates[0]["lineage_component"]})
        self.assertTrue(all(row["training_admitted"] is False for row in rows))

    def test_exclusion_propagates_across_shards_and_transfer_hashes(self):
        family = "common_pile_wikimedia_filtered"
        first, _, raw = self.acquire_and_stage(family, "one", [
            self.source_row(family, "a1", "blocked document", "https://example.org/source")])
        second, _, _ = self.acquire_and_stage(family, "two", [
            self.source_row(family, "b1", "derived content", "https://example.org/source")])
        hash_to_block = sha256(b"blocked document").hexdigest()
        receipt_list, list_sha, exclusion, exclusion_sha = self.list_and_exclusions(
            [first, second], excluded_hashes=(hash_to_block,))
        result = reconcile(receipt_list=receipt_list, receipt_list_sha=list_sha,
                           exclusions=exclusion, exclusions_sha=exclusion_sha,
                           output=self.base / "blocked_global")
        self.assertEqual(result["totals"]["excluded_hold"], 2)
        with raw.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_shard(first / "shard_receipt.json", raw_file=raw)
        with next(first.glob("candidates-*.jsonl")).open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            reconcile(receipt_list=receipt_list, receipt_list_sha=list_sha,
                      exclusions=exclusion, exclusions_sha=exclusion_sha,
                      output=self.base / "tampered_global")
        self.assertFalse((self.base / "tampered_global").exists())

    def test_acquisition_refuses_unverified_repository_object(self):
        family = "common_pile_wikimedia_filtered"
        source = self.sources[family]
        class NoHashAPI:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha=source["revision"])
            def list_repo_tree(self, **kwargs):
                yield SimpleNamespace(path="data/f.parquet", size=10, blob_id=None,
                                      lfs=None)

        with self.assertRaisesRegex(ValueError, "no verifiable 40-hex object ID"):
            self.frozen_repo(family, NoHashAPI(), "invalid")

    def test_acquisition_verifies_regular_git_blob(self):
        family = "common_pile_wikimedia_filtered"
        source = self.sources[family]
        data = b"small uncompressed repository file\n"
        blob = sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()

        class GitAPI:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha=source["revision"])

            def list_repo_tree(self, **kwargs):
                yield SimpleNamespace(path="data/f.jsonl", size=len(data), blob_id=blob,
                                      lfs=None)

            def get_paths_info(self, *args, **kwargs):
                return [SimpleNamespace(path="data/f.jsonl", size=len(data),
                                        blob_id=blob, lfs=None)]

        def download(*, local_dir, filename, **kwargs):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return str(path)

        frozen_file, frozen_sha, _, _ = self.frozen_repo(family, GitAPI(), "git")
        receipt = fetch_exact_file(candidate_file=CANDIDATE,
                                   candidate_sha=self.candidate_sha,
                                   n0_file=N0, n0_sha=self.n0_sha, candidate_id=family,
                                   upstream_repo_path="data/f.jsonl",
                                   frozen_inventory_file=frozen_file,
                                   frozen_inventory_sha=frozen_sha,
                                   output=self.base / "git_blob", api=GitAPI(),
                                   downloader=download)
        self.assertEqual(receipt["upstream_git_blob_id"], blob)
        self.assertEqual(receipt["downloaded_sha256"], sha256(data).hexdigest())
        self.assertEqual(receipt["frozen_repo_inventory_sha256"], frozen_sha)

    def test_pinned_tree_is_exhausted_and_selection_accounts_for_every_file(self):
        family = "common_pile_wikimedia_filtered"
        revision = self.sources[family]["revision"]
        paths = [f"data/train-{i:05d}.parquet" for i in range(1207)]

        class PaginatedAPI:
            def dataset_info(self, **kwargs):
                self_outer.assertEqual(kwargs["revision"], revision)
                return SimpleNamespace(sha=revision)

            def list_repo_tree(self, **kwargs):
                self_outer.assertEqual(kwargs["revision"], revision)
                self_outer.assertTrue(kwargs["recursive"])
                self_outer.assertFalse(kwargs["expand"])
                self_outer.assertEqual(kwargs["repo_type"], "dataset")
                yield SimpleNamespace(path="data", tree_id="f" * 40)
                for path in paths:
                    yield SimpleNamespace(path=path, size=2, blob_id="c" * 40,
                                          lfs=SimpleNamespace(sha256="d" * 64, size=2))
                yield SimpleNamespace(path="README.md", size=10,
                                      blob_id="e" * 40, lfs=None)
                yield SimpleNamespace(path="extra.csv", size=100,
                                      blob_id="a" * 40, lfs=None)

        self_outer = self
        tree_file = self.base / "many-tree.json"
        tree = inventory_exact_repo(candidate_file=CANDIDATE,
                                    candidate_sha=self.candidate_sha,
                                    n0_file=N0, n0_sha=self.n0_sha,
                                    candidate_id=family, output=tree_file,
                                    api=PaginatedAPI())
        self.assertEqual(tree["file_count"], 1209)
        self.assertEqual(tree["files"][0]["path"], "README.md")
        self.assertEqual(tree["files"][-1]["path"], "extra.csv")
        self.assertEqual(tree["files"][-1]["suggestion"], "manual_review_required")
        selection = selection_template(tree)
        selection["repo_tree_sha256"] = digest(tree_file)
        selection["review_record"] = "fixture repository tree and exclusions reviewed"
        selection_path = self.base / "many-selection.json"
        selection_sha = freeze(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            freeze_selection(tree_file=tree_file, tree_sha=digest(tree_file),
                             selection_file=selection_path, selection_sha=selection_sha,
                             output=self.base / "incomplete-frozen.json")
        selection["files"][-1] = {"path": "extra.csv", "decision": "exclude",
                                  "reason": "CSV needs separate parser and terms review"}
        selection_sha = freeze(selection_path, selection)
        frozen = freeze_selection(tree_file=tree_file, tree_sha=digest(tree_file),
                                  selection_file=selection_path,
                                  selection_sha=selection_sha,
                                  output=self.base / "many-frozen.json")
        self.assertEqual(frozen["selected_data_count"], len(paths))
        self.assertEqual(frozen["selected_data_bytes"], 2 * len(paths))
        self.assertEqual([row["path"] for row in frozen["files"]
                          if row["decision"] == "exclude"], ["README.md", "extra.csv"])
        selection["files"].pop()
        selection_sha = freeze(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "every repository file"):
            freeze_selection(tree_file=tree_file, tree_sha=digest(tree_file),
                             selection_file=selection_path,
                             selection_sha=selection_sha,
                             output=self.base / "missing-frozen.json")

    def test_inventory_refuses_duplicate_paths_and_revision_drift(self):
        family = "common_pile_news_filtered"
        revision = self.sources[family]["revision"]

        class DuplicateAPI:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha=revision)

            def list_repo_tree(self, **kwargs):
                for _ in range(2):
                    yield SimpleNamespace(path="data/train.parquet", size=2,
                                          blob_id="e" * 40, lfs=None)

        with self.assertRaisesRegex(ValueError, "duplicate repository tree path"):
            self.frozen_repo(family, DuplicateAPI(), "duplicate")

        class DriftAPI(DuplicateAPI):
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha="f" * 40)

        with self.assertRaisesRegex(ValueError, "exact pinned dataset commit"):
            self.frozen_repo(family, DriftAPI(), "drift")

    def test_inventory_is_available_for_all_pinned_n0_source_families(self):
        class EverySourceAPI:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha=kwargs["revision"])

            def list_repo_tree(self, **kwargs):
                yield SimpleNamespace(path="data/train.parquet", size=20,
                                      blob_id="a" * 40,
                                      lfs=SimpleNamespace(sha256="c" * 64, size=20))

        self.assertEqual(len(self.sources), 21)
        for family in self.sources:
            with self.subTest(family=family):
                frozen, _, tree, selection = self.frozen_repo(family, EverySourceAPI(), family)
                self.assertEqual(tree["dataset_repo_id"], self.sources[family]["repo_id"])
                self.assertEqual(selection["revision"], self.sources[family]["revision"])
                self.assertTrue(frozen.is_file())

    def test_acquisition_coverage_requires_every_selected_shard_once(self):
        family = "common_pile_arxiv_abstracts_filtered"
        revision = self.sources[family]["revision"]
        data = {"data/first.jsonl": b'{"one":1}\n',
                "data/second.jsonl": b'{"two":2}\n'}

        def entry(path, raw):
            return SimpleNamespace(path=path, size=len(raw), blob_id="a" * 40,
                                   lfs=SimpleNamespace(sha256=sha256(raw).hexdigest(),
                                                       size=len(raw)))

        class API:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha=revision)

            def list_repo_tree(self, **kwargs):
                for path, raw in data.items():
                    yield entry(path, raw)
                yield SimpleNamespace(path="README.md", size=10,
                                      blob_id="b" * 40, lfs=None)

            def get_paths_info(self, repo_id, *, paths, **kwargs):
                return [entry(paths[0], data[paths[0]])]

        def download(*, filename, local_dir, **kwargs):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data[filename])
            return str(path)

        frozen_file, frozen_sha, _, _ = self.frozen_repo(family, API(), "coverage")
        class ChangedAPI(API):
            def get_paths_info(self, repo_id, *, paths, **kwargs):
                altered = entry(paths[0], data[paths[0]])
                altered.lfs.sha256 = "f" * 64
                return [altered]

        with self.assertRaisesRegex(ValueError, "differs from frozen repo tree"):
            fetch_exact_file(candidate_file=CANDIDATE, candidate_sha=self.candidate_sha,
                             n0_file=N0, n0_sha=self.n0_sha, candidate_id=family,
                             upstream_repo_path="data/first.jsonl",
                             output=self.base / "changed",
                             frozen_inventory_file=frozen_file,
                             frozen_inventory_sha=frozen_sha,
                             api=ChangedAPI(),
                             downloader=lambda **_: self.fail("unexpected download"))
        with self.assertRaisesRegex(ValueError, "not included"):
            fetch_exact_file(candidate_file=CANDIDATE, candidate_sha=self.candidate_sha,
                             n0_file=N0, n0_sha=self.n0_sha, candidate_id=family,
                             upstream_repo_path="README.md", output=self.base / "metadata",
                             frozen_inventory_file=frozen_file,
                             frozen_inventory_sha=frozen_sha,
                             api=API(), downloader=download)
        receipt_paths = []
        for index, path in enumerate(data):
            output = self.base / f"shard-{index}"
            fetch_exact_file(candidate_file=CANDIDATE,
                             candidate_sha=self.candidate_sha,
                             n0_file=N0, n0_sha=self.n0_sha, candidate_id=family,
                             upstream_repo_path=path, output=output,
                             frozen_inventory_file=frozen_file,
                             frozen_inventory_sha=frozen_sha,
                             api=API(), downloader=download)
            receipt = output / "acquisition_receipt.json"
            receipt_paths.append({"path": str(receipt.relative_to(self.base)),
                                  "sha256": digest(receipt)})
        receipt_list = self.base / "acquisitions.json"
        partial_sha = freeze(receipt_list, {"schema": "mfm-native-hf-acquisition-receipt-list-v1",
                                            "frozen_repo_inventory_sha256": frozen_sha,
                                            "receipts": receipt_paths[:1]})
        with self.assertRaisesRegex(ValueError, "omits 1 selected shards"):
            verify_complete_acquisitions(frozen_inventory_file=frozen_file,
                                         frozen_inventory_sha=frozen_sha,
                                         receipt_list=receipt_list,
                                         receipt_list_sha=partial_sha,
                                         output=self.base / "partial.json")
        complete_sha = freeze(receipt_list,
                              {"schema": "mfm-native-hf-acquisition-receipt-list-v1",
                               "frozen_repo_inventory_sha256": frozen_sha,
                               "receipts": receipt_paths})
        result = verify_complete_acquisitions(frozen_inventory_file=frozen_file,
                                              frozen_inventory_sha=frozen_sha,
                                              receipt_list=receipt_list,
                                              receipt_list_sha=complete_sha,
                                              output=self.base / "coverage.json")
        self.assertEqual(result["shard_count"], 2)
        self.assertFalse(result["training_admitted"])

    def test_parquet_scalar_hold_digest_is_order_independent(self):
        row = {"metadata": {"archive_date": date(2024, 3, 5),
                            "observed_at": datetime(2024, 3, 5, tzinfo=timezone.utc),
                            "opaque": b"\xff\x00", "price": Decimal("1.20")},
               "id": "row"}
        reordered = {"id": "row", "metadata": dict(reversed(list(row["metadata"].items())))}
        self.assertEqual(_row_sha256(row), _row_sha256(reordered))
        self.assertRegex(_row_sha256(row), r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
