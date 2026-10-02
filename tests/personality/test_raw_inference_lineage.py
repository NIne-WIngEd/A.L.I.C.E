"""Public synthetic archive fixtures prove custody mechanics, never readiness.

No approved archive, private identity payload or evaluation FINAL is opened.
All fixture pins are hashes of these independently created public ZIP bytes.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from src.alice_personality.n1 import raw_inference_lineage as r


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def fixture_rows():
    return [{"proposal_id": "public.raw.b", "provenance_class": "E-INF",
             "behavioral_proposal": "Fictional public inference, never historical truth.",
             "supporting_EINF_ids": ["dangling.must.not.be.harvested"],
             "generator_has_acceptance_authority": False, "model_training_authority": False},
            {"proposal_id": "public.raw.a", "provenance_class": "E-INF",
             "behavioral_proposal": "Another fictional fixture inference."}]


def fixture_members(*, rows=None, row_bytes=None, root="public-package", ledger_style="relative"):
    prefix = root + "/" if root else ""
    if row_bytes is None:
        row_bytes = b"".join(canonical(row) + b"\n" for row in (fixture_rows() if rows is None else rows))
    members = {prefix + "einf_proposals.jsonl": row_bytes,
               prefix + "generation_manifest.json": canonical({"fixture_only": True,
                                                              "fixture_row_count": 2}),
               prefix + "asyn_proposals.jsonl": b"DO NOT OPEN ASYN FIXTURE SENTINEL",
               prefix + "historical_unknown_competitors.jsonl": b"DO NOT OPEN UNKNOWN FIXTURE SENTINEL",
               prefix + "counterfactual_competitors.jsonl": b"DO NOT OPEN ALTERNATIVES FIXTURE SENTINEL",
               prefix + "sealed_FINAL_evaluation.jsonl": b"DO NOT OPEN FINAL FIXTURE SENTINEL"}
    ledger = []
    for name, payload in members.items():
        named = name[len(prefix):] if ledger_style == "relative" else name
        ledger.append(sha256(payload).hexdigest() + "  " + named)
    members[prefix + "SHA256SUMS.txt"] = ("\n".join(ledger) + "\n").encode()
    return members


def refresh_ledger(members, *, root="public-package"):
    prefix = root + "/" if root else ""
    ledger_name = prefix + "SHA256SUMS.txt"
    members[ledger_name] = b"".join((sha256(data).hexdigest() + "  " + name[len(prefix):]
                                    + "\n").encode() for name, data in members.items()
                                   if name != ledger_name)


class RawInferenceLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def archive(self, members=None, *, extra_entries=(), compression=zipfile.ZIP_DEFLATED):
        self.serial += 1
        path = self.root / f"public-fixture-{self.serial}.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=compression) as archive:
                for name, data in (fixture_members() if members is None else members).items():
                    archive.writestr(name, data)
                for name, data in extra_entries:
                    archive.writestr(name, data)
        return path, sha256(path.read_bytes()).hexdigest()

    def derive(self, members=None, **kwargs):
        path, digest = self.archive(members, **kwargs)
        return r.derive_raw_inference_registry(path, expected_archive_sha256=digest)

    def refused(self, members=None, reason=None, **kwargs):
        path, digest = self.archive(members, **kwargs)
        with self.assertRaises(r.RawInferenceLineageError) as result:
            r.derive_raw_inference_registry(path, expected_archive_sha256=digest)
        message = str(result.exception)
        self.assertNotIn("public.raw", message)
        self.assertNotIn("Fictional", message)
        self.assertNotIn(str(self.root), message)
        if reason:
            self.assertEqual(reason, message)

    def test_exact_source_registry_and_sanitized_sealed_receipt(self):
        path, digest = self.archive()
        before = path.read_bytes()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            registry, receipt = r.derive_raw_inference_registry(path, expected_archive_sha256=digest)
        self.assertEqual(registry, {"EINF": ["public.raw.a", "public.raw.b"]})
        self.assertEqual(receipt["schema"], "alice-personality-raw-inference-lineage-v1")
        self.assertEqual(receipt["state"], "DERIVED_UNQUALIFIED")
        self.assertEqual(receipt["source_archive_sha256"], digest)
        self.assertEqual(receipt["source_member_path"], "public-package/einf_proposals.jsonl")
        self.assertEqual(receipt["raw_inference_count"], 2)
        self.assertEqual(receipt["registry_sha256"], sha256(canonical(registry)).hexdigest())
        self.assertEqual(receipt["code_sha256"], sha256(Path(r.__file__).read_bytes()).hexdigest())
        seal = receipt.pop("receipt_sha256")
        self.assertEqual(seal, sha256(canonical(receipt)).hexdigest())
        self.assertTrue(receipt["mechanical_fixture_only"])
        for name in ("acceptance_authority", "private_gradient_authorized",
                     "historical_authority_granted", "training_authorized"):
            self.assertIs(receipt[name], False)
        self.assertIsNone(receipt["behavior_qualification"])
        self.assertNotIn("public.raw", json.dumps(receipt))
        self.assertNotIn("Fictional", json.dumps(receipt))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [path])
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_only_selected_einf_and_same_root_metadata_are_opened(self):
        path, digest = self.archive()
        opened = []
        original = zipfile.ZipFile.open

        def tracked(archive, name, *args, **kwargs):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original(archive, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", tracked):
            r.derive_raw_inference_registry(path, expected_archive_sha256=digest)
        self.assertEqual(opened, ["public-package/SHA256SUMS.txt",
                                  "public-package/generation_manifest.json",
                                  "public-package/einf_proposals.jsonl"])

    def test_wrong_or_missing_pin_refuses_before_zip_open(self):
        path, _ = self.archive()
        with patch.object(r.zipfile, "ZipFile", side_effect=AssertionError("must not open ZIP")):
            with self.assertRaisesRegex(r.RawInferenceLineageError, "source_archive_sha256_mismatch"):
                r.derive_raw_inference_registry(path, expected_archive_sha256="0" * 64)
        for value in (None, True, 123, "", "0" * 63, "g" * 64):
            with self.subTest(value=value), self.assertRaisesRegex(
                    r.RawInferenceLineageError, "invalid_expected_archive_sha256"):
                r.derive_raw_inference_registry(path, expected_archive_sha256=value)

    def test_exact_ledgers_rooted_unrooted_and_binary_marker(self):
        for root in ("public-package", "", "unknown/real/prefix"):
            for style in ("relative", "full"):
                with self.subTest(root=root, style=style):
                    registry, receipt = self.derive(fixture_members(root=root, ledger_style=style))
                    self.assertEqual(len(registry["EINF"]), 2)
                    self.assertEqual(receipt["source_member_path"],
                                     (root + "/" if root else "") + "einf_proposals.jsonl")
        members = fixture_members()
        members["public-package/SHA256SUMS.txt"] = members["public-package/SHA256SUMS.txt"].replace(b"  ", b" *")
        self.assertEqual(len(self.derive(members)[0]["EINF"]), 2)

    def test_missing_ambiguous_and_root_mismatch_logical_members(self):
        for logical in r._LOGICAL_NAMES:
            with self.subTest(logical=logical, variant="missing"):
                members = fixture_members()
                del members["public-package/" + logical]
                self.refused(members, "missing_or_ambiguous_logical_member")
            with self.subTest(logical=logical, variant="ambiguous"):
                self.refused(extra_entries=[("other/" + logical, b"unopened")],
                             reason="missing_or_ambiguous_logical_member")
        members = fixture_members()
        members["other/generation_manifest.json"] = members.pop("public-package/generation_manifest.json")
        self.refused(members, "logical_member_root_mismatch")

    def test_checksum_mismatch_missing_duplicate_or_fabricated_members(self):
        members = fixture_members()
        members["public-package/einf_proposals.jsonl"] += b" "
        self.refused(members, "proposal_member_checksum_mismatch")
        members = fixture_members()
        members["public-package/generation_manifest.json"] += b" "
        self.refused(members, "manifest_checksum_mismatch")
        ledger_name = "public-package/SHA256SUMS.txt"
        for logical in ("einf_proposals.jsonl", "generation_manifest.json"):
            members = fixture_members()
            members[ledger_name] = b"\n".join(line for line in members[ledger_name].split(b"\n")
                                            if not line.endswith(logical.encode()))
            self.refused(members, "required_member_missing_from_checksum_ledger")
        members = fixture_members()
        members[ledger_name] += members[ledger_name].splitlines()[0] + b"\n"
        self.refused(members, "duplicate_checksum_member")
        for payload in (b"", b"not a checksum\n", b"0" * 64 + b"  ../einf_proposals.jsonl\n",
                        b"0" * 64 + b"  absent.jsonl\n"):
            with self.subTest(payload=payload):
                members = fixture_members()
                members[ledger_name] = payload
                self.refused(members)

    def test_ledger_namespace_and_alias_conflicts(self):
        members = fixture_members()
        members["outside.txt"] = b"public outside"
        members["public-package/SHA256SUMS.txt"] += sha256(b"public outside").hexdigest().encode() + b"  outside.txt\n"
        self.refused(members, "checksum_member_namespace_conflict")
        members = fixture_members()
        members["einf_proposals.jsonl"] = b"outside logical alias"
        self.refused(members, "missing_or_ambiguous_logical_member")
        members = fixture_members()
        members["public-package/public-package/asyn_proposals.jsonl"] = b"ambiguous nested member"
        members["public-package/SHA256SUMS.txt"] = members["public-package/SHA256SUMS.txt"].replace(
            b"  asyn_proposals.jsonl", b"  public-package/asyn_proposals.jsonl")
        self.refused(members, "checksum_member_missing_or_ambiguous")

    def test_duplicate_case_implicit_directory_aliases_and_unsafe_paths(self):
        for name in ("public-package/einf_proposals.jsonl", "PUBLIC-PACKAGE/extra.txt",
                     "public-package/GENERATION_MANIFEST.json", "../escape", "/absolute",
                     "C:/drive", "dot/./member", "repeat//member", "trailing. ",
                     "public-package/CON", "control\x01member", "decomposed/e\u0301.txt"):
            with self.subTest(name=name):
                self.refused(extra_entries=[(name, b"unopened public sentinel")])
        self.refused(extra_entries=[("public-package/einf_proposals.jsonl/child", b"unopened")],
                     reason="archive_file_directory_alias")
        # Windows ZipInfo normalizes backslashes on write; create the raw unsafe
        # central/local name bytes deliberately to exercise the reader instead.
        path, _ = self.archive(extra_entries=[("backXslash", b"unopened")])
        payload = path.read_bytes().replace(b"backXslash", b"back\\slash")
        path.write_bytes(payload)
        with self.assertRaisesRegex(r.RawInferenceLineageError, "unsafe_archive_member_path"):
            r.derive_raw_inference_registry(path, expected_archive_sha256=sha256(payload).hexdigest())

    def test_links_special_type_directory_payload_and_encryption_refuse(self):
        for kind in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK):
            with self.subTest(kind=kind):
                info = zipfile.ZipInfo("public-package/special")
                info.create_system = 3
                info.external_attr = (kind | 0o600) << 16
                self.refused(extra_entries=[(info, b"unopened")], reason="nonregular_archive_member")
        self.refused(extra_entries=[("public-package/directory/", b"payload")],
                     reason="directory_archive_member_has_payload")
        for flag in (0x1, 0x40, 0x2000):
            with self.subTest(encryption_flag=flag):
                path, _ = self.archive()
                data = bytearray(path.read_bytes())
                offset = data.index(b"PK\x01\x02")
                value = int.from_bytes(data[offset + 8:offset + 10], "little") | flag
                data[offset + 8:offset + 10] = value.to_bytes(2, "little")
                path.write_bytes(data)
                digest = sha256(data).hexdigest()
                with self.assertRaisesRegex(r.RawInferenceLineageError, "encrypted_archive_member"):
                    r.derive_raw_inference_registry(path, expected_archive_sha256=digest)

    def test_duplicate_keys_invalid_json_and_nonfinite_reject_without_payloads(self):
        rows = (b'{"proposal_id":"public.raw.a","proposal_id":"public.raw.b","provenance_class":"E-INF"}\n',
                b'{"proposal_id":"public.raw.a","provenance_class":"E-INF","nested":{"x":1,"x":2}}\n',
                b'{"proposal_id":"public.raw.a","provenance_class":"E-INF","number":NaN}\n',
                b'{"proposal_id":"public.raw.a","provenance_class":"E-INF","number":1e9999}\n',
                b'not JSON public.raw.secret\n', b'\xff\n', b'[]\n', b'null\n')
        for payload in rows:
            with self.subTest(payload=payload):
                self.refused(fixture_members(row_bytes=payload))
        for payload in (b'{"x":1,"x":2}', b'{"x":Infinity}', b'{"x":1e999}', b'[]', b'null', b'\xff'):
            with self.subTest(manifest=payload):
                members = fixture_members()
                members["public-package/generation_manifest.json"] = payload
                refresh_ledger(members)
                self.refused(members)

    def test_original_id_and_einf_class_strict_not_inferred_from_references(self):
        for value in (None, True, 1, [], {}, "", "two ids", "control\x01", "x" * 513,
                      "\ud800", "control\x80"):
            with self.subTest(id=value):
                rows = fixture_rows()
                rows[0]["proposal_id"] = value
                self.refused(fixture_members(rows=rows), "invalid_original_proposal_id")
        for value in (None, True, 1, [], "EINF", "A-SYN", "UNKNOWN", "E0", "E-INF "):
            with self.subTest(source_class=value):
                rows = fixture_rows()
                rows[0]["provenance_class"] = value
                self.refused(fixture_members(rows=rows), "invalid_proposal_namespace")
        self.refused(fixture_members(rows=[fixture_rows()[0], fixture_rows()[0]]),
                     "duplicate_original_proposal_id")
        self.refused(fixture_members(rows=[{"supporting_raw_EINF_ids": ["invented.from.refs"],
                                           "provenance_class": "E-INF"}]), "invalid_original_proposal_id")
        registry, _ = self.derive()
        self.assertNotIn("dangling.must.not.be.harvested", registry["EINF"])
        self.assertNotIn("invented.from.refs", registry["EINF"])

    def test_empty_blank_rows_and_operating_bounds_refuse_without_truncation(self):
        for payload in (b"", b"\n", b"\n" + canonical(fixture_rows()[0]) + b"\n"):
            with self.subTest(payload=payload):
                self.refused(fixture_members(row_bytes=payload))
        for name, value, reason in (("MAX_ARCHIVE_BYTES", 1, "source_archive_resource_bound"),
                                    ("MAX_ARCHIVE_ENTRIES", 1, "archive_inventory_bound"),
                                    ("MAX_METADATA_BYTES", 1, "selected_member_resource_bound"),
                                    ("MAX_PROPOSAL_BYTES", 1, "selected_member_resource_bound"),
                                    ("MAX_JSONL_LINE_BYTES", 1, "proposal_line_resource_bound"),
                                    ("MAX_PROPOSAL_ROWS", 1, "proposal_row_count_bound")):
            with self.subTest(bound=name), patch.object(r, name, value):
                self.refused(reason=reason)
        registry, receipt = self.derive(fixture_members(rows=[fixture_rows()[0]]))
        self.assertEqual(receipt["raw_inference_count"], 1)
        self.assertEqual(registry["EINF"], ["public.raw.b"])

    def test_source_hash_and_file_identity_rechecked_after_member_reads(self):
        path, digest = self.archive()
        original = r._hash_stream
        calls = []

        def changed(stream):
            calls.append(True)
            if len(calls) == 2:
                return "0" * 64
            return original(stream)

        with patch.object(r, "_hash_stream", changed), self.assertRaisesRegex(
                r.RawInferenceLineageError, "source_archive_changed"):
            r.derive_raw_inference_registry(path, expected_archive_sha256=digest)
        self.assertEqual(len(calls), 2)
        replacement, _ = self.archive()
        replacement.write_bytes(path.read_bytes())
        actual_path = r._archive_path
        path_calls = []

        def changed_path(value):
            path_calls.append(True)
            return actual_path(value) if len(path_calls) == 1 else actual_path(replacement)

        with patch.object(r, "_archive_path", changed_path), self.assertRaisesRegex(
                r.RawInferenceLineageError, "source_archive_changed"):
            r.derive_raw_inference_registry(path, expected_archive_sha256=digest)

    def test_linked_source_paths_and_nonarchives_fail_sanitized(self):
        path, digest = self.archive()
        link = self.root / "linked.zip"
        try:
            link.symlink_to(path)
        except (OSError, NotImplementedError):
            pass  # Windows may deny public-fixture symlink creation.
        else:
            with self.assertRaisesRegex(r.RawInferenceLineageError, "linked_archive_path"):
                r.derive_raw_inference_registry(link, expected_archive_sha256=digest)
        with self.assertRaisesRegex(r.RawInferenceLineageError, "nonregular_source_archive"):
            r.derive_raw_inference_registry(self.root, expected_archive_sha256=digest)
        bad = self.root / "not-a-zip"
        bad.write_bytes(b"PUBLIC synthetic invalid ZIP")
        with self.assertRaisesRegex(r.RawInferenceLineageError, "raw_lineage_source_unreadable"):
            r.derive_raw_inference_registry(bad, expected_archive_sha256=sha256(bad.read_bytes()).hexdigest())

    def test_member_crc_is_checked_and_code_changes_refuse(self):
        path, _ = self.archive()
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("public-package/einf_proposals.jsonl")
            crc = info.CRC
        data = bytearray(path.read_bytes())
        # Change the selected central-directory CRC while keeping a valid externally pinned ZIP.
        index = 0
        while True:
            index = data.index(b"PK\x01\x02", index)
            size = int.from_bytes(data[index + 28:index + 30], "little")
            if bytes(data[index + 46:index + 46 + size]) == info.filename.encode():
                data[index + 16:index + 20] = (crc ^ 1).to_bytes(4, "little")
                break
            index += 4
        path.write_bytes(data)
        with self.assertRaisesRegex(r.RawInferenceLineageError, "raw_lineage_source_unreadable"):
            r.derive_raw_inference_registry(path, expected_archive_sha256=sha256(data).hexdigest())
        path, digest = self.archive()
        original = Path.read_bytes
        count = []

        def changed_code(value):
            payload = original(value)
            if value == Path(r.__file__):
                count.append(True)
                if len(count) == 2:
                    return payload + b"public-code-change"
            return payload

        with patch.object(Path, "read_bytes", changed_code), self.assertRaisesRegex(
                r.RawInferenceLineageError, "lineage_reader_code_changed"):
            r.derive_raw_inference_registry(path, expected_archive_sha256=digest)

    def test_malformed_supported_compression_errors_are_sanitized(self):
        for compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
            with self.subTest(compression=compression):
                path, _ = self.archive(compression=compression)
                with zipfile.ZipFile(path) as archive:
                    entry = archive.getinfo("public-package/einf_proposals.jsonl")
                data = bytearray(path.read_bytes())
                offset = entry.header_offset
                name_size = int.from_bytes(data[offset + 26:offset + 28], "little")
                extra_size = int.from_bytes(data[offset + 28:offset + 30], "little")
                start = offset + 30 + name_size + extra_size
                data[start:start + entry.compress_size] = b"\xff" * entry.compress_size
                path.write_bytes(data)
                with self.assertRaisesRegex(r.RawInferenceLineageError, "raw_lineage_source_unreadable"):
                    r.derive_raw_inference_registry(path, expected_archive_sha256=sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
