"""Source anchoring and as-of checks for the official ICSI extraction route."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.mfm.acquire_v16_icsi_source_windows import acquire
from scripts.mfm.prepare_v16_icsi_source_excerpt import excerpt


def fixture(path: Path) -> bytes:
    observations = "".join(f'<observation name="Bmr{i:03d}"/>'
                           for i in range(1, 76))
    words = (b'<nite:root xmlns:nite="http://nite.sourceforge.net/">'
             b'<w nite:id="w1" starttime="1" endtime="1.1" c="W">Here</w>'
             b'<w nite:id="w2" starttime="1.2" endtime="2" c="W">now</w>'
             b'<w nite:id="w3" starttime="599.9" endtime="600" c="W">boundary</w>'
             b'<w nite:id="w4" starttime="600" endtime="600.1" c="W">later</w>'
             b'<w nite:id="w5" starttime="" endtime="" c="W">untimed</w>'
             b'</nite:root>')
    with ZipFile(path, "w") as archive:
        archive.writestr("ICSI/ICSI-metadata.xml",
                         f'<corpus><observations>{observations}</observations></corpus>')
        archive.writestr("ICSI/LICENCE.txt", "Creative Commons Attribution 4.0")
        archive.writestr("ICSI/Words/Bmr001.A.words.xml", words)
        archive.writestr("ICSI/DialogueActs/Bmr001.A.dialogue-acts.xml",
                         '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
                         '<dialogueact participant="person-1"/></nite:root>')
    return words


class ICSISourceTests(unittest.TestCase):
    def test_exact_byte_anchor_and_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            words = fixture(source)
            receipt = acquire(source, root / "windows", expected_sha256=sha256(
                source.read_bytes()).hexdigest())
            rows = [json.loads(line) for line in
                    (root / "windows/source_windows.jsonl").read_bytes().splitlines()]
            self.assertEqual([(r["as_of_end_seconds"], len(r["observations"]))
                              for r in rows], [(600, 3), (1200, 1)])
            first = rows[0]["observations"][0]
            anchor = first["source"]
            original = words[anchor["byte_start"]:anchor["byte_end"]]
            self.assertEqual(sha256(original).hexdigest(), anchor["element_sha256"])
            self.assertIn(b"Here", original)
            self.assertEqual(first["participant_id"], "person-1")
            self.assertEqual(receipt["skipped_word_count_by_reason"], {"untimed": 1})
            self.assertEqual(receipt["window_count"], 2)
            packet = excerpt(root / "windows/source_windows.jsonl",
                             root / "windows/acquisition_receipt.json",
                             meeting_id="Bmr001", as_of_end_seconds=600,
                             clip_start_seconds=0, clip_end_seconds=120)
            self.assertEqual([word["word_id"] for span in packet["spans"]
                              for word in span["words"]], ["w1", "w2"])

    def test_corrupt_zip_and_future_clip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            fixture(source)
            with self.assertRaisesRegex(ValueError, "pinned SHA"):
                acquire(source, root / "invalid")
            acquire(source, root / "windows", expected_sha256=sha256(
                source.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "as-of boundary"):
                excerpt(root / "windows/source_windows.jsonl",
                        root / "windows/acquisition_receipt.json",
                        meeting_id="Bmr001", as_of_end_seconds=600,
                        clip_start_seconds=590, clip_end_seconds=601)


if __name__ == "__main__":
    unittest.main()
