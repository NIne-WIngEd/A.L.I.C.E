"""Byte-accurate AMI window provenance and as-of bounds."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest
import zipfile

from scripts.mfm.inventory_ami_meeting_sources import (
    ARCHIVE_SHA256, attributed_turns, build, original_words,
)


ARCHIVE = (Path(__file__).resolve().parents[3] /
           "ami-source-custody/ami_public_manual_1.6.2.zip")


class OriginalWordContractTests(unittest.TestCase):
    def test_original_span_and_as_of_boundary(self):
        raw = (b'<?xml version="1.0"?>\n'
               b'<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
               b'<w nite:id="M.A.words0" starttime="1.00" endtime="1.25">A&amp;B</w>\n'
               b'<w nite:id="M.A.words1" starttime="5.00" endtime="5.25">later</w>\n'
               b'</nite:root>')
        result = original_words(raw, member="words/M.A.words.xml")
        self.assertEqual(len(result), 2)
        for word in result:
            span = raw[word["byte_start"]:word["byte_end"]]
            self.assertEqual(sha256(span).hexdigest(), word["raw_span_sha256"])
        self.assertEqual(result[0]["text"], "A&B")
        self.assertEqual([word["text"] for word in result
                          if word["end_seconds"] <= 2.0], ["A&B"])
        altered = raw.replace(b'later', b'other')
        self.assertNotEqual(sha256(altered[result[1]["byte_start"]:
                                           result[1]["byte_end"]]).hexdigest(),
                            result[1]["raw_span_sha256"])

    def test_interleaved_speakers_remain_separate_attributed_turns(self):
        words = [
            {"speaker": "A", "participant_id": "p1", "role": "PM",
             "start_seconds": 1.0, "end_seconds": 1.2,
             "text": "Thank", "word_id": "a1"},
            {"speaker": "B", "participant_id": "p2", "role": "ME",
             "start_seconds": 1.1, "end_seconds": 1.3,
             "text": "Thank", "word_id": "b1"},
            {"speaker": "A", "participant_id": "p1", "role": "PM",
             "start_seconds": 1.2, "end_seconds": 1.2,
             "text": ".", "word_id": "a2"},
            {"speaker": "B", "participant_id": "p2", "role": "ME",
             "start_seconds": 1.3, "end_seconds": 1.5,
             "text": "you", "word_id": "b2"},
        ]
        turns = attributed_turns(words)
        self.assertEqual([(x["speaker"], x["text"], x["word_ids"]) for x in turns],
                         [("A", "Thank.", ["a1", "a2"]),
                          ("B", "Thank you", ["b1", "b2"])])

    @unittest.skipUnless(ARCHIVE.is_file(), "pinned official archive not staged")
    def test_official_window_bytes_and_replay(self):
        self.assertEqual(sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        inventory, windows = build(ARCHIVE)
        self.assertEqual((inventory.count(b"\n"), windows.count(b"\n")), (1, 72))
        self.assertEqual((inventory, windows), build(ARCHIVE))
        cached = {}
        with zipfile.ZipFile(ARCHIVE) as z:
            for line in windows.splitlines():
                row = json.loads(line)
                self.assertFalse(row["training_admitted"])
                self.assertIsNone(row["formation_targets"])
                self.assertEqual(sha256(row["derived_display"].encode()).hexdigest(),
                                 row["derived_display_sha256"])
                self.assertEqual(row["derived_turns"], attributed_turns(row["words"]))
                begin = row["window"]["start_seconds"]
                cutoff = row["as_of"]["window_end_seconds"]
                for word in row["words"]:
                    self.assertGreaterEqual(word["start_seconds"], begin)
                    self.assertLessEqual(word["end_seconds"], cutoff)
                    member = word["member"]
                    raw = cached.setdefault(member, z.read(member))
                    self.assertEqual(sha256(raw).hexdigest(), word["member_sha256"])
                    self.assertEqual(sha256(raw[word["byte_start"]:
                                                word["byte_end"]]).hexdigest(),
                                     word["raw_span_sha256"])


if __name__ == "__main__":
    unittest.main()
