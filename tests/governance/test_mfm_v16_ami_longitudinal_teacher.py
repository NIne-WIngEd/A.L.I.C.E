"""The AMI teacher view rejects reordered or future transcript turns."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.mfm.prepare_v16_ami_longitudinal_teacher_case import _opened, prepare


def _row(stage: str, end: float = 10.0) -> bytes:
    row = {
        "schema": "mfm-ami-original-word-window-v1",
        "status": "source_candidate_unreviewed", "training_admitted": False,
        "archive_sha256": "a" * 64, "session_id": "AB1234",
        "meeting_id": "AB1234" + stage, "stage": stage,
        "as_of": {"window_end_seconds": 11.0},
        "words": [{"word_id": "w0"}],
        "derived_turns": [{"participant_id": "speaker1", "start_seconds": 9.0,
                           "end_seconds": end, "word_ids": ["w0"], "text": "Observed text."}],
    }
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()


class AmiLongitudinalTeacherTest(unittest.TestCase):
    def test_stage_and_as_of_are_model_visible_and_replayed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = []
            for stage in "bc":
                raw = _row(stage)
                path = root / (stage + ".json")
                path.write_bytes(raw)
                originals.append((path, sha256(raw).hexdigest()))
            private = root / "private"
            prepared = prepare(windows=tuple(originals), selection=("ab1234b:0", "ab1234c:0"),
                               case_id="test-cross-stage", authorization_id="unit-test", output_dir=private)
            envelope = json.loads((private / "envelope.json").read_bytes())
            opened = _opened(envelope, private / "envelope.json")
            self.assertEqual(len(opened), 2)
            self.assertIn(b"meeting: AB1234b\nstage: b\nas_of: AB1234c@11.0s", opened[0][1])
            self.assertIn(b"meeting: AB1234c\nstage: c", opened[1][1])
            self.assertFalse(prepared["training_admitted"])

            source = private / envelope["sources"][0]["path"]
            source.write_bytes(source.read_bytes().replace(b"stage: b", b"stage: c"))
            with self.assertRaisesRegex(ValueError, "model-visible"):
                _opened(envelope, private / "envelope.json")

    def test_rejects_future_and_reordered_turns(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = []
            for stage in "bc":
                raw = _row(stage, end=12.0 if stage == "c" else 10.0)
                path = root / (stage + ".json")
                path.write_bytes(raw)
                originals.append((path, sha256(raw).hexdigest()))
            with self.assertRaisesRegex(ValueError, "beyond original window"):
                prepare(windows=tuple(originals), selection=("ab1234b:0", "ab1234c:0"),
                        case_id="test-future", authorization_id="unit-test",
                        output_dir=root / "future")
            with self.assertRaisesRegex(ValueError, "ordered team history"):
                prepare(windows=tuple(reversed(originals)), selection=("ab1234c:0", "ab1234b:0"),
                        case_id="test-reversed", authorization_id="unit-test",
                        output_dir=root / "reversed")


if __name__ == "__main__":
    unittest.main()
