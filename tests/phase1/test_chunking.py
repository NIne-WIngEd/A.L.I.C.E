from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.chunking import chunk_text, load_chunking_policy, stable_chunk_id

class ChunkingTests(unittest.TestCase):
    def test_chunking_is_deterministic_and_overlapping(self):
        policy = load_chunking_policy(ROOT / "policies" / "chunking_policy.json")
        paragraph = "A deterministic paragraph about research and education. It has repeatable boundaries. "
        text = "\n\n".join(f"Section {i}. " + paragraph * 15 for i in range(12))
        normalized_a, chunks_a = chunk_text(text, policy)
        normalized_b, chunks_b = chunk_text(text, policy)
        self.assertEqual((normalized_a, chunks_a), (normalized_b, chunks_b))
        self.assertGreater(len(chunks_a), 1)
        self.assertEqual(chunks_a[0].start, 0)
        self.assertEqual(chunks_a[-1].end, len(normalized_a))
        for previous, current in zip(chunks_a, chunks_a[1:]):
            self.assertLess(current.start, previous.end)
            self.assertGreater(current.end, previous.end)
            self.assertLessEqual(len(current.text), policy.max_chars)
        ids = [stable_chunk_id("a" * 64, policy.digest, span) for span in chunks_a]
        self.assertEqual(len(ids), len(set(ids)))

    def test_normalization_is_stable(self):
        policy = load_chunking_policy(ROOT / "policies" / "chunking_policy.json")
        first, spans_first = chunk_text("Title  \r\n\r\n\r\nBody\x00 text.\r\n", policy)
        second, spans_second = chunk_text("Title\n\nBody  text.\n", policy)
        self.assertEqual(first, second)
        self.assertEqual(spans_first, spans_second)

if __name__ == "__main__":
    unittest.main()
