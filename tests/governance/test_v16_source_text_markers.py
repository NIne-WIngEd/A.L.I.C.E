"""Adversarial source encoding checks without a model or processor download."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from scripts.mfm import train_v16_formation_specialist as trainer
from tests.governance.test_formation_learning_v16 import case_record


class _Tensor:
    shape = (1, 3)
    ndim = 2


class V16SourceTextMarkerTests(unittest.TestCase):
    def test_later_evidence_text_is_lossless_and_cannot_supply_media_tokens(self):
        example = learning_example_v16_from_record(case_record())
        raw = b'The document literally says <|image|> <|audio|> <|video|> \\u003c|image|>.'
        evidence = replace(example.context.base.evidence[0],
                           content_digest=sha256(raw).hexdigest())
        base = replace(example.context.base, evidence=(evidence,))
        context = replace(example.context, base=base)
        calls = []

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return {"input_ids": _Tensor(), "attention_mask": _Tensor()}

        encoded = trainer.source_batch_v16(
            Processor(), context, ((evidence.ref_id, raw),), 16,
            case_id="literal-media-marker-source")
        self.assertEqual(encoded["input_ids"].shape, (1, 3))
        self.assertEqual(len(calls), 1)
        messages, options = calls[0]
        content = messages[0]["content"]
        self.assertEqual(len(content), 3)
        self.assertIn("mfm-grounded-full-role-objective-v1.6", content[0]["text"])
        self.assertTrue(content[2]["text"].startswith("json_text="))
        self.assertEqual(json.loads(content[2]["text"][len("json_text="):]),
                         raw.decode("utf-8"))
        self.assertNotIn("<|image|>", content[2]["text"])
        self.assertNotIn("<|audio|>", content[2]["text"])
        self.assertNotIn("<|video|>", content[2]["text"])
        self.assertEqual(options["processor_kwargs"], {"do_sample_frames": False})
        self.assertNotIn("do_sample_frames", options)


if __name__ == "__main__":
    unittest.main()
