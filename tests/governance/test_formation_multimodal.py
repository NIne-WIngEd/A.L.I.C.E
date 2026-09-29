from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
import wave
from unittest.mock import patch

import numpy as np
from PIL import Image

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_gold import load_frozen_formation_gold
from cognitive_kernel.formation_learning import (
    FormationLearningExample, output_record,
)
from cognitive_kernel.formation_multimodal import (
    MultimodalFormationCandidate, formation_media_messages,
    supervised_multimodal_batch,
)
from scripts.mfm.train_multimodal_formation import load_multimodal_candidate


MANIFEST = (Path(__file__).resolve().parents[1] /
            "fixtures/mfm/fictional_formation_gold_v1.manifest.json")


def _example(modality: str, raw: bytes) -> FormationLearningExample:
    fixture = load_frozen_formation_gold(MANIFEST)[0]
    refs = fixture.gold.context.evidence
    updated_ref = replace(refs[0], modality=modality,
                          content_digest=sha256(raw).hexdigest())
    context = replace(fixture.gold.context, evidence=(updated_ref, *refs[1:]))
    original = dict(fixture.texts)
    opened = tuple((ref.ref_id,
                    raw if ref.ref_id == updated_ref.ref_id else original[ref.ref_id].encode())
                   for ref in context.evidence)
    # There is no semantic gold for the synthetic media bytes in this test.
    from cognitive_kernel.formation_contracts import FormationDisposition, MemoryProposalBundle
    bundle = MemoryProposalBundle(
        scope=context.scope, authority_namespace_id=context.authority_namespace_id,
        bundle_id="formation-multimodal-fixture",
        experience_refs=context.experience_refs,
        context_digest=context.content_digest(),
        model_artifact_digest="0" * 64, inference_run_id="multimodal-fixture",
        proposals=(), dispositions=(FormationDisposition(
            scope_ref="media_observation", action="abstain",
            evidence_refs=(updated_ref.ref_id,)),),
    )
    return FormationLearningExample("multimodal-fixture", context, opened, bundle)


def _png() -> bytes:
    image = Image.new("RGB", (16, 16), (12, 143, 77))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _wav(seconds: int) -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 16000 * seconds)
    return stream.getvalue()


class FakeTensor:
    def __init__(self, values):
        self.array = np.asarray(values)

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def clone(self):
        return FakeTensor(self.array.copy())

    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, *, omit_image=False):
        self.omit_image = omit_image

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        media = [part for part in messages[0]["content"] if part["type"] == "image"]
        assert media and Path(media[0]["path"]).read_bytes() == _png()
        assert kwargs["do_sample_frames"] is False
        result = {"input_ids": FakeTensor([[1, 2, 3] if add_generation_prompt
                                            else [1, 2, 3, 4, 5]])}
        if not self.omit_image:
            result["pixel_values"] = FakeTensor([[0.1, 0.2]])
        return result

    def decode(self, tokens, *, skip_special_tokens):
        return self.answer

    @property
    def tokenizer(self):
        return type("Tokenizer", (), {"eos_token_id": 9})()


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return FakeTensor([[1, 2, 3, 4]])


class MultimodalFormationTests(unittest.TestCase):
    def test_raw_image_reaches_processor_and_only_answer_is_supervised(self):
        example = _example("image", _png())
        with formation_media_messages(example) as messages:
            parts = messages[0]["content"]
            image = next(part for part in parts if part["type"] == "image")
            self.assertEqual(Path(image["path"]).read_bytes(), _png())
            self.assertFalse(any("iVBOR" in p["text"] for p in parts if p["type"] == "text"))
        self.assertFalse(Path(image["path"]).exists())
        batch = supervised_multimodal_batch(FakeProcessor(), example, 10)
        self.assertEqual(batch["labels"].array.tolist(), [[-100, -100, -100, 4, 5]])
        with self.assertRaisesRegex(CognitiveKernelContractError, "do not truncate"):
            supervised_multimodal_batch(FakeProcessor(), example, 4)
        with self.assertRaisesRegex(CognitiveKernelContractError, "omitted image"):
            supervised_multimodal_batch(FakeProcessor(omit_image=True), example, 10)

    def test_multimodal_inference_preserves_exact_scope_and_grounding(self):
        example = _example("image", _png())
        processor = FakeProcessor()
        processor.answer = json.dumps(output_record(example.target))
        candidate = MultimodalFormationCandidate(
            FakeModel(), processor, "a" * 64, "media-inference", 100, 30)
        fake_torch = type("Torch", (), {"inference_mode": staticmethod(nullcontext)})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = candidate.infer(context=example.context,
                                     opened_sources=example.opened_sources)
        self.assertEqual(result.dispositions[0].action, "abstain")
        self.assertEqual(result.context_digest, example.context.content_digest())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(CognitiveKernelContractError, "budget"):
                MultimodalFormationCandidate(
                    FakeModel(), processor, "a" * 64, "media-inference", 2, 30).infer(
                        context=example.context, opened_sources=example.opened_sources)

    def test_changed_artifact_refuses_load_before_transformers_import(self):
        import tempfile
        from scripts.mfm.train_formation_model import write_artifact_receipt

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            write_artifact_receipt(model, {"test": True})
            (model / "config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(CognitiveKernelContractError, "differs"):
                load_multimodal_candidate(model, inference_run_id="read",
                                          max_input_tokens=100, max_new_tokens=30)

    def test_audio_over_one_window_is_segmented_without_base64_prompt(self):
        example = _example("audio", _wav(31))
        with formation_media_messages(example) as messages:
            parts = messages[0]["content"]
            blocks = [p for p in parts if p["type"] == "audio"]
            self.assertEqual(len(blocks), 2)
            for item in blocks:
                self.assertTrue(Path(item["path"]).read_bytes().startswith(b"RIFF"))

    def test_video_and_its_soundtrack_reach_distinct_media_inputs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                "-i", "color=c=blue:s=32x32:r=2:d=2", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=2", "-c:v", "mpeg4",
                "-c:a", "aac", "-shortest", str(output),
            ], check=True)
            example = _example("video", output.read_bytes())
            with formation_media_messages(example) as messages:
                parts = messages[0]["content"]
                self.assertEqual(sum(p["type"] == "video" for p in parts), 1)
                self.assertEqual(sum(p["type"] == "audio" for p in parts), 1)
                self.assertTrue(all(Path(p["path"]).stat().st_size > 0 for p in parts
                                    if p["type"] in {"video", "audio"}))

    def test_unregistered_sensor_cannot_be_represented_as_text(self):
        with self.assertRaisesRegex(CognitiveKernelContractError, "registered modality decoder"):
            with formation_media_messages(_example("sensor", b"\0\1\2")):
                pass


if __name__ == "__main__":
    unittest.main()
