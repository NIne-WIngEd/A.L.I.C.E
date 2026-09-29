"""Exact-source MFM prompt construction for a real Gemma 4 Unified processor.

The original source bytes remain the custody authority. Media are decoded from
those bytes and passed as media to ``AutoProcessor.apply_chat_template``. A
caption, digest, or base64 string is never substituted for sensory input.

Gemma 4 12B accepts at most 30 seconds per audio block and 60 video frames at
1 fps per video block. Longer files become consecutive, time-labelled blocks;
their total still has to fit the configured model context. ffmpeg/ffprobe are
needed for deterministic media segmentation, not for text sources.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator

from .canonical import CognitiveKernelContractError, canonical_json_bytes
from .formation_contracts import (
    FormationContextPacket, MemoryProposalBundle, validate_formation_grounding,
)
from .formation_learning import (
    FormationLearningExample, SYSTEM_INSTRUCTION, bundle_from_output,
    output_record,
)


GEMMA_4_12B_MODEL = "google/gemma-4-12B-it"
GEMMA_4_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MULTIMODAL_OBJECTIVE = "mfm-grounded-multimodal-disposition-objective-v1"
_TEXT = frozenset({"text", "code", "structured"})
_MEDIA = frozenset({"image", "audio", "video"})


def _suffix(raw: bytes, modality: str) -> str:
    if modality == "image":
        for header, suffix in ((b"\x89PNG\r\n\x1a\n", ".png"),
                               (b"\xff\xd8\xff", ".jpg"),
                               (b"GIF87a", ".gif"), (b"GIF89a", ".gif"),
                               (b"BM", ".bmp"), (b"II*\x00", ".tiff"),
                               (b"MM\x00*", ".tiff")):
            if raw.startswith(header):
                return suffix
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return ".webp"
    if modality == "audio":
        if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
            return ".wav"
        if raw.startswith(b"fLaC"):
            return ".flac"
        if raw.startswith(b"ID3") or raw[:1] == b"\xff":
            return ".mp3"
        if raw.startswith(b"OggS"):
            return ".ogg"
        if raw[4:8] == b"ftyp":
            return ".m4a"
        if raw.startswith(b"\x1a\x45\xdf\xa3"):
            return ".webm"
        if raw.startswith(b"RIFF") and raw[8:12] == b"AVI ":
            return ".avi"
    if modality == "video":
        if raw[4:8] == b"ftyp":
            return ".mp4"
        if raw.startswith(b"\x1a\x45\xdf\xa3"):
            return ".webm"
        if raw.startswith(b"RIFF") and raw[8:12] == b"AVI ":
            return ".avi"
    raise CognitiveKernelContractError(f"unrecognized {modality} source format")


def _run(command: list[str]) -> bytes:
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CognitiveKernelContractError(
            f"media decoder failed: {command[0]}; install ffmpeg/ffprobe and inspect the source"
        ) from exc
    return result.stdout


def _probe(path: Path, kind: str) -> tuple[float, bool]:
    info = json.loads(_run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(path),
    ]))
    streams = {entry.get("codec_type") for entry in info.get("streams", ())}
    if kind not in streams:
        raise CognitiveKernelContractError(f"source contains no {kind} stream")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError) as exc:
        raise CognitiveKernelContractError("media source lacks a finite duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise CognitiveKernelContractError("media source lacks a positive duration")
    return duration, "audio" in streams


def _segments(duration: float, seconds: int) -> Iterator[tuple[int, float, float]]:
    for i in range(math.ceil(duration / seconds)):
        start = i * seconds
        yield i, float(start), min(float(seconds), duration - start)


def _media_blocks(root: Path, index: int, modality: str, raw: bytes) -> list[dict]:
    source = root / f"source-{index}{_suffix(raw, modality)}"
    source.write_bytes(raw)
    if modality == "image":
        return [{"type": "image", "path": str(source)}]
    if modality == "audio":
        duration, _ = _probe(source, "audio")
        blocks = []
        for n, start, length in _segments(duration, 30):
            clip = root / f"audio-{index}-{n}.wav"
            _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
                  "-i", str(source), "-t", str(length), "-vn", "-ac", "1", "-ar",
                  "16000", "-c:a", "pcm_s16le", str(clip)])
            if not clip.exists() or not clip.stat().st_size:
                raise CognitiveKernelContractError("empty decoded audio segment")
            blocks.extend(({"type": "text", "text": f"Audio interval {start:g}-{start+length:g}s"},
                           {"type": "audio", "path": str(clip)}))
        return blocks
    duration, has_audio = _probe(source, "video")
    blocks = []
    for n, start, length in _segments(duration, 60):
        clip = root / f"video-{index}-{n}.mp4"
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
              "-i", str(source), "-t", str(length), "-vf", "fps=1", "-an",
              "-c:v", "mpeg4", "-q:v", "3", str(clip)])
        if not clip.exists() or not clip.stat().st_size:
            raise CognitiveKernelContractError("empty decoded video segment")
        blocks.extend(({"type": "text", "text": f"Video interval {start:g}-{start+length:g}s, 1 fps"},
                       {"type": "video", "path": str(clip)}))
    if has_audio:
        # Preserve the sound track as actual waveform blocks too. The video
        # processor itself sees only sampled visual frames.
        blocks.extend(_media_blocks(root, index + 1_000_000, "audio", raw))
    return blocks


@contextmanager
def formation_context_media_messages(
    context: FormationContextPacket, opened_sources: tuple[tuple[str, bytes], ...],
) -> Iterator[list[dict]]:
    """Build a chat whose media paths exist only for the processor call."""
    context.validate()
    refs = {ref.ref_id: ref for ref in context.evidence}
    if len(refs) != len(opened_sources) or {key for key, _ in opened_sources} != set(refs):
        raise CognitiveKernelContractError("multimodal prompt needs every exact source")
    with tempfile.TemporaryDirectory(prefix="mfm-source-") as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        content: list[dict] = [{"type": "text", "text":
            SYSTEM_INSTRUCTION + "\n" + canonical_json_bytes({
                "objective": MULTIMODAL_OBJECTIVE,
                "context": context.metadata_record(),
            }).decode("utf-8")}]
        for index, (ref_id, raw) in enumerate(opened_sources):
            ref = refs[ref_id]
            if sha256(raw).hexdigest() != ref.content_digest:
                raise CognitiveKernelContractError("multimodal source digest mismatch")
            content.append({"type": "text", "text":
                canonical_json_bytes({"source_ref": ref_id, "modality": ref.modality}).decode("utf-8")})
            if ref.modality in _TEXT:
                try:
                    string = raw.decode("utf-8", "strict")
                except UnicodeDecodeError as exc:
                    raise CognitiveKernelContractError("text source is not UTF-8") from exc
                content.append({"type": "text", "text": string})
            elif ref.modality in _MEDIA:
                content.extend(_media_blocks(root, index, ref.modality, raw))
            else:
                raise CognitiveKernelContractError(
                    f"{ref.modality} needs a registered modality decoder, not a text surrogate")
        yield [{"role": "user", "content": content}]


@contextmanager
def formation_media_messages(example: FormationLearningExample) -> Iterator[list[dict]]:
    example.validate()
    with formation_context_media_messages(example.context, example.opened_sources) as messages:
        yield messages


def _assert_modality_tensors(context: FormationContextPacket, payload) -> None:
    required = {"image": "pixel_values", "audio": "input_features",
                "video": "pixel_values_videos"}
    modalities = {ref.modality for ref in context.evidence}
    for modality, key in required.items():
        if modality in modalities and key not in payload:
            raise CognitiveKernelContractError(f"processor omitted {modality} payload")
        if modality in modalities and not hasattr(payload[key], "shape"):
            raise CognitiveKernelContractError(f"processor did not tensorize {modality} payload")


def supervised_multimodal_batch(processor, example: FormationLearningExample,
                                max_sequence_tokens: int):
    """Supervise the exact JSON response, with all source tokens masked."""
    if max_sequence_tokens < 1:
        raise CognitiveKernelContractError("invalid multimodal context budget")
    with formation_media_messages(example) as messages:
        answer = canonical_json_bytes(output_record(example.target)).decode("utf-8")
        # The staged video has already been sampled to 1 fps and split into
        # <=60-frame chunks. The processor must preserve all staged frames.
        options = {"tokenize": True, "return_dict": True, "return_tensors": "pt",
                   "do_sample_frames": False}
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, **options)
        full = processor.apply_chat_template(
            [*messages, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
            add_generation_prompt=False, **options)
        _assert_modality_tensors(example.context, full)
        prefix = prompt["input_ids"][0]
        tokens = full["input_ids"]
        if (tokens.shape[0] != 1 or tokens.shape[1] <= len(prefix)
                or tokens.shape[1] > max_sequence_tokens
                or not (tokens[0, :len(prefix)] == prefix).all().item()):
            raise CognitiveKernelContractError(
                f"case {example.case_id} has a non-prefix assistant answer or exceeds the full context; do not truncate")
        full["labels"] = tokens.clone()
        full["labels"][0, :len(prefix)] = -100
        # Keep only real model inputs. The processor can return diagnostic
        # metadata such as video_metadata alongside tensor payloads.
        return {key: value for key, value in full.items()
                if hasattr(value, "shape") and key not in {"num_soft_tokens_per_image",
                                                              "num_soft_tokens_per_video"}}


class MultimodalFormationCandidate:
    """Inference from the trained media model, still behind the authority gate."""

    def __init__(self, model, processor, artifact_sha256: str, inference_run_id: str,
                 max_input_tokens: int, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.artifact_sha256 = artifact_sha256
        self.inference_run_id = inference_run_id
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def infer(self, *, context: FormationContextPacket,
              opened_sources: tuple[tuple[str, bytes], ...]) -> MemoryProposalBundle:
        import torch

        with formation_context_media_messages(context, opened_sources) as messages:
            encoded = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", do_sample_frames=False)
            _assert_modality_tensors(context, encoded)
            length = encoded["input_ids"].shape[-1]
            if length > self.max_input_tokens:
                raise CognitiveKernelContractError("multimodal context exceeds model input budget")
            model_inputs = {key: value.to(self.model.device) for key, value in encoded.items()
                            if hasattr(value, "shape") and key not in {
                                "num_soft_tokens_per_image", "num_soft_tokens_per_video"}}
            with torch.inference_mode():
                output = self.model.generate(
                    **model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id)
            answer = self.processor.decode(output[0][length:], skip_special_tokens=True).strip()
        try:
            record = json.loads(answer)
        except (ValueError, TypeError) as exc:
            raise CognitiveKernelContractError("multimodal model output is not exact JSON") from exc
        bundle = bundle_from_output(context, record,
                                    artifact_sha256=self.artifact_sha256,
                                    inference_run_id=self.inference_run_id)
        validate_formation_grounding(context, bundle, opened_sources)
        return bundle
