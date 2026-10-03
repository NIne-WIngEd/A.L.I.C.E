"""Complete raw-text ACFP documents to frozen full-layer tensor frames.

No chat template, generation, gold input, truncation, persistent feature cache
or source-authority promotion is implemented. All returned values are protected
runtime data. The caller must establish the private execution boundary before
reading a private document; this module does not create OS isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from .codec import CognitiveFrameDocument, canonical, fingerprint
from .contracts import (FrozenTokenBank, HEAD_FAMILIES, IdentityError,
                        IdentityFrame, IdentityModelConfig, SupportGraph)


@dataclass(frozen=True)
class ProducedFrame:
    frame: IdentityFrame
    binding: dict


class FrozenFrameProducer:
    """An admitted provider or explicitly marked public mechanical fixture.

    A production session loads verified publisher files through ``load``. Its
    close-and-verify receipt is required before treating its produced data as
    source-bound evidence. It never grants identity or training acceptance.
    """
    def __init__(self, tokenizer: Any, backbone: Any, config: IdentityModelConfig,
                 *, _fixture: bool = False):
        if _fixture is not True:
            raise IdentityError("use the verified loader for production feature sessions")
        config.validate()
        if (backbone.hidden_size, backbone.hidden_state_count) != (config.provider_width, config.provider_state_count):
            raise IdentityError("producer config differs from actual provider geometry")
        self.tokenizer, self.backbone, self.config = tokenizer, backbone, config
        self._prepared_path = None
        self._prepared_digest = None
        self._prepared_receipt_sha256 = None
        self._documents = []
        self._closed = False
        self._source_class = "public_mechanical_fixture"
        self._implementation = self._code_hashes()

    @staticmethod
    def _code_hashes() -> dict:
        root = Path(__file__).resolve().parent
        return {name: sha256((root / name).read_bytes()).hexdigest()
            for name in ("feature_producer.py", "codec.py", "contracts.py", "__init__.py")}

    @classmethod
    def load(cls, preparation_receipt: str | Path, *, expected_file_sha256: str,
             device: str = "cpu", learned_width: int = 256, attention_heads: int = 8,
             graph_layers: int = 3, readout_layers: int = 2) -> "FrozenFrameProducer":
        path = Path(preparation_receipt)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or path != path.resolve(strict=True)):
            raise IdentityError("preparation receipt requires an absolute nonsymlink path")
        if sha256(path.read_bytes()).hexdigest() != expected_file_sha256:
            raise IdentityError("preparation receipt differs from external file binding")
        from ..gemma_n0.preparation import verify_prepared
        from ..gemma_n0.backbone import load_prepared_gemma_n0
        from transformers import AutoProcessor
        prepared = verify_prepared(path)
        processor = AutoProcessor.from_pretrained(prepared["snapshot_path"],
            local_files_only=True, trust_remote_code=False)
        tokenizer = getattr(processor, "tokenizer", None)
        if not callable(tokenizer):
            raise IdentityError("prepared source has no local raw-text tokenizer")
        backbone = load_prepared_gemma_n0(path, device=device)
        config = IdentityModelConfig(backbone.hidden_size, backbone.hidden_state_count,
            learned_width, attention_heads, graph_layers, readout_layers)
        result = cls(tokenizer, backbone, config, _fixture=True)
        result._prepared_path, result._prepared_digest = path, expected_file_sha256
        result._prepared_receipt_sha256 = prepared["receipt_sha256"]
        result._source_class = "prepared_frozen_gemma_features_unqualified"
        return result

    def _tokenize(self, text: str) -> dict:
        value = self.tokenizer([text], padding=False, truncation=False,
            add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        if set(value) != {"input_ids", "attention_mask"}:
            raise IdentityError("raw tokenizer returned unsupported non-source fields")
        ids, mask = value["input_ids"], value["attention_mask"]
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                or ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any())
                or not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(ids.shape)
                or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any())
                or not bool(mask.bool().all())):
            raise IdentityError("raw tokenizer must preserve one complete nonpadded source")
        if ids.shape[1] > self.backbone.max_source_tokens:
            raise IdentityError("complete semantic input exceeds source capacity; truncation is forbidden")
        return {name: tensor.to(self.backbone.device) for name, tensor in value.items()}

    def produce(self, document: CognitiveFrameDocument) -> ProducedFrame:
        if self._closed:
            raise IdentityError("closed feature sessions cannot emit more evidence")
        if self._code_hashes() != self._implementation:
            raise IdentityError("producer implementation changed during the session")
        document.validate()
        document_sha256 = document.sha256
        # Pointer IDs are exclusively metadata. No preferred candidate or target
        # is available to these renderers; source classes remain explicit.
        texts = {"query": [(field.field_id, field.semantic_text()) for field in document.fields],
            "sources": [(entry.record.record_id, f"Evidence class: {entry.record.provenance_class}\n{entry.text}")
                        for entry in document.sources],
            "concepts": [(entry.record.concept_id, entry.text) for entry in document.concepts],
            "candidates": [(entry.candidate_id, entry.text) for entry in document.candidates],
            "relations": [(entry.entry_id, entry.text) for entry in document.relations]}
        for family, labels in document.labels.items():
            texts["label:" + family] = [(label.label_id,
                f"{label.name}\n{label.description}\nUnit: {label.unit}" +
                (f"\nRange: {label.minimum} to {label.maximum}" if family == "voice" else "")) for label in labels]
        # Admit every full entry before any forward. There is no partial-frame
        # fallback when one selected span or description exceeds capacity.
        encoded = {text: self._tokenize(text) for entries in texts.values() for _, text in entries}
        features = {}
        for text, inputs in encoded.items():
            result = self.backbone.extract_features(**inputs)
            if (len(result.all_hidden_states) != self.config.provider_state_count
                    or not torch.equal(result.attention_mask.bool(), inputs["attention_mask"].bool())):
                raise IdentityError("provider changed complete source geometry or attention")
            states = torch.stack(result.all_hidden_states, dim=1).detach()
            if (tuple(states.shape) != (1, self.config.provider_state_count, inputs["input_ids"].shape[1], self.config.provider_width)
                    or not bool(torch.isfinite(states).all())
                    or not torch.equal(result.hidden_states, result.all_hidden_states[-1])):
                raise IdentityError("provider must emit all actual finite source-aligned layers")
            features[text] = states
        reference = next(iter(features.values()))
        def bank(entries):
            count, length = max(1, len(entries)), max((features[text].shape[2] for _, text in entries), default=1)
            states = torch.zeros((1, count, self.config.provider_state_count, length, self.config.provider_width),
                                 dtype=reference.dtype, device=reference.device)
            token_mask = torch.zeros((1, count, length), dtype=torch.bool, device=reference.device)
            entry_mask = torch.zeros((1, count), dtype=torch.bool, device=reference.device)
            ids = [None] * count
            for index, (pointer, text) in enumerate(entries):
                value = features[text]
                n = value.shape[2]
                states[0, index, :, :n] = value[0]
                token_mask[0, index, :n] = True
                entry_mask[0, index] = True
                ids[index] = pointer
            return FrozenTokenBank(states.detach(), token_mask, entry_mask, (tuple(ids),))
        banks = {name: bank(entries) for name, entries in texts.items()}
        source_records = (tuple(entry.record for entry in document.sources) or (None,),)
        concept_records = (tuple(entry.record for entry in document.concepts) or (None,),)
        node_ids = [entry.record.record_id for entry in document.sources]
        # Empty source bank still occupies its one explicit masked placeholder.
        if not node_ids:
            node_ids = [None]
        node_ids += [entry.record.concept_id for entry in document.concepts]
        relation_ids = [entry.entry_id for entry in document.relations]
        long = lambda values: torch.tensor([values], dtype=torch.long, device=reference.device)
        edges = document.edges
        graph = SupportGraph(long([node_ids.index(edge.sender_id) for edge in edges]),
            long([node_ids.index(edge.receiver_id) for edge in edges]),
            long([relation_ids.index(edge.relation_id) for edge in edges]),
            torch.ones((1, len(edges)), dtype=torch.bool, device=reference.device),
            (tuple(edge.role for edge in edges),))
        allowed = lambda name: torch.tensor([[getattr(entry, name) for entry in document.candidates] or [False]],
                                           dtype=torch.bool, device=reference.device)
        frame = IdentityFrame(banks["query"], banks["sources"], banks["concepts"], banks["candidates"],
            banks["relations"], {family: banks["label:" + family] for family in HEAD_FAMILIES},
            source_records, concept_records, {family: (labels,) for family, labels in document.labels.items()},
            graph, allowed("authority_allowed"), allowed("context_allowed"))
        frame.validate(self.config)
        input_bindings = []
        for namespace, entries in texts.items():
            for pointer, text in entries:
                inputs = encoded[text]
                input_bindings.append({"namespace": namespace, "pointer": pointer,
                    "semantic_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "token_ids_sha256": sha256(canonical(inputs["input_ids"].cpu().tolist())).hexdigest(),
                    "token_count": inputs["input_ids"].shape[1]})
        if document.sha256 != document_sha256 or self._code_hashes() != self._implementation:
            raise IdentityError("document or producer implementation changed during emission")
        binding = {"schema": "alice-personality-produced-frame-binding-v1", "document_sha256": document_sha256,
            "frame_sha256": fingerprint(frame),
            "source_class": self._source_class, "source_geometry": [self.config.provider_state_count, self.config.provider_width],
            "prepared_receipt_file_sha256": self._prepared_digest, "entries": input_bindings,
            "chat_template": False, "truncation": False, "generation": False,
            "source_acceptance_authority": False, "qualification": "UNQUALIFIED",
            "persistent_feature_values_retained": False, "session_source_recheck_pending": True,
            "implementation_sha256": dict(self._implementation)}
        binding["binding_sha256"] = sha256(canonical(binding)).hexdigest()
        self._documents.append(binding["binding_sha256"])
        return ProducedFrame(frame, binding)

    def close_and_verify(self) -> dict:
        if self._closed:
            raise IdentityError("feature session was already closed")
        if self._code_hashes() != self._implementation:
            raise IdentityError("producer implementation changed during the session")
        if self._prepared_path is not None:
            if sha256(self._prepared_path.read_bytes()).hexdigest() != self._prepared_digest:
                raise IdentityError("preparation metadata changed during the feature session")
            from ..gemma_n0.preparation import verify_prepared
            prepared = verify_prepared(self._prepared_path)
            if prepared["receipt_sha256"] != self._prepared_receipt_sha256:
                raise IdentityError("source preparation changed during the feature session")
        receipt = {"schema": "alice-personality-feature-session-v1", "source_class": self._source_class,
            "prepared_receipt_file_sha256": self._prepared_digest, "frame_binding_sha256": list(self._documents),
            "prepared_receipt_sha256": self._prepared_receipt_sha256,
            "implementation_sha256": dict(self._implementation),
            "status": "CLOSED_SOURCE_VERIFIED" if self._prepared_path is not None else "PUBLIC_MECHANICS_ONLY",
            "session_source_recheck_passed": self._prepared_path is not None,
            "mechanical_fixture_only": self._prepared_path is None, "qualification": "UNQUALIFIED",
            "source_acceptance_authority": False, "persistent_feature_values_retained": False}
        receipt["receipt_sha256"] = sha256(canonical(receipt)).hexdigest()
        self._closed = True
        return receipt
