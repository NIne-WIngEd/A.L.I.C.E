"""Frozen, features-only Gemma representations for the personality V1 route.

N0 supplies source representations, not identity, personal judgment or voice.
Bypassing the publisher language head does not remove inherited priors from
these representations. Runtime admission and the learned personal stages need
their own observed qualification. Importing this file does not import Torch or
Transformers, read a model, or open private source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


# Gemma4UnifiedModel.forward media inputs, intersected with processor outputs.
# Processor bookkeeping and caller-selected attention/position controls are
# outside this boundary. See the official Transformers gemma4_unified model
# and processing modules; the runtime itself remains pinned by preparation.
_PROCESSOR_MEDIA_INPUTS = frozenset({
    "pixel_values", "pixel_values_videos", "input_features", "input_features_mask",
    "mm_token_type_ids", "image_position_ids", "video_position_ids",
})


class BackboneError(ValueError):
    """A prepared backbone or source tensor violates the features-only contract."""


@dataclass(frozen=True)
class FrozenFeatures:
    """Detached source states and their complete binary attention mask.

    These tensors carry licensed representation ancestry. They have no direct
    authority over the person's identity, memory or downstream judgment.
    """

    hidden_states: Any
    attention_mask: Any


class _FrozenGemmaN0:
    """Internal wrapper; only the verified loader admits a publisher artifact."""

    def __init__(self, representation: Any, *, torch: Any, hidden_size: int,
                 max_source_tokens: int, device: Any):
        if type(hidden_size) is not int or hidden_size < 1:
            raise BackboneError("representation hidden size must be a positive integer")
        if type(max_source_tokens) is not int or max_source_tokens < 1:
            raise BackboneError("representation context budget must be a positive integer")
        self._representation = representation
        self._torch = torch
        self._device = torch.device(device)
        self.hidden_size = hidden_size
        self.max_source_tokens = max_source_tokens
        self._freeze()

    @property
    def training(self) -> bool:
        return False

    @property
    def device(self) -> Any:
        return self._device

    def _freeze(self) -> None:
        self._representation.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "_FrozenGemmaN0":
        if type(mode) is not bool:
            raise BackboneError("training mode must be a boolean")
        self._freeze()
        return self

    def eval(self) -> "_FrozenGemmaN0":
        return self.train(False)

    def extract_features(self, *, input_ids: Any, attention_mask: Any,
                         **source_tensors: Any) -> FrozenFeatures:
        """Encode all admitted source positions; never generate publisher prose.

        Processor-produced media tensors can accompany source token IDs. No
        string prompt, chat template, decoder input, label or generation option
        may enter this interface. Exceeded budgets fail without truncation.
        """
        torch = self._torch
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise BackboneError("source IDs and attention mask must be tensors")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise BackboneError("source IDs and attention mask must be two-dimensional")
        if input_ids.shape != attention_mask.shape or any(size < 1 for size in input_ids.shape):
            raise BackboneError("source IDs and attention mask require the same nonempty shape")
        if input_ids.shape[1] > self.max_source_tokens:
            raise BackboneError("complete source exceeds the context budget; truncation is forbidden")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if input_ids.dtype not in integer_dtypes or bool((input_ids < 0).any()):
            raise BackboneError("source token IDs must be nonnegative integers")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise BackboneError("source attention mask must contain only zero or one")
        mask = attention_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise BackboneError("every source row must expose at least one evidence position")
        forbidden = {
            "labels", "decoder_input_ids", "decoder_attention_mask", "inputs_embeds",
            "past_key_values", "use_cache", "return_dict", "output_hidden_states",
            "output_attentions", "logits_to_keep", "generation_config", "max_new_tokens",
        }
        if forbidden.intersection(source_tensors):
            raise BackboneError("decoder and generation controls are outside the N0 feature interface")
        if set(source_tensors) - _PROCESSOR_MEDIA_INPUTS:
            raise BackboneError("additional inputs must be known Gemma 4 processor media tensors")
        if any(not isinstance(value, torch.Tensor) for value in source_tensors.values()):
            raise BackboneError("additional source inputs must be processor-produced tensors")
        payload = {"input_ids": input_ids.to(self._device),
                   "attention_mask": mask.to(self._device)}
        payload.update({name: value.to(self._device) for name, value in source_tensors.items()})
        # Reapply after any external training-mode change; only .model is held,
        # so neither publisher LM forward nor generate is reachable here.
        self._freeze()
        with torch.no_grad():
            encoded = self._representation(**payload, use_cache=False, return_dict=True)
        states = getattr(encoded, "last_hidden_state", None)
        expected_shape = (*input_ids.shape, self.hidden_size)
        if not isinstance(states, torch.Tensor) or tuple(states.shape) != expected_shape:
            raise BackboneError("representation states do not preserve source positions and hidden size")
        if not states.is_floating_point() or not bool(torch.isfinite(states).all()):
            raise BackboneError("representation states must be finite floating-point features")
        return FrozenFeatures(hidden_states=states.detach(),
                              attention_mask=payload["attention_mask"].detach())

    def __call__(self, *, input_ids: Any, attention_mask: Any,
                 **source_tensors: Any) -> FrozenFeatures:
        return self.extract_features(input_ids=input_ids, attention_mask=attention_mask,
                                     **source_tensors)


def load_prepared_gemma_n0(receipt_path: str | Path, *, device: str = "cpu",
                          dtype: str | None = None) -> _FrozenGemmaN0:
    """Load only a freshly verified personality-role clone with pinned runtime.

    This is source-custody and feature-interface admission, not approval of a
    neutral foundation or a qualified personality model. Model packages are
    imported only after the complete preparation/custody check succeeds.
    """
    from .preparation import verify_prepared

    prepared = verify_prepared(receipt_path)
    runtime = prepared["runtime"]
    selected_dtype = runtime["dtype"] if dtype is None else dtype
    if selected_dtype != runtime["dtype"]:
        raise BackboneError("requested dtype differs from the frozen preparation runtime")
    torch = import_module("torch")
    transformers = import_module("transformers")
    actual_runtime = {
        "backend": "transformers", "dtype": selected_dtype,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }
    if actual_runtime != runtime:
        raise BackboneError("installed runtime differs from the frozen preparation runtime")
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if selected_dtype not in dtypes:
        raise BackboneError("unsupported prepared representation dtype")
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise BackboneError("representation device must be CPU or CUDA")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise BackboneError("requested CUDA representation device is unavailable")
    base = transformers.AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtypes[selected_dtype])
    if getattr(base.config, "model_type", None) != "gemma4_unified" or not hasattr(base, "model"):
        raise BackboneError("prepared source lacks the expected Gemma 4 representation path")
    text_config = getattr(base.config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    context_budget = getattr(text_config, "max_position_embeddings", None)
    if type(hidden_size) is not int or hidden_size < 1 or type(context_budget) is not int or context_budget < 1:
        raise BackboneError("prepared source lacks explicit positive feature and context dimensions")
    base.to(selected_device).requires_grad_(False).eval()
    if any(parameter.device != selected_device for parameter in base.parameters()):
        # CUDA without an explicit index resolves to the current CUDA device.
        resolved_device = (torch.device("cuda", torch.cuda.current_device())
                           if selected_device.type == "cuda" and selected_device.index is None
                           else selected_device)
        if any(parameter.device != resolved_device for parameter in base.parameters()):
            raise BackboneError("prepared source was offloaded or loaded on an unexpected device")
    # The full language-model object goes out of scope. Keep only its source
    # representation module, whose features remain dependent on publisher priors.
    representation = _FrozenGemmaN0(base.model, torch=torch, hidden_size=hidden_size,
                                    max_source_tokens=context_budget, device=selected_device)
    # Loading is an interval in which a persistent snapshot, receipt, interface
    # or code change could otherwise survive the initial custody check. Direct
    # consumers receive a wrapper only if the complete admission still holds.
    reverified = verify_prepared(receipt_path, expected_runtime=actual_runtime)
    if reverified != prepared:
        raise BackboneError("prepared artifact changed while loading the representation")
    return representation
