"""Legacy pretrained text MFM research route, superseded for personal MFM.

Examples come either from a separately admitted train/development corpus or
from an explicitly owner-authorized synthetic *training-only* curriculum.
The latter cannot by itself qualify a model. No FINAL data is opened here.

Run with --help. Torch/Transformers/PEFT are imported only for an actual run.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, require_identifier,
    require_sha256, require_text,
)
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning import (
    OBJECTIVE_VERSION, admitted_rows, curriculum_rows, mixture_rows, output_record,
    prompt_for_text_backbone, chat_messages_for_text_backbone,
)
from cognitive_kernel.formation_evaluation import FormationGoldCase, assess_formation


def supervised_tokens(tokenizer, example, max_sequence_tokens: int) -> dict[str, list[int]]:
    """Mask prompt tokens; supervise only the complete structured decision.

    Reject overlength histories explicitly. Cropping evidence, anchors, or
    dispositions silently would train a different and ungrounded objective.
    """
    example.validate()
    body = prompt_for_text_backbone(example.context, example.opened_sources)
    target = canonical_json_bytes(output_record(example.target)).decode("utf-8")
    if tokenizer.chat_template is None:
        raise CognitiveKernelContractError("pretrained tokenizer has no native chat template")
    messages = chat_messages_for_text_backbone(body)
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True)
    complete_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target}], tokenize=True,
        add_generation_prompt=False)
    if not isinstance(prompt_ids, list) or not isinstance(complete_ids, list):
        raise CognitiveKernelContractError("chat template did not return token IDs")
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise CognitiveKernelContractError("chat template target prefix differs from inference")
    answer_ids = complete_ids[len(prompt_ids):]
    if not answer_ids or len(complete_ids) > max_sequence_tokens:
        raise CognitiveKernelContractError(
            f"case {example.case_id} exceeds full-context token budget; do not truncate")
    return {"input_ids": complete_ids,
            "attention_mask": [1] * len(complete_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids}


def artifact_inventory(model_dir: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"path": path.relative_to(model_dir).as_posix(),
                     "sha256": digest.hexdigest(), "size": path.stat().st_size})
    if not rows:
        raise CognitiveKernelContractError("trained artifact has no model files")
    return tuple(rows)


def write_artifact_receipt(model_dir: Path, provenance: dict[str, object]) -> str:
    """Content-address the final model and tokenizer; provenance is separate."""
    from cognitive_kernel.canonical import canonical_sha256
    inventory = artifact_inventory(model_dir)
    digest = canonical_sha256(inventory)
    receipt = {"schema": "mfm-learned-artifact-v1", "artifact_sha256": digest,
               "inventory": inventory, "provenance": provenance}
    (model_dir.parent / "artifact-receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return digest


def verify_artifact_receipt(model_dir: Path) -> str:
    from cognitive_kernel.canonical import canonical_sha256
    receipt = json.loads((model_dir.parent / "artifact-receipt.json").read_bytes())
    if receipt.get("schema") != "mfm-learned-artifact-v1":
        raise CognitiveKernelContractError("unknown learned artifact receipt")
    current = artifact_inventory(model_dir)
    if list(current) != receipt.get("inventory") or canonical_sha256(current) != receipt.get("artifact_sha256"):
        raise CognitiveKernelContractError("model or tokenizer differs from frozen artifact")
    return require_sha256(receipt["artifact_sha256"], "artifact_sha256")


def evaluate_development(candidate, examples) -> dict[str, object]:
    """Report field-level errors; this is never a sealed FINAL assessment."""
    totals = {key: 0 for key in (
        "proposal_true_positives", "proposal_false_positives", "proposal_false_negatives",
        "disposition_true_positives", "disposition_false_positives",
        "disposition_false_negatives", "critical_failures", "invalid_outputs")}
    invalid: list[dict[str, str]] = []
    for example in examples:
        gold = FormationGoldCase(example.case_id, example.context,
                                 example.target.proposals, (),
                                 example.target.dispositions)
        try:
            output = candidate.infer(context=example.context,
                                     opened_sources=example.opened_sources)
            assessment = assess_formation(gold, output)
        except (CognitiveKernelContractError, ValueError) as exc:
            totals["invalid_outputs"] += 1
            invalid.append({"case_id": example.case_id, "error": str(exc)})
            continue
        for name in ("true_positives", "false_positives", "false_negatives"):
            totals[f"proposal_{name}"] += getattr(assessment, name)
            totals[f"disposition_{name}"] += getattr(assessment, f"disposition_{name}")
        totals["critical_failures"] += len(assessment.critical_failures)
    return {"schema": "mfm-development-diagnostics-v1", "cases": len(examples),
            "totals": totals, "invalid_cases": invalid,
            "qualification_status": "not-a-sealed-final-or-independent-semantic-review"}


def load_hf_candidate(artifact_dir: str | Path, *, inference_run_id: str,
                      max_input_tokens: int, max_new_tokens: int,
                      allow_derivative_research: bool = False):
    """Load a pretrained-derivative artifact for explicitly selected research."""
    if not allow_derivative_research:
        raise CognitiveKernelContractError(
            "pretrained-derived MFM cannot be loaded as the personal core; "
            "pass allow_derivative_research only for a labeled baseline")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from cognitive_kernel.formation_learning import PretrainedFormationCandidate

    artifact_dir = Path(artifact_dir)
    digest = verify_artifact_receipt(artifact_dir)
    if max_input_tokens < 1 or max_new_tokens < 1:
        raise CognitiveKernelContractError("generation token budgets must be positive")
    require_identifier(inference_run_id, "inference_run_id")
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(artifact_dir, local_files_only=True,
                                                 trust_remote_code=False, device_map="auto")
    model.eval()

    def generate(prompt: str) -> str:
        if tokenizer.chat_template is None:
            raise CognitiveKernelContractError("pretrained tokenizer has no native chat template")
        input_ids = tokenizer.apply_chat_template(
            chat_messages_for_text_backbone(prompt), tokenize=True,
            add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[-1] > max_input_tokens:
            raise CognitiveKernelContractError("formation context exceeds model input budget")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][input_ids.shape[-1]:],
                                skip_special_tokens=True).strip()

    return PretrainedFormationCandidate(digest, generate, inference_run_id)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--curriculum", type=Path,
                        help="frozen owner-authorized synthetic JSONL; training-only")
    inputs.add_argument("--curriculum-manifest", type=Path,
                        help="frozen manifest of multiple synthetic training inputs")
    inputs.add_argument("--admitted-manifest", type=Path,
                        help="frozen independent train/development/FINAL metadata manifest")
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--owner-authorization-ref")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate exact training data and report shape without loading weights")
    parser.add_argument("--research-derivative-only", action="store_true",
                        help="allow a nonproduct pretrained-weight research run")
    parser.add_argument("--base-model", help="pretrained open-weight HF model ID")
    parser.add_argument("--model-revision",
                        help="exact 40-hex upstream git commit, not a moving branch/tag")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequence-tokens", required=True, type=int)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--deepspeed-config", type=Path)
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="0 trains full weights; >0 uses a learned LoRA adapter")
    parser.add_argument("--lora-targets", nargs="*", default=("q_proj", "v_proj"))
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    require_sha256(args.input_sha256, "input_sha256")
    if not args.preflight_only:
        if not args.research_derivative_only:
            raise CognitiveKernelContractError(
                "pretrained MFM route is superseded for the personal model; "
                "see docs/MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md. "
                "Only explicit --research-derivative-only permits historical experiments")
        if not args.base_model or not args.model_revision:
            raise CognitiveKernelContractError("train run requires pretrained model ID and revision")
        if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision):
            raise CognitiveKernelContractError("base model revision must be an exact git commit")
    if any(x <= 0 for x in (args.max_sequence_tokens, args.epochs, args.batch_size,
                           args.gradient_accumulation, args.learning_rate,
                           args.max_new_tokens)) or args.lora_rank < 0:
        raise CognitiveKernelContractError("training hyperparameters must be positive")
    if not args.preflight_only and args.max_new_tokens >= args.max_sequence_tokens:
        raise CognitiveKernelContractError("answer generation budget leaves no source context")
    if args.curriculum or args.curriculum_manifest:
        if not args.owner_authorization_ref:
            raise CognitiveKernelContractError("synthetic curriculum needs owner authorization")
        require_text(args.owner_authorization_ref, "owner_authorization_ref", maximum=128)
        corpus_status = "owner-authorized-training-only-unqualified"
        def curriculum():
            if args.curriculum_manifest:
                return mixture_rows(args.curriculum_manifest,
                                    expected_sha256=args.input_sha256,
                                    owner_authorization_ref=args.owner_authorization_ref)
            return curriculum_rows(args.curriculum, expected_sha256=args.input_sha256,
                                   owner_authorization_ref=args.owner_authorization_ref)
        if args.preflight_only:
            counts = {"train_cases": 0, "development_cases": 0,
                      "source_count": 0, "proposal_count": 0,
                      "disposition_count": 0}
            for example in curriculum():
                prompt_for_text_backbone(example.context, example.opened_sources)
                counts["train_cases"] += 1
                counts["source_count"] += len(example.opened_sources)
                counts["proposal_count"] += len(example.target.proposals)
                counts["disposition_count"] += len(example.target.dispositions)
            print(json.dumps({"objective": OBJECTIVE_VERSION,
                              "corpus_status": corpus_status,
                              "train_sha256": args.input_sha256, **counts}, sort_keys=True))
            return
        train = tuple(curriculum())
        dev = ()
    else:
        if args.owner_authorization_ref:
            raise CognitiveKernelContractError("owner authorization ref belongs to synthetic route")
        admitted = admit_formation_corpus(args.admitted_manifest, expected_sha256=args.input_sha256)
        train = tuple(admitted_rows(admitted, split="train"))
        dev = tuple(admitted_rows(admitted, split="development"))
        corpus_status = "admitted-train-development-final-sealed"
    if not train:
        raise CognitiveKernelContractError("no training examples")
    for example in (*train, *dev):
        prompt_for_text_backbone(example.context, example.opened_sources)
    if args.preflight_only:
        print(json.dumps({"objective": OBJECTIVE_VERSION, "corpus_status": corpus_status,
                          "train_cases": len(train), "development_cases": len(dev),
                          "source_count": sum(len(x.opened_sources) for x in (*train, *dev)),
                          "proposal_count": sum(len(x.target.proposals) for x in (*train, *dev)),
                          "disposition_count": sum(len(x.target.dispositions) for x in (*train, *dev)),
                          "train_sha256": args.input_sha256}, sort_keys=True))
        return
    # Import after source admission and complete contract validation; a bad
    # curriculum cannot consume a GPU just to discover malformed labels.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    if not torch.cuda.is_available():
        raise CognitiveKernelContractError("learned MFM weight run requires an available GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise CognitiveKernelContractError("bf16 precision is unavailable on this GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Verify the actual tokenizer's full-context fit before allocating model
    # weights. Retain source examples, not three enormous Python token arrays.
    longest_prompt = 0
    longest_answer = 0
    for example in (*train, *dev):
        sample = supervised_tokens(tokenizer, example, args.max_sequence_tokens)
        prompt_length = sample["labels"].count(-100)
        longest_prompt = max(longest_prompt, prompt_length)
        longest_answer = max(longest_answer, len(sample["labels"]) - prompt_length)
    if longest_answer > args.max_new_tokens:
        raise CognitiveKernelContractError(
            f"longest formation answer needs {longest_answer} tokens; generation budget is "
            f"{args.max_new_tokens}")
    if longest_prompt + args.max_new_tokens > args.max_sequence_tokens:
        raise CognitiveKernelContractError(
            "longest context plus generation budget exceeds full model token budget")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.model_revision,
                                                trust_remote_code=False,
                                                torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    if args.lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
            lora_alpha=2 * args.lora_rank, target_modules=list(args.lora_targets)))
    model.config.use_cache = False

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return supervised_tokens(tokenizer, self.values[index], args.max_sequence_tokens)

    def collate(items):
        max_len = max(len(item["input_ids"]) for item in items)
        return {name: torch.tensor([
            item[name] + [(-100 if name == "labels" else
                           0 if name == "attention_mask" else tokenizer.pad_token_id)] *
            (max_len - len(item[name])) for item in items], dtype=torch.long)
            for name in ("input_ids", "attention_mask", "labels")}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model, train_dataset=Dataset(train),
        eval_dataset=Dataset(dev) if dev else None,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate, bf16=args.precision == "bf16",
            gradient_checkpointing=True, eval_strategy="epoch" if dev else "no",
            save_strategy="epoch", seed=args.seed, save_safetensors=True,
            deepspeed=str(args.deepspeed_config) if args.deepspeed_config else None,
            report_to="none"))
    trainer.train()
    if dev:
        trainer.evaluate()
    final_dir = args.output_dir / "model"
    final_dir.mkdir(parents=True, exist_ok=True)
    if args.lora_rank:
        # Shipping an adapter alone silently depends on an unfrozen external
        # base model. Merge weights into one self-contained local artifact.
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    provenance = {"objective": OBJECTIVE_VERSION, "corpus_status": corpus_status,
                  "weight_lineage": "third-party-pretrained-derivative-research-only",
                  "qualified_for_product": False,
                  "corpus_sha256": args.input_sha256,
                  "owner_authorization_ref": args.owner_authorization_ref,
                  "base_model": args.base_model, "model_revision": args.model_revision,
                  "train_cases": len(train), "development_cases": len(dev),
                  "max_sequence_tokens": args.max_sequence_tokens,
                  "longest_prompt_tokens": longest_prompt,
                  "longest_answer_tokens": longest_answer,
                  "max_new_tokens": args.max_new_tokens,
                  "epochs": args.epochs, "batch_size": args.batch_size,
                  "gradient_accumulation": args.gradient_accumulation,
                  "learning_rate": args.learning_rate,
                  "precision": args.precision, "lora_rank": args.lora_rank,
                  "lora_targets": args.lora_targets,
                  "deepspeed_sha256": (sha256(args.deepspeed_config.read_bytes()).hexdigest()
                                       if args.deepspeed_config else None),
                  "trainer_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "learning_contract_sha256": sha256(
                      Path(sys.modules[prompt_for_text_backbone.__module__].__file__).read_bytes()
                  ).hexdigest(),
                  "transformers_version": __import__("transformers").__version__,
                  "torch_version": torch.__version__, "seed": args.seed}
    digest = write_artifact_receipt(final_dir, provenance)
    if dev:
        candidate = load_hf_candidate(final_dir, inference_run_id="development-eval",
                                      max_input_tokens=args.max_sequence_tokens - args.max_new_tokens,
                                      max_new_tokens=args.max_new_tokens,
                                      allow_derivative_research=True)
        report = evaluate_development(candidate, dev)
        (args.output_dir / "development-diagnostics.json").write_bytes(
            canonical_json_bytes(report) + b"\n")
    print(json.dumps({"artifact_sha256": digest, "corpus_status": corpus_status,
                      "train_cases": len(train), "development_cases": len(dev)}, sort_keys=True))


if __name__ == "__main__":
    main()
