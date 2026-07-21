from __future__ import annotations
import json, math, re, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[tuple[str,str], Any] = {}

@dataclass(frozen=True)
class ClaimEntailmentPolicy:
    policy_id: str
    enabled: bool
    model_id: str
    revision: str
    label_order: tuple[str, str, str]
    entailment_threshold: float
    contradiction_threshold: float
    maximum_evidence_passages_per_claim: int
    sentence_window_size: int
    sentence_window_stride: int
    maximum_window_characters: int
    maximum_evidence_windows_per_claim: int
    batch_size: int
    drop_neutral_claims: bool
    drop_contradicted_claims: bool
    fallback_answer: str
    private_text_uploaded: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path

def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "claim_entailment_policy.json"

def load_claim_entailment_policy(path: Path|None=None) -> ClaimEntailmentPolicy:
    source=(path or default_policy_path()).expanduser().resolve(strict=True)
    data=json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("claim_entailment_policy_schema_version",-1)) != 1:
        raise ValueError("Unsupported claim-entailment policy schema")
    p=ClaimEntailmentPolicy(
        policy_id=str(data["policy_id"]),
        enabled=bool(data["enabled"]),
        model_id=str(data["model_id"]),
        revision=str(data.get("revision","main")),
        label_order=tuple(str(value) for value in data.get("label_order", ["contradiction","entailment","neutral"])),
        entailment_threshold=float(data["entailment_threshold"]),
        contradiction_threshold=float(data["contradiction_threshold"]),
        maximum_evidence_passages_per_claim=int(data["maximum_evidence_passages_per_claim"]),
        sentence_window_size=int(data.get("sentence_window_size", 3)),
        sentence_window_stride=int(data.get("sentence_window_stride", 1)),
        maximum_window_characters=int(data.get("maximum_window_characters", 900)),
        maximum_evidence_windows_per_claim=int(
            data.get("maximum_evidence_windows_per_claim", 12)
        ),
        batch_size=int(data["batch_size"]),
        drop_neutral_claims=bool(data["drop_neutral_claims"]),
        drop_contradicted_claims=bool(data["drop_contradicted_claims"]),
        fallback_answer=str(data["fallback_answer"]),
        private_text_uploaded=bool(data["private_text_uploaded"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )
    if len(p.label_order) != 3:
        raise ValueError("label_order must contain exactly three labels")
    if set(p.label_order) != {"entailment","neutral","contradiction"}:
        raise ValueError("label_order must contain entailment, neutral, and contradiction")
    if not 0 <= p.entailment_threshold <= 1:
        raise ValueError("Invalid entailment threshold")
    if not 0 <= p.contradiction_threshold <= 1:
        raise ValueError("Invalid contradiction threshold")
    if p.maximum_evidence_passages_per_claim < 1:
        raise ValueError("Evidence-passage limit must be positive")
    if p.sentence_window_size < 1:
        raise ValueError("sentence_window_size must be positive")
    if p.sentence_window_stride < 1:
        raise ValueError("sentence_window_stride must be positive")
    if p.maximum_window_characters < 200:
        raise ValueError("maximum_window_characters is too small")
    if p.maximum_evidence_windows_per_claim < 1:
        raise ValueError("maximum_evidence_windows_per_claim must be positive")
    if p.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if p.private_text_uploaded:
        raise ValueError("Private text may not be uploaded")
    if any((p.memory_write_allowed,p.external_action_allowed,p.tool_calling_allowed,p.web_access_allowed)):
        raise ValueError("Claim support gate must remain read-only and offline")
    return p

def model_directory(*, vault_root: Path, policy: ClaimEntailmentPolicy) -> Path:
    safe=policy.model_id.replace("/","__")
    rev=policy.revision.replace("/","_").replace(":","_")
    return vault_root / "models" / "nli" / f"{safe}__{rev}"

def prepare_claim_entailment_model(*, vault_root: Path, policy_path: Path|None=None) -> dict[str,Any]:
    from huggingface_hub import HfApi, snapshot_download
    vault_root=vault_root.expanduser().resolve(strict=True)
    policy=load_claim_entailment_policy(policy_path)
    root=model_directory(vault_root=vault_root, policy=policy)
    root.parent.mkdir(parents=True, exist_ok=True)
    info=HfApi().model_info(policy.model_id, revision=policy.revision)
    resolved=str(info.sha)
    snapshot_download(repo_id=policy.model_id, revision=resolved, local_dir=str(root))
    result={
        "claim_entailment_model_manifest_schema_version":1,
        "model_id":policy.model_id,
        "requested_revision":policy.revision,
        "resolved_revision":resolved,
        "prepared_at":datetime.now(timezone.utc).isoformat(),
        "private_data_used_during_download":False,
        "model_path":str(root),
    }
    export=vault_root/"manifests"/"exports"/f"claim-entailment-model-summary-{uuid.uuid4().hex}.json"
    export.parent.mkdir(parents=True,exist_ok=True)
    export.write_text(json.dumps(result,indent=2),encoding="utf-8")
    result["summary_path"]=str(export)
    return result

def load_local_claim_entailment_model(*, vault_root: Path, policy_path: Path|None=None, device: str="auto"):
    policy=load_claim_entailment_policy(policy_path)
    root=model_directory(vault_root=vault_root.expanduser().resolve(strict=True), policy=policy)
    if not root.is_dir():
        raise FileNotFoundError(
            "Local NLI claim-support model is missing. Run "
            "scripts/prepare_claim_entailment_model.py first."
        )
    key=(str(root),device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], policy
    from sentence_transformers import CrossEncoder
    kwargs={}
    if device!="auto":
        kwargs["device"]=device
    model=CrossEncoder(str(root), **kwargs)
    _MODEL_CACHE[key]=model
    return model, policy

def _softmax(values) -> list[float]:
    vals=[float(v) for v in values]
    peak=max(vals)
    exps=[math.exp(v-peak) for v in vals]
    total=sum(exps)
    return [v/total for v in exps]

def _split_passages(text: str) -> list[str]:
    text=str(text or "").strip()
    if not text:
        return []
    parts=re.split(r"\n\s*\n(?=Passage\s+\d+\s*:)", text)
    cleaned=[]
    for part in parts:
        part=re.sub(r"^Passage\s+\d+\s*:\s*","",part.strip())
        if part:
            cleaned.append(part)
    return cleaned or [text]

def _premise_prefix(item: dict[str,Any]) -> str:
    relation=str(item.get("owner_relation","unknown"))
    if relation=="owner_self_record":
        return "Trusted metadata: this source is a self-record belonging to the user. "
    if relation=="owner_related_record":
        return "Trusted metadata: this source is related to the user, but not every statement necessarily describes the user. "
    return ""

def _sentence_split(text: str) -> list[str]:
    """Conservative sentence splitting for private local NLI windows."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )
    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def _lexical_terms(text: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "did", "do", "does", "for", "from", "have", "i",
        "in", "is", "it", "my", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "what", "with",
        "you", "your",
    }
    return {
        token
        for token in re.findall(
            r"[a-z0-9][a-z0-9_-]+",
            str(text or "").casefold(),
        )
        if token not in stop
    }


def _claim_window_score(
    claim_text: str,
    window_text: str,
) -> float:
    claim_terms = _lexical_terms(claim_text)
    if not claim_terms:
        return 0.0
    window_terms = _lexical_terms(window_text)
    overlap = len(
        claim_terms.intersection(window_terms)
    )
    return overlap / len(claim_terms)


def _build_sentence_windows(
    *,
    passage: str,
    window_size: int,
    stride: int,
    maximum_characters: int,
) -> list[str]:
    sentences = _sentence_split(passage)
    if not sentences:
        return []

    windows: list[str] = []
    for start in range(0, len(sentences), stride):
        selected = sentences[
            start : start + window_size
        ]
        if not selected:
            continue

        text = " ".join(selected).strip()
        if not text:
            continue

        if len(text) > maximum_characters:
            text = text[:maximum_characters].rsplit(
                " ",
                1,
            )[0].strip()

        if text and text not in windows:
            windows.append(text)

        if start + window_size >= len(sentences):
            break

    return windows


def cited_passages_for_claim(
    *,
    claim: dict[str,Any],
    context_package: dict[str,Any],
    limit: int,
    sentence_window_size: int = 3,
    sentence_window_stride: int = 1,
    maximum_window_characters: int = 900,
    maximum_windows: int = 12,
) -> list[dict[str,Any]]:
    """Return compact claim-focused NLI windows from cited evidence only.

    The original response evidence passages can be long and contain multiple
    unrelated resume/project facts. NLI models trained on sentence pairs may
    classify a broad passage as neutral even when one compact span entails the
    claim. This function builds short overlapping sentence windows, ranks them
    by deterministic lexical overlap with the claim, and returns only windows
    from the claim's own cited sources.
    """
    evidence={
        str(i["citation"]):i
        for i in context_package.get("evidence",[])
    }
    claim_text = str(
        claim.get("text", "")
    ).strip()
    candidates: list[dict[str, Any]] = []

    for citation in claim.get("citations",[]):
        citation=str(citation)
        item=evidence.get(citation)
        if item is None:
            continue

        prefix=_premise_prefix(item)
        raw_passages = _split_passages(
            str(item.get("context_text",""))
        )[:limit]

        for passage_index, passage in enumerate(
            raw_passages,
            start=1,
        ):
            windows = _build_sentence_windows(
                passage=passage,
                window_size=sentence_window_size,
                stride=sentence_window_stride,
                maximum_characters=maximum_window_characters,
            )

            # If sentence splitting fails to create windows, retain a compact
            # fallback rather than scoring the full long passage.
            if not windows and passage.strip():
                windows = [
                    passage[:maximum_window_characters]
                ]

            for window_index, window in enumerate(
                windows,
                start=1,
            ):
                candidates.append(
                    {
                        "citation": citation,
                        "premise": prefix + window,
                        "lexical_score": _claim_window_score(
                            claim_text,
                            window,
                        ),
                        "passage_index": passage_index,
                        "window_index": window_index,
                        "window_character_count": len(window),
                    }
                )

    candidates.sort(
        key=lambda item: (
            -float(item["lexical_score"]),
            str(item["citation"]),
            int(item["passage_index"]),
            int(item["window_index"]),
        )
    )

    return candidates[:maximum_windows]

def score_claim_support(
    *,
    claim: dict[str,Any],
    context_package: dict[str,Any],
    model,
    policy: ClaimEntailmentPolicy,
) -> dict[str,Any]:
    passages=cited_passages_for_claim(
        claim=claim,
        context_package=context_package,
        limit=policy.maximum_evidence_passages_per_claim,
        sentence_window_size=policy.sentence_window_size,
        sentence_window_stride=policy.sentence_window_stride,
        maximum_window_characters=policy.maximum_window_characters,
        maximum_windows=policy.maximum_evidence_windows_per_claim,
    )
    if not passages:
        return {
            "decision":"drop_no_evidence",
            "best_entailment_probability":0.0,
            "maximum_contradiction_probability":0.0,
            "supporting_citation":"",
            "passage_count":0,
        }

    hypothesis=str(claim.get("text","")).strip()
    pairs=[(p["premise"],hypothesis) for p in passages]
    logits=model.predict(pairs,batch_size=policy.batch_size,show_progress_bar=False)

    best_entailment=-1.0
    max_contradiction=-1.0
    best_citation=""
    best_probs=None
    for passage,row in zip(passages,logits):
        probs=_softmax(row)
        probabilities = {label: float(probability) for label, probability in zip(policy.label_order, probs)}
        entailment = probabilities["entailment"]
        contradiction = probabilities["contradiction"]
        if entailment>best_entailment:
            best_entailment=entailment
            best_citation=passage["citation"]
            best_probs=probabilities
        max_contradiction=max(max_contradiction,contradiction)

    if best_entailment >= policy.entailment_threshold:
        decision="keep_entailment"
    elif max_contradiction >= policy.contradiction_threshold and policy.drop_contradicted_claims:
        decision="drop_contradiction"
    elif policy.drop_neutral_claims:
        decision="drop_neutral"
    else:
        decision="keep_uncertain"

    return {
        "decision":decision,
        "best_entailment_probability":round(best_entailment,6),
        "maximum_contradiction_probability":round(max_contradiction,6),
        "supporting_citation":best_citation,
        "best_label_probabilities":{
            label: round(float(probability), 6)
            for label, probability in (best_probs or {}).items()
        },
        "passage_count":len(passages),
        "evidence_windowing": True,
        "maximum_window_characters": (
            policy.maximum_window_characters
        ),
    }

def filter_model_output_by_entailment(
    *,
    model_output: dict[str,Any],
    context_package: dict[str,Any],
    model,
    policy: ClaimEntailmentPolicy,
    answer_renderer,
) -> tuple[dict[str,Any],dict[str,Any]]:
    import copy
    output=copy.deepcopy(model_output)
    claims=list(output.get("claims",[]))
    kept=[]
    assessments=[]
    for index,claim in enumerate(claims,start=1):
        assessment=score_claim_support(
            claim=claim,
            context_package=context_package,
            model=model,
            policy=policy,
        )
        assessment["claim_index"]=index
        assessments.append(assessment)
        if assessment["decision"].startswith("keep_"):
            kept.append(claim)

    output["claims"]=kept
    if kept:
        output["answer"]=answer_renderer(kept)
        if output.get("answer_type")=="insufficient_evidence":
            output["answer_type"]="grounded"
    else:
        output["answer_type"]="insufficient_evidence"
        output["answer"]=policy.fallback_answer
        output["contradiction_notes"]=[]
        output["uncertainty_notes"]=[
            "Generated claims were removed because the cited evidence did not meet the local entailment threshold."
        ]

    summary={
        "enabled": True,
        "policy_id":policy.policy_id,
        "model_id":policy.model_id,
        "label_order":list(policy.label_order),
        "entailment_threshold":policy.entailment_threshold,
        "contradiction_threshold":policy.contradiction_threshold,
        "input_claim_count":len(claims),
        "kept_claim_count":len(kept),
        "dropped_claim_count":len(claims)-len(kept),
        "assessments":assessments,
        "private_text_uploaded":False,
        "memory_write_allowed":False,
        "external_action_allowed":False,
        "tool_calling_allowed":False,
        "web_access_allowed":False,
    }
    return output,summary
