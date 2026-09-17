#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr


PREP_SCHEMA = "alice.eipm.n0.v02-evidence-selector-repair-preparation.v0.1"
CHECKPOINT_SCHEMA = "alice.eipm.n0.v02-evidence-selector-repair-checkpoint.v0.1"
RESULT_SCHEMA = "alice.eipm.n0.v02-evidence-selector-repair-result.v0.1"


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def adapter_forward(adapter: EvidenceViewAdapter, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return adapter(
        parent_field_states=batch["parent_field_states"],
        valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"],
        parent_field_weights=batch["parent_field_weights"],
    )


def set_prior_only(adapter: EvidenceViewAdapter) -> list[torch.nn.Parameter]:
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    for parameter in adapter.prior_head.parameters():
        parameter.requires_grad = True
    return [p for p in adapter.parameters() if p.requires_grad]


def dist_ce(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()


def kl_to_baseline(current: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    c = current.clamp_min(1e-8)
    b = baseline.clamp_min(1e-8)
    term = b * (b.log() - c.log())
    term = torch.where(valid, term, torch.zeros_like(term))
    return term.sum(dim=-1).mean()


def target_margin(weights: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    support = target > 0
    target_mass = (weights * support.to(weights.dtype)).sum(dim=-1)
    other = weights.masked_fill(~(valid & ~support), -1.0).max(dim=-1).values
    other = torch.where(other < 0.0, torch.zeros_like(other), other)
    return target_mass - other


def evaluate_pairs(graph: DualEndpointEvidenceGraphEncoder, adapter: EvidenceViewAdapter,
                   dataset: rr.PairDataset, payload: dict[str, Any], device: torch.device,
                   batch_size: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval(); adapter.eval()
    pairs = pair_ok_total = rows = row_ok_total = 0
    margins: list[float] = []
    candidate_min: list[float] = []
    family_ok: dict[str, list[int]] = defaultdict(list)
    family_margin: dict[str, list[float]] = defaultdict(list)
    family_candidate: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            batch = rr.to_device(batch, device)
            a = rr.unprefix(batch, "a"); b = rr.unprefix(batch, "b")
            aa = adapter_forward(adapter, a)
            ao = rr.graph_forward(graph, adapter, a); bo = rr.graph_forward(graph, adapter, b)
            at = a["target_distribution"].argmax(dim=-1); bt = b["target_distribution"].argmax(dim=-1)
            idx = torch.arange(at.size(0), device=device)
            aok = ao["field_weights"].argmax(dim=-1) == at
            bok = bo["field_weights"].argmax(dim=-1) == bt
            pok = aok & bok
            pm = torch.minimum(
                target_margin(ao["field_weights"], a["target_distribution"], a["valid_mask"]),
                target_margin(bo["field_weights"], b["target_distribution"], b["valid_mask"]),
            )
            cmin = torch.minimum(aa["field_weights"][idx, at], aa["field_weights"][idx, bt])
            pairs += int(pok.numel()); pair_ok_total += int(pok.sum().item())
            rows += int(aok.numel() + bok.numel()); row_ok_total += int(aok.sum().item() + bok.sum().item())
            margins.extend(float(x) for x in pm.cpu().tolist())
            candidate_min.extend(float(x) for x in cmin.cpu().tolist())
            for local, global_index in enumerate(batch["a_global_index"].cpu().tolist()):
                family = str(payload["families"][int(global_index)])
                family_ok[family].append(int(pok[local].item()))
                family_margin[family].append(float(pm[local].item()))
                family_candidate[family].append(float(cmin[local].item()))
    fam_acc = {k: sum(v)/len(v) for k,v in sorted(family_ok.items())}
    return {
        "pairs": pairs,
        "pair_accuracy": pair_ok_total / max(pairs,1),
        "family_pair_accuracy": fam_acc,
        "family_min_pair_accuracy": min(fam_acc.values()),
        "row_accuracy": row_ok_total / max(rows,1),
        "mean_graph_target_margin": sum(margins)/max(len(margins),1),
        "family_graph_target_margin": {k: sum(v)/len(v) for k,v in sorted(family_margin.items())},
        "mean_adapter_candidate_min_mass": sum(candidate_min)/max(len(candidate_min),1),
        "family_adapter_candidate_min_mass": {k: sum(v)/len(v) for k,v in sorted(family_candidate.items())},
    }


def evaluate_replay(graph: DualEndpointEvidenceGraphEncoder, adapter: EvidenceViewAdapter,
                    dataset: rr.RowDataset, payload: dict[str, Any], device: torch.device,
                    batch_size: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval(); adapter.eval()
    mass_by_family: dict[str, list[float]] = defaultdict(list)
    top1_by_family: dict[str, list[int]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            batch = rr.to_device(batch, device)
            out = rr.graph_forward(graph, adapter, batch)
            support = batch["target_distribution"] > 0
            mass = (out["field_weights"] * support.to(out["field_weights"].dtype)).sum(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            idx = torch.arange(pred.size(0), device=device)
            top1 = support[idx, pred]
            for local, global_index in enumerate(batch["global_index"].cpu().tolist()):
                family = str(payload["families"][int(global_index)])
                mass_by_family[family].append(float(mass[local].item()))
                top1_by_family[family].append(int(top1[local].item()))
    mass = {k: sum(v)/len(v) for k,v in sorted(mass_by_family.items())}
    top1 = {k: sum(v)/len(v) for k,v in sorted(top1_by_family.items())}
    return {
        "family_macro_target_support_mass": sum(mass.values())/max(len(mass),1),
        "family_min_target_support_mass": min(mass.values()),
        "family_target_support_mass": mass,
        "family_macro_top1_support_accuracy": sum(top1.values())/max(len(top1),1),
        "family_min_top1_support_accuracy": min(top1.values()),
        "family_top1_support_accuracy": top1,
    }


def save_checkpoint(adapter: EvidenceViewAdapter, output: Path, step: int, dev: dict[str, Any],
                    replay: dict[str, Any], parent_adapter_hash: str, parent_graph_hash: str,
                    curriculum_hash: str, prep_hash: str, root: Path) -> dict[str, Any]:
    from safetensors.torch import save_file
    cp = output / f"step-{step:08d}"; cp.mkdir(parents=True, exist_ok=True)
    weights = cp / "evidence_view_adapter.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()}, str(weights))
    receipt = {
        "schema": CHECKPOINT_SCHEMA, "status": "TRAINED_NOT_RATIFIED", "step": step,
        "git_revision": git_revision(root), "adapter_sha256": sha(weights),
        "parent_adapter_sha256": parent_adapter_hash, "parent_graph_sha256": parent_graph_hash,
        "curriculum_sha256": curriculum_hash, "preparation_receipt_sha256": prep_hash,
        "adapter_total_parameters": adapter.parameter_report()["total_parameters"],
        "trainable_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "trainable_scope": "prior_head_only", "trainable_scope_is_permanent_architecture_limit": False,
        "evidence_graph_mutated": False, "adapter_field_state_transform_mutated": False,
        "semantic_parent_mutated": False, "structured_parent_mutated": False,
        "graph_pooled_cosine_optimized": False, "hard_parameter_ceiling": None,
        "private_identity_data": False, "private_identity_gradient": False,
        "dev_selector_metrics": dev, "ordinary_replay_metrics": replay,
        "production_promotion_authorized": False,
    }
    (cp / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True); p.add_argument("--prep-receipt", required=True)
    p.add_argument("--curriculum", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--semantic-config", required=True); p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True); p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True); p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True); p.add_argument("--replay-cache", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=64); p.add_argument("--pair-batch-size", type=int, default=16)
    p.add_argument("--replay-batch-size", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=160); p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=10); p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--graph-loss-weight", type=float, default=0.60); p.add_argument("--neutral-loss-weight", type=float, default=0.25)
    p.add_argument("--replay-loss-weight", type=float, default=0.15)
    p.add_argument("--replay-macro-mass-tolerance", type=float, default=0.015)
    p.add_argument("--replay-min-mass-tolerance", type=float, default=0.025)
    p.add_argument("--replay-top1-tolerance", type=float, default=0.03)
    p.add_argument("--test-pair-accuracy", type=float, default=0.90)
    p.add_argument("--test-family-min-pair-accuracy", type=float, default=0.80)
    p.add_argument("--test-row-accuracy", type=float, default=0.95)
    args = p.parse_args()

    if not torch.cuda.is_available(): raise SystemExit("selector repair requires one CUDA device")
    device = torch.device("cuda"); seed_all(args.seed)
    root = Path(args.repo_root).resolve(); prep_path = Path(args.prep_receipt).resolve()
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("schema") != PREP_SCHEMA or prep.get("status") != "PASS_EVIDENCE_SELECTOR_REPAIR_READY_FOR_GPU":
        raise SystemExit("selector-repair preparation contract mismatch")
    if prep.get("git_revision") != git_revision(root): raise SystemExit("selector-repair preparation git drift")
    if prep.get("hard_parameter_ceiling") is not None or prep.get("scale_authorized") is not False:
        raise SystemExit("selector-repair preparation scaling contract drift")

    curriculum_path = Path(args.curriculum).resolve(); manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-evidence-selector-repair-curriculum.v0.1":
        raise SystemExit("selector-repair manifest schema mismatch")
    if manifest.get("compiled_sha256") != sha(curriculum_path): raise SystemExit("selector-repair curriculum hash mismatch")
    if manifest.get("frozen_latent_challenge_rows_used_for_training") is not False or manifest.get("parent_value_path_diagnostic_rows_used_for_training") is not False:
        raise SystemExit("selector-repair evaluation leakage")
    if manifest.get("graph_pooled_cosine_is_optimization_target") is not False:
        raise SystemExit("selector-repair objective drift")

    hash_inputs = prep.get("artifact_sha256", {})
    check_paths = {
        "curriculum": curriculum_path, "manifest": manifest_path,
        "semantic_config": Path(args.semantic_config).resolve(),
        "semantic_checkpoint": Path(args.semantic_checkpoint).resolve()/"alice_n0_v02.safetensors",
        "tokenizer": Path(args.tokenizer_dir).resolve()/"tokenizer.json",
        "structured_config": Path(args.structured_config).resolve(),
        "structured_checkpoint": Path(args.structured_checkpoint).resolve()/"structured_state.safetensors",
        "parent_adapter": Path(args.parent_adapter).resolve(), "parent_graph": Path(args.parent_graph).resolve(),
        "replay_cache": Path(args.replay_cache).resolve(),
        "trainer": Path(__file__).resolve(),
    }
    for key,path in check_paths.items():
        if not path.is_file(): raise SystemExit(f"missing selector-repair artifact: {key}={path}")
        if hash_inputs.get(key) != sha(path): raise SystemExit(f"selector-repair artifact drift: {key}")

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise SystemExit(f"refusing to overwrite selector repair output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file
    semantic_cfg = load_n0_config(Path(args.semantic_config).resolve())
    semantic = AliceN0V02Model(semantic_cfg)
    state = load_file(str(Path(args.semantic_checkpoint).resolve()/"alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    if missing or unexpected: raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS: raise SystemExit("semantic parameter count drift")
    for param in semantic.parameters(): param.requires_grad=False
    semantic.to(device).eval(); tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_cfg = rr.load_structured_config(Path(args.structured_config).resolve())
    structured = StructuredStateEncoder(structured_cfg)
    structured_cp = Path(args.structured_checkpoint).resolve()
    structured_state = load_file(str(structured_cp/"structured_state.safetensors"), device="cpu")
    structured.load_state_dict(structured_state, strict=True)
    for param in structured.parameters(): param.requires_grad=False

    rows = read_jsonl(curriculum_path)
    print("selector_repair_semantic_cache_start=true", flush=True)
    payload = rr.build_repair_cache(rows, semantic_model=semantic, tokenizer=tokenizer,
        structured_parent=structured, device=device, encode_batch_size=args.encode_batch_size, max_length=args.max_length)
    cache = output/"selector_repair_semantic_cache.pt"; torch.save(payload, cache)
    print(f"selector_repair_semantic_cache_complete=true path={cache}", flush=True)
    del semantic, state, structured, structured_state; torch.cuda.empty_cache()

    replay_payload = torch.load(Path(args.replay_cache).resolve(), map_location="cpu")
    parent_adapter_path = Path(args.parent_adapter).resolve(); adapter_state = load_file(str(parent_adapter_path), device="cpu")
    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device); adapter.load_state_dict(adapter_state, strict=True)
    baseline = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device); baseline.load_state_dict(adapter_state, strict=True)
    for param in baseline.parameters(): param.requires_grad=False
    baseline.eval(); trainable = set_prior_only(adapter)
    if sum(p.numel() for p in trainable) != sum(p.numel() for p in adapter.prior_head.parameters()):
        raise SystemExit("selector-repair trainable scope drift")

    graph = DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config()).to(device)
    graph.load_state_dict(load_file(str(Path(args.parent_graph).resolve()), device="cpu"), strict=True)
    for param in graph.parameters(): param.requires_grad=False
    graph.eval()

    train_pairs = rr.PairDataset(payload, rr.pair_indices(payload,"train")); dev_pairs = rr.PairDataset(payload, rr.pair_indices(payload,"dev")); test_pairs = rr.PairDataset(payload, rr.pair_indices(payload,"test"))
    replay_train_idx = [i for i,s in enumerate(replay_payload["splits"]) if s=="train"]
    replay_dev_idx = [i for i,s in enumerate(replay_payload["splits"]) if s=="dev"]
    replay_train = rr.RowDataset(replay_payload, replay_train_idx); replay_dev = rr.RowDataset(replay_payload, replay_dev_idx)

    initial_dev = evaluate_pairs(graph,adapter,dev_pairs,payload,device,args.eval_batch_size)
    baseline_replay = evaluate_replay(graph,baseline,replay_dev,replay_payload,device,args.eval_batch_size)
    print("initial_selector_dev="+json.dumps(initial_dev,sort_keys=True), flush=True)
    print("baseline_replay="+json.dumps(baseline_replay,sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps: return max((step+1)/max(args.warmup_steps,1),1e-6)
        progress=(step-args.warmup_steps)/max(args.max_steps-args.warmup_steps,1)
        return 0.5*(1.0+math.cos(math.pi*min(max(progress,0.0),1.0)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    pair_loader=DataLoader(train_pairs,batch_size=args.pair_batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    replay_loader=DataLoader(replay_train,batch_size=args.replay_batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed+1),num_workers=0)
    pair_iter=iter(pair_loader); replay_iter=iter(replay_loader); receipts={}

    for step in range(1,args.max_steps+1):
        try: pb=next(pair_iter)
        except StopIteration: pair_iter=iter(pair_loader); pb=next(pair_iter)
        try: rb=next(replay_iter)
        except StopIteration: replay_iter=iter(replay_loader); rb=next(replay_iter)
        pb=rr.to_device(pb,device); rb=rr.to_device(rb,device); a=rr.unprefix(pb,"a"); b=rr.unprefix(pb,"b")
        adapter.train(); optimizer.zero_grad(set_to_none=True)
        aa=adapter_forward(adapter,a); ba=adapter_forward(adapter,b)
        ao=rr.graph_forward(graph,adapter,a); bo=rr.graph_forward(graph,adapter,b)
        graph_loss=0.5*(dist_ce(ao["field_weights"],a["target_distribution"])+dist_ce(bo["field_weights"],b["target_distribution"]))
        union=((a["target_distribution"]>0)|(b["target_distribution"]>0)).to(aa["field_weights"].dtype)
        neutral_loss=0.5*(dist_ce(aa["field_weights"],union)+dist_ce(ba["field_weights"],union))
        current_replay=adapter_forward(adapter,rb)
        with torch.inference_mode(): base_replay=adapter_forward(baseline,rb)
        preserve_loss=kl_to_baseline(current_replay["field_weights"],base_replay["field_weights"],rb["valid_mask"])
        total=args.graph_loss_weight*graph_loss+args.neutral_loss_weight*neutral_loss+args.replay_loss_weight*preserve_loss
        total.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step(); scheduler.step()
        if step==1 or step%20==0:
            print(json.dumps({"step":step,"lr":scheduler.get_last_lr()[0],"loss":float(total.detach().cpu()),"graph_selection_loss":float(graph_loss.detach().cpu()),"adapter_neutrality_loss":float(neutral_loss.detach().cpu()),"replay_kl_loss":float(preserve_loss.detach().cpu())},sort_keys=True),flush=True)
        if step%args.save_every==0 or step==args.max_steps:
            dev=evaluate_pairs(graph,adapter,dev_pairs,payload,device,args.eval_batch_size)
            replay=evaluate_replay(graph,adapter,replay_dev,replay_payload,device,args.eval_batch_size)
            rec=save_checkpoint(adapter,output,step,dev,replay,sha(parent_adapter_path),sha(Path(args.parent_graph).resolve()),sha(curriculum_path),sha(prep_path),root)
            receipts[f"step-{step:08d}"]=rec; print("selector_checkpoint="+json.dumps(rec,sort_keys=True),flush=True)

    bm=float(baseline_replay["family_macro_target_support_mass"]); bmin=float(baseline_replay["family_min_target_support_mass"]); bt=float(baseline_replay["family_macro_top1_support_accuracy"])
    def replay_ok(m: dict[str,Any]) -> bool:
        return float(m["family_macro_target_support_mass"])>=bm-args.replay_macro_mass_tolerance and float(m["family_min_target_support_mass"])>=bmin-args.replay_min_mass_tolerance and float(m["family_macro_top1_support_accuracy"])>=bt-args.replay_top1_tolerance
    def score(r: dict[str,Any]) -> tuple[float,...]:
        d=r["dev_selector_metrics"]; rp=r["ordinary_replay_metrics"]
        return (1.0 if replay_ok(rp) else 0.0,float(d["family_min_pair_accuracy"]),float(d["pair_accuracy"]),float(d["row_accuracy"]),float(d["mean_graph_target_margin"]),float(d["mean_adapter_candidate_min_mass"]),-int(r["step"]))
    winner_key=max(receipts,key=lambda k:score(receipts[k])); winner=receipts[winner_key]
    adapter.load_state_dict(load_file(str(output/winner_key/"evidence_view_adapter.safetensors"),device="cpu"),strict=True); adapter.to(device).eval()
    test=evaluate_pairs(graph,adapter,test_pairs,payload,device,args.eval_batch_size)
    replay=evaluate_replay(graph,adapter,replay_dev,replay_payload,device,args.eval_batch_size)
    passed=replay_ok(replay) and float(test["pair_accuracy"])>=args.test_pair_accuracy and float(test["family_min_pair_accuracy"])>=args.test_family_min_pair_accuracy and float(test["row_accuracy"])>=args.test_row_accuracy and float(test["mean_graph_target_margin"])>0.0
    result={
        "schema":RESULT_SCHEMA,
        "status":"PASS_SELECTOR_REPAIR_HELDOUT_READY_FOR_EXISTING_LATENT_FRONTIER_CHALLENGE" if passed else "FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX",
        "git_revision":git_revision(root),"parent_adapter_sha256":sha(parent_adapter_path),"parent_graph_sha256":sha(Path(args.parent_graph).resolve()),"preparation_receipt_sha256":sha(prep_path),"curriculum_sha256":sha(curriculum_path),
        "initial_dev":initial_dev,"baseline_ordinary_replay":baseline_replay,"checkpoints":receipts,"selected_checkpoint":winner_key,"selected_adapter_sha256":winner["adapter_sha256"],
        "selection_policy":"replay_floor_then_worst_family_dev_pair_accuracy_then_dev_pair_accuracy_then_dev_row_accuracy_then_graph_margin_then_nonstarvation_then_earlier_checkpoint",
        "heldout_test_evaluated_once":True,"heldout_test_metrics":test,"selected_ordinary_replay":replay,
        "heldout_gate":{"pass":passed,"test_pair_accuracy_min":args.test_pair_accuracy,"test_family_min_pair_accuracy_min":args.test_family_min_pair_accuracy,"test_row_accuracy_min":args.test_row_accuracy,"mean_graph_target_margin_must_be_positive":True,"replay_macro_mass_tolerance":args.replay_macro_mass_tolerance,"replay_min_mass_tolerance":args.replay_min_mass_tolerance,"replay_top1_tolerance":args.replay_top1_tolerance,"thresholds_are_decision_gates_not_architecture_limits":True},
        "trainable_scope":"evidence_view_adapter.prior_head_only","trainable_scope_is_permanent_architecture_limit":False,"evidence_graph_mutated":False,"adapter_field_state_transform_mutated":False,"semantic_parent_mutated":False,"structured_parent_mutated":False,"graph_pooled_cosine_optimized":False,
        "frozen_latent_challenge_rows_used_for_training":False,"parent_value_path_diagnostic_rows_used_for_training":False,"private_identity_data":False,"private_identity_gradient":False,"hard_parameter_ceiling":None,"hard_parameter_floor":None,"scale_authorized":False,
        "scaling_interpretation":{"if_pass":"do not scale yet; run the existing latent frontier challenge with the repaired adapter and unchanged relation graph","if_fail_with_relation_signal_good_but_adapter_still_starves":"evidence supports broadening adapter selector trainable scope; prior-head-only is not a permanent limit","if_future_parent_trace_shows_target_signal_reaches_latent_input_but_latent_loses_it":"latent capacity/objective becomes eligible for a bounded scale study around the current operating point; no hard maximum or minimum is encoded","no_scale_from_this_result_alone":True},
        "automatic_rerun_or_hotfix_authorized":False,"next_action":"run_existing_latent_frontier_challenge_once_with_selected_repaired_adapter" if passed else "inspect_heldout_failure_once_and_choose_one_causal_next_change","production_promotion_authorized":False,"n0_complete":False,
    }
    (output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
