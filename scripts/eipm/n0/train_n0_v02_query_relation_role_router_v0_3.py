#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_graph_query_relation_role_router import QueryRelationRoleRouterEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_evidence_selector_repair_v0_1 as sr
import train_n0_v02_relation_endpoint_repair_v0_2 as endpoint

PREP_SCHEMA="alice.eipm.n0.v02-query-relation-role-router-preparation.v0.3"
MANIFEST_SCHEMA="alice.eipm.n0.v02-query-relation-role-router-curriculum.v0.3"
RESULT_SCHEMA="alice.eipm.n0.v02-query-relation-role-router-result.v0.3"
CHECKPOINT_SCHEMA="alice.eipm.n0.v02-query-relation-role-router-checkpoint.v0.3"

def sha(path:Path)->str:
    import hashlib
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def git_revision(root:Path)->str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()

def seed_all(seed:int)->None:
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def read_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def load_parent(graph:QueryRelationRoleRouterEvidenceGraphEncoder,parent:dict[str,torch.Tensor])->None:
    missing,unexpected=graph.load_state_dict(parent,strict=False)
    expected=sorted(graph.router_parameter_names())
    if sorted(missing)!=expected or unexpected:
        raise SystemExit(f"router parent-load drift missing={sorted(missing)} expected={expected} unexpected={unexpected}")

def configure_router_only(graph:QueryRelationRoleRouterEvidenceGraphEncoder)->list[torch.nn.Parameter]:
    allowed=set(graph.router_parameter_names())
    for name,p in graph.named_parameters():p.requires_grad=name in allowed
    observed={n for n,p in graph.named_parameters() if p.requires_grad}
    if observed!=allowed:raise SystemExit(f"router trainable-scope drift {sorted(observed)} != {sorted(allowed)}")
    return [p for p in graph.parameters() if p.requires_grad]

def assert_parent_unchanged(graph:QueryRelationRoleRouterEvidenceGraphEncoder,parent:dict[str,torch.Tensor])->None:
    state=graph.state_dict();changed=[]
    for name,t in parent.items():
        if name not in state or not torch.equal(state[name].detach().cpu(),t.detach().cpu()):changed.append(name)
    if changed:raise SystemExit("frozen parent graph mutated: "+repr(changed[:20]))

def active_edge_logits(out:dict[str,torch.Tensor],batch:dict[str,torch.Tensor])->torch.Tensor:
    logits=out["semantic_router_role_logits"];mask=batch["edge_valid_mask"]
    if int(mask.sum(dim=1).min())<1:raise RuntimeError("row without active relation edge")
    return logits[mask]

def semantic_role_targets(batch:dict[str,torch.Tensor])->torch.Tensor:
    mask=batch["edge_valid_mask"]
    if not torch.all(mask.sum(dim=1)==1):raise RuntimeError("semantic role rows require exactly one active edge")
    edge_pos=mask.float().argmax(dim=1)
    b=torch.arange(mask.size(0),device=mask.device)
    source=batch["edge_index"][b,edge_pos,0]
    target=batch["edge_index"][b,edge_pos,1]
    answer=batch["target_distribution"].argmax(dim=1)
    source_hit=answer.eq(source);target_hit=answer.eq(target)
    if not torch.all(source_hit|target_hit):raise RuntimeError("semantic target is not a relation endpoint")
    return torch.where(source_hit,torch.zeros_like(answer),torch.ones_like(answer))

def semantic_role_ce(out:dict[str,torch.Tensor],batch:dict[str,torch.Tensor])->torch.Tensor:
    return F.cross_entropy(active_edge_logits(out,batch),semantic_role_targets(batch))

def defer_ce(out:dict[str,torch.Tensor],batch:dict[str,torch.Tensor])->torch.Tensor:
    logits=active_edge_logits(out,batch)
    target=torch.full((logits.size(0),),QueryRelationRoleRouterEvidenceGraphEncoder.DEFER,device=logits.device,dtype=torch.long)
    return F.cross_entropy(logits,target)

def metrics_close(a:dict[str,Any],b:dict[str,Any],tol:float=1e-5)->bool:
    keys=["pair_accuracy","row_accuracy","family_min_pair_accuracy","mean_graph_target_margin"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys if k in a and k in b)

@torch.inference_mode()
def evaluate_router_roles(graph,adapter,dataset,payload,device,batch_size)->dict[str,Any]:
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    correct=total=0;by_family=defaultdict(lambda:[0,0]);defer=[];route=[]
    for pair_batch in loader:
        pair_batch=rr.to_device(pair_batch,device)
        a=rr.unprefix(pair_batch,"a")
        out=rr.graph_forward(graph,adapter,a)
        logits=active_edge_logits(out,a)
        target=semantic_role_targets(a)
        pred=logits.argmax(dim=-1)
        ok=pred.eq(target)
        probs=torch.softmax(logits,dim=-1)
        defer.extend(probs[:,QueryRelationRoleRouterEvidenceGraphEncoder.DEFER].detach().cpu().tolist())
        route.extend(out["semantic_router_route_probability"].detach().cpu().tolist())
        idx=pair_batch["a_global_index"].detach().cpu().tolist()
        for j,g in enumerate(idx):
            fam=str(payload["families"][int(g)])
            by_family[fam][0]+=int(ok[j].item());by_family[fam][1]+=1
        correct+=int(ok.sum().item());total+=ok.numel()
    fam_acc={k:v[0]/v[1] for k,v in sorted(by_family.items())}
    return {"pairs":total,"role_accuracy":correct/max(total,1),"family_role_accuracy":fam_acc,
            "family_min_role_accuracy":min(fam_acc.values()) if fam_acc else 0.0,
            "mean_defer_probability":sum(defer)/max(len(defer),1),
            "mean_route_probability":sum(route)/max(len(route),1)}

def ordinary_ok(cur,base,macro_tol,min_tol,top1_tol):
    return (float(cur["family_macro_target_support_mass"])>=float(base["family_macro_target_support_mass"])-macro_tol and
            float(cur["family_min_target_support_mass"])>=float(base["family_min_target_support_mass"])-min_tol and
            float(cur["family_macro_top1_support_accuracy"])>=float(base["family_macro_top1_support_accuracy"])-top1_tol)

def endpoint_ok(cur,base,pair_tol,family_tol,row_tol):
    return (float(cur["pair_accuracy"])>=float(base["pair_accuracy"])-pair_tol and
            float(cur["family_min_pair_accuracy"])>=float(base["family_min_pair_accuracy"])-family_tol and
            float(cur["row_accuracy"])>=float(base["row_accuracy"])-row_tol)

def main():
    p=argparse.ArgumentParser()
    for name in ["repo-root","prep-receipt","curriculum","manifest","semantic-config","semantic-checkpoint","tokenizer-dir",
                 "structured-config","structured-checkpoint","parent-adapter","parent-graph","ordinary-replay-cache",
                 "endpoint-replay-cache","builder","router-source","research-note","output-dir"]:
        p.add_argument("--"+name,required=True)
    p.add_argument("--max-length",type=int,default=128);p.add_argument("--encode-batch-size",type=int,default=64)
    p.add_argument("--pair-batch-size",type=int,default=16);p.add_argument("--replay-batch-size",type=int,default=32)
    p.add_argument("--eval-batch-size",type=int,default=64);p.add_argument("--max-steps",type=int,default=260)
    p.add_argument("--save-every",type=int,default=40);p.add_argument("--learning-rate",type=float,default=8e-5)
    p.add_argument("--weight-decay",type=float,default=0.02);p.add_argument("--warmup-steps",type=int,default=12)
    p.add_argument("--semantic-role-weight",type=float,default=0.50);p.add_argument("--semantic-field-weight",type=float,default=0.20)
    p.add_argument("--pair-flip-weight",type=float,default=0.10);p.add_argument("--endpoint-defer-weight",type=float,default=0.05)
    p.add_argument("--ordinary-defer-weight",type=float,default=0.05);p.add_argument("--endpoint-kl-weight",type=float,default=0.05)
    p.add_argument("--ordinary-kl-weight",type=float,default=0.05)
    p.add_argument("--ordinary-macro-tolerance",type=float,default=0.015);p.add_argument("--ordinary-min-tolerance",type=float,default=0.025)
    p.add_argument("--ordinary-top1-tolerance",type=float,default=0.03);p.add_argument("--endpoint-pair-tolerance",type=float,default=0.02)
    p.add_argument("--endpoint-family-tolerance",type=float,default=0.05);p.add_argument("--endpoint-row-tolerance",type=float,default=0.02)
    p.add_argument("--dev-role-accuracy",type=float,default=0.90);p.add_argument("--dev-family-min-role-accuracy",type=float,default=0.75)
    p.add_argument("--dev-field-pair-accuracy",type=float,default=0.80);p.add_argument("--dev-field-family-min-pair-accuracy",type=float,default=0.50)
    p.add_argument("--test-role-accuracy",type=float,default=0.92);p.add_argument("--test-family-min-role-accuracy",type=float,default=0.80)
    p.add_argument("--test-field-pair-accuracy",type=float,default=0.85);p.add_argument("--test-field-family-min-pair-accuracy",type=float,default=0.66)
    p.add_argument("--seed",type=int,default=20260918)
    a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit("query-relation role router requires CUDA")
    device=torch.device("cuda");seed_all(a.seed);root=Path(a.repo_root).resolve()
    prep_path=Path(a.prep_receipt).resolve();prep=json.loads(prep_path.read_text())
    if prep.get("schema")!=PREP_SCHEMA or prep.get("status")!="PASS_QUERY_RELATION_ROLE_ROUTER_READY_FOR_GPU":raise SystemExit("prep contract mismatch")
    if prep.get("git_revision")!=git_revision(root) or prep.get("scale_authorized") is not False:raise SystemExit("prep lineage/governance drift")
    curriculum=Path(a.curriculum).resolve();manifest_path=Path(a.manifest).resolve();manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!=MANIFEST_SCHEMA or manifest.get("compiled_sha256")!=sha(curriculum):raise SystemExit("curriculum contract drift")
    for k in ("frozen_latent_challenge_rows_used_for_training","missing_evidence_localization_rows_used_for_training",
              "relation_semantic_v0_1_rows_reused","role_residual_v0_2_rows_reused","endpoint_repair_v0_2_heldout_rows_reused"):
        if manifest.get(k) is not False:raise SystemExit("forbidden row reuse "+k)
    if manifest.get("train_dev_test_templates_lexically_disjoint") is not True:raise SystemExit("lexical split contract drift")
    paths={"curriculum":curriculum,"manifest":manifest_path,"semantic_config":Path(a.semantic_config).resolve(),
      "semantic_checkpoint":Path(a.semantic_checkpoint).resolve()/"alice_n0_v02.safetensors","tokenizer":Path(a.tokenizer_dir).resolve()/"tokenizer.json",
      "structured_config":Path(a.structured_config).resolve(),"structured_checkpoint":Path(a.structured_checkpoint).resolve()/"structured_state.safetensors",
      "parent_adapter":Path(a.parent_adapter).resolve(),"parent_graph":Path(a.parent_graph).resolve(),
      "ordinary_replay_cache":Path(a.ordinary_replay_cache).resolve(),"endpoint_replay_cache":Path(a.endpoint_replay_cache).resolve(),
      "builder":Path(a.builder).resolve(),"router_source":Path(a.router_source).resolve(),"research_note":Path(a.research_note).resolve(),
      "trainer":Path(__file__).resolve()}
    for k,path in paths.items():
        if not path.is_file() or prep["artifact_sha256"].get(k)!=sha(path):raise SystemExit(f"artifact drift {k}")
    out=Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit("refusing overwrite")
    out.mkdir(parents=True,exist_ok=True)

    sem_cfg=load_n0_config(paths["semantic_config"]);sem=AliceN0V02Model(sem_cfg)
    missing,unexpected=sem.load_state_dict(load_file(str(paths["semantic_checkpoint"]),device="cpu"),strict=False)
    if missing or unexpected:raise SystemExit("semantic parent state drift")
    for q in sem.parameters():q.requires_grad=False
    sem.to(device).eval();tokenizer=load_tokenizer(Path(a.tokenizer_dir).resolve())
    scfg=rr.load_structured_config(paths["structured_config"]);structured=StructuredStateEncoder(scfg)
    structured.load_state_dict(load_file(str(paths["structured_checkpoint"]),device="cpu"),strict=True)
    for q in structured.parameters():q.requires_grad=False
    rows=read_jsonl(curriculum)
    payload=rr.build_repair_cache(rows,semantic_model=sem,tokenizer=tokenizer,structured_parent=structured,
                                  device=device,encode_batch_size=a.encode_batch_size,max_length=a.max_length)
    torch.save(payload,out/"qrr_semantic_cache.pt");del sem,structured;torch.cuda.empty_cache()
    ordinary_payload=torch.load(paths["ordinary_replay_cache"],map_location="cpu")
    endpoint_payload=torch.load(paths["endpoint_replay_cache"],map_location="cpu")

    adapter=EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(paths["parent_adapter"]),device="cpu"),strict=True)
    for q in adapter.parameters():q.requires_grad=False
    adapter.eval()
    parent_state=load_file(str(paths["parent_graph"]),device="cpu")
    graph=QueryRelationRoleRouterEvidenceGraphEncoder(rr.expanded_graph_config()).to(device);load_parent(graph,parent_state)
    baseline=DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config()).to(device);baseline.load_state_dict(parent_state,strict=True)
    for q in baseline.parameters():q.requires_grad=False
    baseline.eval();trainable=configure_router_only(graph)

    train_pairs=rr.PairDataset(payload,rr.pair_indices(payload,"train"));dev_pairs=rr.PairDataset(payload,rr.pair_indices(payload,"dev"));test_pairs=rr.PairDataset(payload,rr.pair_indices(payload,"test"))
    ord_train=rr.RowDataset(ordinary_payload,[i for i,s in enumerate(ordinary_payload["splits"]) if s=="train"])
    ord_dev=rr.RowDataset(ordinary_payload,[i for i,s in enumerate(ordinary_payload["splits"]) if s=="dev"])
    ep_train=rr.RowDataset(endpoint_payload,[i for i,s in enumerate(endpoint_payload["splits"]) if s=="train"])
    ep_dev=rr.PairDataset(endpoint_payload,rr.pair_indices(endpoint_payload,"dev"))

    initial_field=sr.evaluate_pairs(graph,adapter,dev_pairs,payload,device,a.eval_batch_size)
    initial_role=evaluate_router_roles(graph,adapter,dev_pairs,payload,device,a.eval_batch_size)
    base_ep=sr.evaluate_pairs(baseline,adapter,ep_dev,endpoint_payload,device,a.eval_batch_size)
    init_ep=sr.evaluate_pairs(graph,adapter,ep_dev,endpoint_payload,device,a.eval_batch_size)
    base_ord=sr.evaluate_replay(baseline,adapter,ord_dev,ordinary_payload,device,a.eval_batch_size)
    init_ord=sr.evaluate_replay(graph,adapter,ord_dev,ordinary_payload,device,a.eval_batch_size)
    if not metrics_close(init_ep,base_ep,1e-4):raise SystemExit("near-defer init failed endpoint parent metric parity")
    if abs(float(init_ord["family_macro_target_support_mass"])-float(base_ord["family_macro_target_support_mass"]))>1e-4:raise SystemExit("near-defer init failed ordinary parity")
    assert_parent_unchanged(graph,parent_state)
    print("initial_field_dev="+json.dumps(initial_field,sort_keys=True),flush=True)
    print("initial_role_dev="+json.dumps(initial_role,sort_keys=True),flush=True)

    opt=torch.optim.AdamW(trainable,lr=a.learning_rate,weight_decay=a.weight_decay)
    def lr_lambda(step):
        if step<a.warmup_steps:return max((step+1)/max(a.warmup_steps,1),1e-6)
        prog=(step-a.warmup_steps)/max(a.max_steps-a.warmup_steps,1)
        return .5*(1+math.cos(math.pi*min(max(prog,0.),1.)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)
    loaders=[
      DataLoader(train_pairs,batch_size=a.pair_batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),num_workers=0),
      DataLoader(ep_train,batch_size=a.replay_batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed+1),num_workers=0),
      DataLoader(ord_train,batch_size=a.replay_batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed+2),num_workers=0)]
    its=[iter(x) for x in loaders]
    receipts={}
    for step in range(1,a.max_steps+1):
        batches=[]
        for j,loader in enumerate(loaders):
            try:b=next(its[j])
            except StopIteration:its[j]=iter(loader);b=next(its[j])
            batches.append(rr.to_device(b,device))
        pair,epb,ordb=batches;pa=rr.unprefix(pair,"a");pb=rr.unprefix(pair,"b")
        graph.train();adapter.eval();opt.zero_grad(set_to_none=True)
        oa=rr.graph_forward(graph,adapter,pa);ob=rr.graph_forward(graph,adapter,pb)
        role_loss=.5*(semantic_role_ce(oa,pa)+semantic_role_ce(ob,pb))
        field_loss=.5*(endpoint.distribution_ce(oa["field_weights"],pa["target_distribution"])+endpoint.distribution_ce(ob["field_weights"],pb["target_distribution"]))
        flip=rr.pair_flip_loss(oa,ob,pa,pb,margin=.25)
        oe=rr.graph_forward(graph,adapter,epb);oo=rr.graph_forward(graph,adapter,ordb)
        with torch.inference_mode():
            be=rr.graph_forward(baseline,adapter,epb);bo=rr.graph_forward(baseline,adapter,ordb)
        ep_defer=defer_ce(oe,epb);ord_defer=defer_ce(oo,ordb)
        ep_kl=endpoint.kl_to_baseline(oe["field_weights"],be["field_weights"],epb["valid_mask"])
        ord_kl=endpoint.kl_to_baseline(oo["field_weights"],bo["field_weights"],ordb["valid_mask"])
        total=(a.semantic_role_weight*role_loss+a.semantic_field_weight*field_loss+a.pair_flip_weight*flip+
               a.endpoint_defer_weight*ep_defer+a.ordinary_defer_weight*ord_defer+a.endpoint_kl_weight*ep_kl+a.ordinary_kl_weight*ord_kl)
        total.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step();sched.step();assert_parent_unchanged(graph,parent_state)
        if step==1 or step%20==0:
            print(json.dumps({"step":step,"loss":float(total.detach().cpu()),"role_ce":float(role_loss.detach().cpu()),
              "field_ce":float(field_loss.detach().cpu()),"pair_flip":float(flip.detach().cpu()),"endpoint_defer_ce":float(ep_defer.detach().cpu()),
              "ordinary_defer_ce":float(ord_defer.detach().cpu()),"endpoint_kl":float(ep_kl.detach().cpu()),"ordinary_kl":float(ord_kl.detach().cpu()),
              "lr":sched.get_last_lr()[0]},sort_keys=True),flush=True)
        if step%a.save_every==0 or step==a.max_steps:
            graph.eval();field_dev=sr.evaluate_pairs(graph,adapter,dev_pairs,payload,device,a.eval_batch_size)
            role_dev=evaluate_router_roles(graph,adapter,dev_pairs,payload,device,a.eval_batch_size)
            epm=sr.evaluate_pairs(graph,adapter,ep_dev,endpoint_payload,device,a.eval_batch_size)
            orm=sr.evaluate_replay(graph,adapter,ord_dev,ordinary_payload,device,a.eval_batch_size)
            cp=out/f"step-{step:08d}";cp.mkdir(parents=True,exist_ok=True);weights=cp/"evidence_graph_query_relation_role_router.safetensors"
            save_file({k:v.detach().cpu().contiguous() for k,v in graph.state_dict().items()},str(weights))
            receipt={"schema":CHECKPOINT_SCHEMA,"status":"TRAINED_NOT_RATIFIED","step":step,"graph_sha256":sha(weights),"git_revision":git_revision(root),
              "semantic_role_dev_metrics":role_dev,"semantic_field_dev_metrics":field_dev,"endpoint_role_preservation_dev_metrics":epm,
              "ordinary_replay_metrics":orm,"parent_graph_sha256":sha(paths["parent_graph"]),"parent_graph_parameters_exactly_unchanged":True,
              "trainable_scope":"raw_query_relation_role_router_side_network_only","raw_query_semantic_used_directly":True,
              "parent_query_projection_used_by_router":False,"composition":"probability_level_gated_expert_mixture",
              "hard_parameter_ceiling":None,"scale_authorized":False,"private_identity_gradient":False}
            (cp/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");receipts[cp.name]=receipt
            print("qrr_checkpoint="+json.dumps(receipt,sort_keys=True),flush=True)

    def preserve(r):
        return ordinary_ok(r["ordinary_replay_metrics"],base_ord,a.ordinary_macro_tolerance,a.ordinary_min_tolerance,a.ordinary_top1_tolerance) and endpoint_ok(r["endpoint_role_preservation_dev_metrics"],base_ep,a.endpoint_pair_tolerance,a.endpoint_family_tolerance,a.endpoint_row_tolerance)
    def ready(r):
        rm=r["semantic_role_dev_metrics"];fm=r["semantic_field_dev_metrics"]
        return (rm["role_accuracy"]>=a.dev_role_accuracy and rm["family_min_role_accuracy"]>=a.dev_family_min_role_accuracy and
                fm["pair_accuracy"]>=a.dev_field_pair_accuracy and fm["family_min_pair_accuracy"]>=a.dev_field_family_min_pair_accuracy and fm["mean_graph_target_margin"]>0)
    preserved={k:r for k,r in receipts.items() if preserve(r)};eligible={k:r for k,r in preserved.items() if ready(r)}
    def score(r):
        rm=r["semantic_role_dev_metrics"];fm=r["semantic_field_dev_metrics"]
        return (rm["family_min_role_accuracy"],rm["role_accuracy"],fm["family_min_pair_accuracy"],fm["pair_accuracy"],fm["row_accuracy"],fm["mean_graph_target_margin"],-r["step"])
    if not eligible:
        pool=preserved or receipts;best_key=max(pool,key=lambda k:score(pool[k]));best=pool[best_key]
        result={"schema":RESULT_SCHEMA,"status":"FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE",
          "git_revision":git_revision(root),"parent_graph_sha256":sha(paths["parent_graph"]),"checkpoints":receipts,
          "preservation_candidate_keys":sorted(preserved),"eligible_candidate_keys":[],"best_dev_checkpoint":best_key,
          "best_dev_checkpoint_receipt":best,"semantic_heldout_test_evaluated":False,"semantic_heldout_rows_opened_after_training":False,
          "parent_graph_parameters_exactly_unchanged":True,"trainable_scope":"raw_query_relation_role_router_side_network_only",
          "frozen_latent_challenge_rows_used_for_training":False,"missing_evidence_localization_rows_used_for_training":False,
          "relation_semantic_v0_1_rows_reused":False,"role_residual_v0_2_rows_reused":False,"endpoint_repair_v0_2_heldout_test_reused":False,
          "hard_parameter_ceiling":None,"scale_authorized":False,"automatic_hotfix_authorized":False,"n0_complete":False,
          "next_action":"architecture_level_audit_required_before_any_further_gradient_run"}
        (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return
    winner_key=max(eligible,key=lambda k:score(eligible[k]));winner=eligible[winner_key]
    graph.load_state_dict(load_file(str(out/winner_key/"evidence_graph_query_relation_role_router.safetensors"),device="cpu"),strict=True);graph.to(device).eval();assert_parent_unchanged(graph,parent_state)
    role_test=evaluate_router_roles(graph,adapter,test_pairs,payload,device,a.eval_batch_size)
    field_test=sr.evaluate_pairs(graph,adapter,test_pairs,payload,device,a.eval_batch_size)
    passed=(role_test["role_accuracy"]>=a.test_role_accuracy and role_test["family_min_role_accuracy"]>=a.test_family_min_role_accuracy and
            field_test["pair_accuracy"]>=a.test_field_pair_accuracy and field_test["family_min_pair_accuracy"]>=a.test_field_family_min_pair_accuracy and
            field_test["mean_graph_target_margin"]>0)
    result={"schema":RESULT_SCHEMA,"status":"PASS_QUERY_RELATION_ROLE_ROUTER_HELDOUT_READY_FOR_ONE_FRESH_DOWNSTREAM_CAUSAL_CHECK" if passed else "FAIL_QUERY_RELATION_ROLE_ROUTER_HELDOUT_STOP_ARCHITECTURE_AUDIT_REQUIRED",
      "git_revision":git_revision(root),"parent_graph_sha256":sha(paths["parent_graph"]),"checkpoints":receipts,"preservation_candidate_keys":sorted(preserved),
      "eligible_candidate_keys":sorted(eligible),"selected_checkpoint":winner_key,"selected_graph_sha256":winner["graph_sha256"],
      "semantic_heldout_test_evaluated":True,"semantic_heldout_rows_opened_after_training":True,"semantic_role_heldout_metrics":role_test,
      "semantic_field_heldout_metrics":field_test,"parent_graph_parameters_exactly_unchanged":True,
      "trainable_scope":"raw_query_relation_role_router_side_network_only","frozen_latent_challenge_rows_used_for_training":False,
      "missing_evidence_localization_rows_used_for_training":False,"relation_semantic_v0_1_rows_reused":False,"role_residual_v0_2_rows_reused":False,
      "endpoint_repair_v0_2_heldout_test_reused":False,"hard_parameter_ceiling":None,"scale_authorized":False,
      "automatic_hotfix_authorized":False,"n0_complete":False,
      "next_action":"run_one_fresh_downstream_causal_check" if passed else "architecture_level_audit_required_before_any_further_gradient_run"}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":main()
