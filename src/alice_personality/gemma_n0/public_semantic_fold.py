"""One research-backed repair candidate with fresh matched TRAIN-family controls.

This uses the existing closed PUBLIC banks, never publisher/private/FINAL inputs.
TRAIN pseudo-unseen evidence is developmental, not independent qualification.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import time
import json
from . import public_feature_handoff as handoff
from . import public_feature_diagnostics as diagnostic
from . import public_semantic_experiment as producer
from . import public_candidate_only as partial

PLAN_SCHEMA = "alice-personality-public-train-family-repair-plan-v1"
SCHEMA = "alice-personality-public-train-family-repair-result-v1"
EXPORT_SCHEMA = "alice-personality-public-train-family-readout-export-v1"
FOLD_SEED = 20261003
_ROOT = Path(__file__).resolve().parents[3]
_CODE = {"public_semantic_fold.py": Path(__file__),
    "public_semantic_experiment.py": Path(producer.__file__),
    "public_feature_diagnostics.py": Path(diagnostic.__file__),
    "public_candidate_only.py": Path(partial.__file__),
    "anchored_semantic_readout.py": Path(__file__).with_name("anchored_semantic_readout.py"),
    "fit_public_semantic_fold.py": _ROOT / "scripts/eipm/gemma_n0/fit_public_semantic_fold.py",
    "magnolia_fit_public_semantic_fold.sbatch": _ROOT / "scripts/eipm/gemma_n0/magnolia_fit_public_semantic_fold.sbatch"}

class FoldError(ValueError):pass

def require(ok,reason):
    if not ok:raise FoldError(reason)

def _key(text):return sha256(handoff._canonical({"text":text,"spans":{}})).hexdigest()

def _code():return {name:handoff._hash(handoff._regular(path)) for name,path in sorted(_CODE.items())}

def build_fold(imported):
    """IDs bind pointers/splits only; no identity, exposure or family flag enters forward."""
    metadata={r["id_metadata"]:r for r in imported.plan["examples"]}
    require(len(metadata)==len(imported.plan["examples"]),"duplicate original example IDs")
    original=[r for r in imported.examples if r["split"]=="train"]
    require(len(original)>=2 and len({r["family"] for r in original})==len(original),"fold requires one original TRAIN row per family")
    bank_by_key,object_keys,family_targets={}, {}, {}
    for row in original:
        meta=metadata[row["id"]]
        require(meta["split"]=="train" and meta["family_metadata"]==row["family"]
            and meta["target_index_metadata"]==row["target"] and len(meta["descriptions"])==len(row["candidates"]),"original pointer bindings differ")
        keys=[]
        for text,bank in zip(meta["descriptions"],row["candidates"]):
            require(type(text) is str and bool(text),"candidate description must be complete text")
            key=_key(text);keys.append(key)
            require(key not in bank_by_key or bank_by_key[key] is bank,"same declared input has inconsistent bank pointers")
            require(id(bank) not in object_keys or object_keys[id(bank)]==key,"bank pointer has conflicting input declarations")
            bank_by_key[key]=bank;object_keys[id(bank)]=key
        require(len(set(keys))==len(keys),"duplicate candidate declarations")
        family_targets[row["family"]]=keys[row["target"]]
    require(len(set(family_targets.values()))==len(family_targets),"distinct TRAIN families need distinct declared descriptions")
    families=sorted(family_targets,key=lambda family:sha256(handoff._canonical({"seed":FOLD_SEED,"family":family})).hexdigest())
    count=max(1,len(families)//4)
    heldout=set(families[:count]);seen=set(families[count:])
    all_keys=sorted(family_targets.values())
    training_keys=sorted(family_targets[f] for f in seen)
    def remap(row,keys):
        target=family_targets[row["family"]]
        return {"id":row["id"],"family":row["family"],"split":"train","source":row["source"],
            "candidates":[bank_by_key[k] for k in keys],"target":keys.index(target),
            "candidate_input_sha256":list(keys),"target_input_sha256":target}
    original=sorted(original,key=lambda r:r["id"])
    training=[remap(r,training_keys) for r in original if r["family"] in seen]
    evaluation=[remap(r,all_keys) for r in original]
    protocol={"fold_seed":FOLD_SEED,"assignment":"lowest SHA256(canonical {seed,family}) quarter; minimum one",
        "train_families_metadata":sorted(seen),"heldout_families_metadata":sorted(heldout),
        "train_candidate_input_sha256":training_keys,"heldout_candidate_input_sha256":sorted(family_targets[f] for f in heldout),
        "evaluation_candidate_input_sha256":all_keys,"original_train_rows":len(original),
        "training_rows":len(training),"heldout_rows":len(heldout),
        "train_pool":"all and only TRAIN-fold positive descriptions, including all lawful distractors",
        "evaluation_pool":"all original TRAIN-family descriptions; familiar/unfamiliar membership is metadata only",
        "dev_examples_used_for_fit_or_scoring":False,"final_payload_opened":False,
        "heldout_descriptions_in_training_positive_or_negative":False,
        "one_source_per_family_limits_independent_context_claims":True}
    return {"protocol":protocol,"train":training,"evaluation":evaluation,
        "heldout":[r for r in evaluation if r["family"] in heldout],"descriptions":bank_by_key}

def _new(kind,geometry,seed,torch):
    from .semantic_readout import SemanticReadout
    from .anchored_semantic_readout import AnchoredSemanticReadout
    torch.manual_seed(seed)
    require(kind in {"fresh_original","anchored_full","anchored_candidate_only"},"unknown readout variant")
    return (SemanticReadout if kind=="fresh_original" else AnchoredSemanticReadout)(**geometry)

def load_export(path,*,binding,kind,expected_sha256,torch):
    payload=handoff._regular(path)
    require(handoff._hash(payload)==expected_sha256,"new export bytes differ")
    data=torch.load(payload,weights_only=True,map_location="cpu")
    require(type(data) is dict and set(data)=={"schema","kind","binding","geometry","model"}
        and data["schema"]==EXPORT_SCHEMA and data["kind"]==kind and data["binding"]==binding,"new export variant/binding differs")
    state=data["model"]
    require(isinstance(state,dict) and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
        and bool(torch.isfinite(v).all()) for v in state.values()),"new export weights must be finite FP32")
    model=_new(kind,data["geometry"],0,torch)
    model.load_state_dict(state,strict=True)
    return model.eval()

def fit_public_semantic_fold(package_directory,*,expected_manifest_sha256,expected_closure_sha256,
        output_directory,cpu_threads=4,allow_mechanics_only=False):
    import torch
    from .semantic_readout import fixed_semantic_scores
    require(type(cpu_threads) is int and cpu_threads==4,"four-thread fixed runtime required")
    require(type(allow_mechanics_only) is bool,"fixture scope must be explicit")
    old_rng=torch.get_rng_state().clone();old_threads=torch.get_num_threads()
    start=time.monotonic()
    try:
        torch.set_num_threads(cpu_threads)
        imported=handoff.import_public_features(package_directory,expected_manifest_sha256=expected_manifest_sha256,
            expected_closure_sha256=expected_closure_sha256,allow_mechanics_only=allow_mechanics_only)
        recipe=imported.plan["recipe"];mechanics=imported.binding["mechanics_only"]
        require(mechanics or recipe==partial.PRODUCTION_RECIPE,"original public recipe differs")
        require(mechanics or str(torch.__version__)==imported.plan["runtime"]["torch_version"],"original Torch runtime differs")
        own_code=_code();fold=build_fold(imported)
        require(mechanics or fold["protocol"]["original_train_rows"]==56,"production original TRAIN coverage differs")
        geometry={"state_count":imported.geometry["hidden_state_count"],"hidden_size":imported.geometry["hidden_size"],"width":recipe["width"]}
        null=partial.artificial_source(geometry,torch)
        banks={id(b):b for row in imported.examples for b in [row["source"],*row["candidates"],*row.get("styles",{}).values()]}
        before={str(key):diagnostic._bank_sha(bank,torch) for key,bank in banks.items()}
        output=Path(output_directory)
        require(output.is_absolute() and not output.exists() and not output.is_symlink(),"output must be fresh absolute path")
        handoff._regular(output.parent,directory=True)
        require(imported.package_directory!=output and imported.package_directory not in output.parents
            and output not in imported.package_directory.parents,"output must be outside immutable package")
        output.mkdir(mode=0o700)
        sampler=torch.Generator(device="cpu").manual_seed(recipe["seed"]+1)
        initial_sampler=diagnostic._tensor_sha(sampler.get_state(),torch)
        indices=[int(torch.randint(len(fold["train"]),(),generator=sampler)) for _ in range(recipe["steps"])]
        selected_counts={row["family"]:0 for row in fold["train"]}
        for i in indices:selected_counts[fold["train"][i]["family"]]+=1
        plan=handoff._seal({"schema":PLAN_SCHEMA,"state":"PREDECLARED_TRAIN_FAMILY_REPAIR_UNQUALIFIED",
            "import":imported.binding,"code":own_code,"recipe":recipe,"geometry":geometry,
            "fold":fold["protocol"],"ordered_training_case_ids_metadata":[r["id"] for r in fold["train"]],
            "update_indices_metadata":indices,"sampler_initial_sha256":initial_sampler,
            "sampler_final_sha256":diagnostic._tensor_sha(sampler.get_state(),torch),
            "sampled_family_positive_updates_metadata":selected_counts,
            "positive_updates":len(indices),"negative_slots":len(indices)*(len(fold["train"])-1),
            "variants":["fresh_original","anchored_full","anchored_candidate_only"],
            "candidate_only_source_artificial":True,"artificial_source_sha256":diagnostic._bank_sha(null,torch),
            "mechanics_only":mechanics,"selection_uses_dev":False,"n0_approved":False,"repair_selected":False})
        handoff._write(output/"fold-plan.json",plan)
        binding={"plan_file_sha256":handoff._hash(output/"fold-plan.json"),"plan_content_sha256":plan["receipt_sha256"],
            "sampler_execution":"immutable precomputed indices with checkpoint step cursor; no random draws during updates",
            "import":imported.binding,"code":own_code,"qualification":"UNQUALIFIED"}
        records=[];exports={};training={}
        subsets={"train_fit":[r for r in fold["evaluation"] if r["family"] in set(fold["protocol"]["train_families_metadata"])],"train_pseudo_unseen":fold["heldout"]}
        for kind in plan["variants"]:
            print(json.dumps({"stage":"matched_fold_fit_started","kind":kind,"optimizer_steps":len(indices)},sort_keys=True),flush=True)
            model=_new(kind,geometry,recipe["seed"],torch)
            initial_state=partial._state_sha(model,torch)
            optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
            def selected(row):return null if kind=="anchored_candidate_only" else row["source"]
            # Cache only the deterministic fixed path over admitted frozen banks.
            # Never cache the learned correction. Full reload checks use its real forward.
            fixed_train={r["id"]:fixed_semantic_scores(selected(r),r["candidates"]).detach().clone() for r in fold["train"]} if kind!="fresh_original" else {}
            def scores(row,network=model):
                if kind=="fresh_original":return network(selected(row),row["candidates"])
                from .semantic_readout import SemanticReadout
                return fixed_train[row["id"]]+SemanticReadout.forward(network,selected(row),row["candidates"])
            with torch.no_grad():
                initial_metrics={name:producer._evaluate(lambda s,c:model(null if kind=="anchored_candidate_only" else s,c),rows,torch) for name,rows in subsets.items()}
                if kind=="anchored_full":
                    require(all(torch.equal(model(r["source"],r["candidates"]),fixed_semantic_scores(r["source"],r["candidates"])) for r in fold["evaluation"]),"anchored initialization does not preserve fixed scores")
            run_binding={**binding,"kind":kind}
            losses=[];checkpoint_sha=None;checkpoint_path=None;checkpoints=[]
            for step,index in enumerate(indices,start=1):
                row=fold["train"][index];optimizer.zero_grad(set_to_none=True)
                logits=scores(row)
                loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([row["target"]]))
                require(bool(torch.isfinite(loss)),"matched fit loss is nonfinite")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"matched fit gradients missing/nonfinite")
                optimizer.step();losses.append(float(loss.detach()))
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"matched parameters became nonfinite")
                if step in {max(1,len(indices)//2),len(indices)-1}:
                    checkpoint_path=output/(kind+"-checkpoint-"+str(step)+".pt")
                    checkpoint_sha=producer._save_torch(checkpoint_path,producer._checkpoint(model,optimizer,sampler,step,run_binding,torch),torch)
                    checkpoints.append({"step":step,"path":str(checkpoint_path),"sha256":checkpoint_sha})
                if step in {max(1,len(indices)//2),len(indices)}:
                    print(json.dumps({"stage":"matched_fold_fit_progress","kind":kind,"completed_optimizer_steps":step},sort_keys=True),flush=True)
            final_state=partial._state_sha(model,torch)
            # Replay the final actual update from its exact typed optimizer checkpoint.
            # Midpoint artifacts remain available; no full midpoint-to-final replay claim.
            if len(indices)>1:
                clone=_new(kind,geometry,recipe["seed"],torch)
                clone_optimizer=torch.optim.AdamW(clone.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
                clone_sampler=torch.Generator(device="cpu")
                recovered=producer.restore_checkpoint(checkpoint_path,model=clone,optimizer=clone_optimizer,sampler=clone_sampler,
                    binding=run_binding,expected_file_sha256=checkpoint_sha,torch=torch)
                require(recovered==len(indices)-1,"resume step differs")
                row=fold["train"][indices[-1]];clone_optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(scores(row,clone)[None],torch.tensor([row["target"]])).backward();clone_optimizer.step()
                require(partial._state_sha(clone,torch)==final_state and producer._equal(clone_optimizer.state_dict(),optimizer.state_dict(),torch),"final update optimizer/state replay differs")
                del clone,clone_optimizer
            model.eval()
            trained=lambda s,c:model(null if kind=="anchored_candidate_only" else s,c)
            export_path=output/(kind+".pt")
            export_sha=producer._save_torch(export_path,{"schema":EXPORT_SCHEMA,"kind":kind,"binding":run_binding,"geometry":geometry,"model":model.state_dict()},torch)
            reloaded=load_export(export_path,binding=run_binding,kind=kind,expected_sha256=export_sha,torch=torch)
            reloaded_score=lambda s,c:reloaded(null if kind=="anchored_candidate_only" else s,c)
            require(partial._state_sha(reloaded,torch)==final_state,"export state reload differs")
            with torch.no_grad():
                require(all(torch.equal(trained(r["source"],r["candidates"]),reloaded_score(r["source"],r["candidates"])) for r in fold["evaluation"]),"full consumer export scores differ")
                for row in fold["evaluation"]:
                    value=diagnostic._score(trained,row["source"],row["candidates"],row["target"],torch)
                    records.append({"kind":kind,"id_metadata":row["id"],"family_metadata":row["family"],
                        "subset":"train_pseudo_unseen" if row["family"] in set(fold["protocol"]["heldout_families_metadata"]) else "train_fit",
                        "target_candidate_index_metadata":row["target"],"candidate_input_sha256":row["candidate_input_sha256"],"result":value})
            permutation=producer._permutations(trained,fold["evaluation"],torch)
            exports[kind]={"path":str(export_path),"sha256":export_sha,"schema":EXPORT_SCHEMA,"reload_exact":True,"kind":kind}
            training[kind]={"initial":initial_metrics,"final":{name:producer._evaluate(trained,rows,torch) for name,rows in subsets.items()},
                "losses":losses,"initial_state_tensor_sha256":initial_state,"final_state_tensor_sha256":final_state,
                "final_optimizer_update_replayed_exact":True,"checkpoint_sha256":checkpoint_sha,"checkpoint_path":str(checkpoint_path),
                "checkpoints":checkpoints,
                "permutation":permutation}
            del model,reloaded,optimizer
        fixed={name:producer._evaluate(fixed_semantic_scores,rows,torch) for name,rows in subsets.items()}
        require(diagnostic._bank_sha(null,torch)==plan["artificial_source_sha256"],"artificial source changed during fitting")
        require(before=={str(key):diagnostic._bank_sha(bank,torch) for key,bank in banks.items()},"frozen features changed during fitting")
        require(all(bank.layers.grad is None and not bank.layers.requires_grad for bank in [null,*banks.values()]),"upstream bank gradient boundary differs")
        fresh=handoff.import_public_features(package_directory,expected_manifest_sha256=expected_manifest_sha256,
            expected_closure_sha256=expected_closure_sha256,allow_mechanics_only=allow_mechanics_only)
        require(fresh.binding==imported.binding and handoff._canonical(fresh.plan)==handoff._canonical(imported.plan),"closed package changed during fitting")
        require(_code()==own_code and handoff._hash(output/"fold-plan.json")==binding["plan_file_sha256"],"code or predeclared plan changed")
        receipt=handoff._seal({"schema":SCHEMA,"state":"TRAIN_FAMILY_REPAIR_CANDIDATE_UNQUALIFIED","binding":binding,
            "status":"PUBLIC_MECHANICS_ONLY" if mechanics else "CLOSED_PUBLIC_TRAIN_FAMILY_FIT_COMPLETE",
            "fold":fold["protocol"],"training":training,"fixed_semantic_control":fixed,"per_example":records,"exports":exports,
            "optimizer_updates_per_variant":len(indices),"elapsed_seconds":time.monotonic()-start,"torch_version":str(torch.__version__),
            "original_features_reverified":True,"complete_upstream_banks_retained":True,"fresh_models_per_fold":True,
            "publisher_checkpoint_loaded":False,"publisher_forward_performed":False,"upstream_gradient":False,
            "private_identity_data":False,"final_payload_opened":False,"dev_used_for_fit_or_scoring":False,
            "n0_approved":False,"personality_qualified":False,"repair_selected":False,"gemma_neutrality_established":False,
            "limits":["One TRAIN-family fold and one original source per family are developmental evidence, not independent confirmation",
                "Initial semantic retention does not guarantee retention after updates or persona suppression",
                "Fresh mixed-pool/fold protocol differs from the original 576600 pools; original all-family-trained model is not a holdout comparator",
                "Full N1/N2/EIPM/N3 and voice competence remain separately built and evaluated",
                "Checkpoint proof replays only the final optimizer update, not the entire saved midpoint continuation"]})
        handoff._write(output/"fold-fit.json",receipt)
        return receipt
    finally:
        torch.set_rng_state(old_rng);torch.set_num_threads(old_threads)
