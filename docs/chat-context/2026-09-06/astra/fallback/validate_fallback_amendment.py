"""Validate this draft's boundaries. Does not ratify it or execute a model."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'c045b20ad44ea240940c550e6fc7447888b84428293a1b9bb2abe89b68cbdc6f'
TASKS_SHA = 'b3abc2e36e4e93d3f8f6c9398c7c39b6ca2f40f9b3dcbe875ce9005d096f9420'
FALSE_AUTHORITIES = ('owner_ratified', 'independent_certification_claimed', 'v186_success_granted',
                    'qualification_granted', 'pointwise_ready_granted', 'private_pointwise_execution_granted',
                    'breadth_prerequisite_satisfied', 'A_SYN_acceptance_granted', 'A_SYN_promotion_granted',
                    'model_training_granted', 'hidden_MC8_access_granted', 'private_candidates_visible_during_qualification')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def check(proposal, source, contract):
    require(proposal['status']=='DRAFT_REQUIRES_EXPLICIT_OWNER_ACTIVATION','draft status')
    require(all(proposal[key] is False for key in FALSE_AUTHORITIES),'no authority may be manufactured')
    require(proposal['source_binding_policy_sha256']==SOURCE_SHA,'source hash')
    require(proposal['source_families']==source['required_family_set_for_this_preexecution_freeze'],'source families')
    require(proposal['proposed_successor_families']==['gemma','qwen','mistral','granite'],'only declared GLM-to-Qwen substitution')
    require(type(proposal['required_independent_judge_families']) is int and proposal['required_independent_judge_families']==source['required_independent_judge_families']==4,'four families retained')
    require(proposal['generator_family_excluded']==source['generator_family']=='gptoss','generator exclusion')
    require(proposal['new_family']=='qwen' and proposal['new_model_tag']==source['candidate_tags']['qwen'],'documented fallback identity')
    require(proposal['quantization']=='Q4_K_M' and proposal['model_digest'] is None,'draft must not invent resolved model identity')
    require(proposal['public_tasks_sha256']==TASKS_SHA,'unchanged public task authority')
    require(canonical(proposal['pass_rule'])==canonical(contract['pass_rule']),'unchanged scoring and hard anchors')
    require(proposal['calibration_only'] is True,'calibration interpretation')
    compute=proposal['compute']
    require(compute['provider']=='magnolia' and compute['accelerator']=='none' and compute['partition']=='node' and compute['qos']=='normal','declared CPU route')
    require(all(compute[key] is False for key in ('kaggle_fallback_automatic','p100_route_enabled','unauthorized_qos_allowed')),'no silent compute fallback')
    require(compute['capability_and_throughput_preflight_required'] is True and compute['runtime_full_identity_required'] is True,'runtime capability must be evidenced')
    require(compute['requested_ram_gib']==48 and compute['requested_cpus']==20 and compute['maximum_job_walltime_hours']==6,'bounded resource envelope')
    require(proposal['preserve']=={'gemma_v182_rows_reuse':True,'mistral_and_granite_bindings':True,'replacement_count':63,'deferred_count':1,'candidate_pool':287,'main':'0abaed85873c3f8de04765847eb7700b0e20433f'},'preserved state')
    profile=proposal['profile']
    require(profile=={'id':'qwen38_thinking_on_public_calibration_v1','think':True,'temperature':1.0,'top_p':0.95,'top_k':20,'min_p':0.0,'presence_penalty':0.0,'repeat_penalty':1.0,'num_ctx':8192,'num_predict':6144,'seed_formula':'9100 + one-based frozen task index','structured_output_schema_changed':False},'declared Qwen profile')
    return True


def main():
    source_bytes=(ROOT/'SOURCE_JUDGE_BINDING_POLICY.json').read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest()==SOURCE_SHA,'source file hash')
    require(hashlib.sha256((ROOT/'PUBLIC_TASKS.json').read_bytes()).hexdigest()==TASKS_SHA,'task file hash')
    source=json.loads(source_bytes)
    contract=json.loads((ROOT/'SOURCE_EFFECTIVE_CALIBRATION_CONTRACT.json').read_text())
    raw=(ROOT/'QWEN_FALLBACK_AMENDMENT_V1_DRAFT.json').read_bytes();proposal=json.loads(raw)
    check(proposal,source,contract)
    changes=[
        lambda p:p['pass_rule'].update(minimum_verdict_matches=9),
        lambda p:p['pass_rule'].update(mandatory_hard_anchors=[]),
        lambda p:p.update(required_independent_judge_families=3),
        lambda p:p.update(proposed_successor_families=['gemma','gptoss','mistral','granite']),
        lambda p:p.update(new_model_tag='qwen3.5:9b'),
        lambda p:p.update(v186_success_granted=True),
        lambda p:p.update(hidden_MC8_access_granted=True),
        lambda p:p['compute'].update(kaggle_fallback_automatic=True),
        lambda p:p['preserve'].update(candidate_pool=288),
        lambda p:p.update(owner_ratified=True),
    ]
    for mutate in changes:
        altered=deepcopy(proposal);mutate(altered)
        try:check(altered,source,contract)
        except ValueError:pass
        else:raise AssertionError('boundary mutation was not rejected')
    receipt={'artifact_id':'alice.mc10d.qwen-fallback-draft-validation.v1',
             'draft_sha256':hashlib.sha256(raw).hexdigest(),'source_binding_policy_sha256':SOURCE_SHA,
             'frozen_tasks_sha256':TASKS_SHA,'positive_draft_validation':'PASS',
             'rejected_boundary_mutations':len(changes),'owner_ratification_established':False,
             'actual_qwen_model_digest_resolved':False,'qwen_model_evaluated':False,'remote_compute_actions':0}
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    main()
