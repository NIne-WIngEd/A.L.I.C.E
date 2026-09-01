from __future__ import annotations
import argparse, hashlib, json, os, re, time, urllib.request, urllib.error
from pathlib import Path
from typing import Any

MC10A_MANIFEST_SHA='C6596A225607D9CA704E1CD5D3EAE5E54A1BC7DEDEDCD34BE2CCCC373BC186D6'
ACT_MANIFEST_SHA='EC7E3C5BDAC5B1124506F34A41784477DC333D216C258E926E0BDCA1C33FDBC2'
EINF_CONTRACT_SHA='B68B04DE95E01088DD42755C0E6C1BB3FFA5C2C020354D748289E51503C45F45'
GENJUDGE_SHA='49B26305F8CC5FD607496726411C0B2EAD401F77E4D5AAE0A55296B85F75C997'
QUAL_RECEIPT_SHA='275FE5B2BBB597CBDFE5838E1B6A8E1ACB53C3BE4FED78397AFF8E577B16FFA6'
OBJ='MOST_ELAINA_ALIGNED_EVIDENCE_CONDITIONED_COMPLETION'
METHODS=['nearest_behavior_extrapolation','constraint_satisfaction','causal_analogical','counterfactual_consistency']
SEEDS=[101,202,303]
PILOT_N=5
PROMPT_VERSION='MC10B1_EINF_PORTFOLIO_PILOT_PROMPT_V1_1_4'
SCHEMA_VERSION='MC10B1_EINF_PORTFOLIO_PILOT_SCHEMA_V1_2'
SMOKE_PREDICT_BUDGETS=(256,1024,2048)
GENERATION_PREDICT_BUDGETS=(1024,2048,4096)
PARTIAL='mc10b1_einf_raw_candidates_v1.partial.jsonl'
FINAL_CAND='mc10b1_einf_raw_candidates_v1.jsonl'
CHALLENGE_PARTIAL='mc10b1_shadow_challenges_v1.partial.jsonl'
CHALLENGE_FINAL='mc10b1_shadow_challenges_v1.jsonl'
CHECKPOINT='mc10b1_generation_checkpoint_v1.json'


def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest().upper()
def sha(p:Path)->str: return sha_bytes(p.read_bytes())
def canonical(o:Any)->str: return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def rhash(o:Any)->str: return sha_bytes(canonical(o).encode('utf-8'))
def jload(p:Path): return json.loads(p.read_text(encoding='utf-8'))
def jlines(p:Path): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def req(c,m):
    if not c: raise RuntimeError(m)
def write_json(p:Path,o): p.write_text(canonical(o)+'\n',encoding='utf-8',newline='\n')
def write_jsonl(p:Path,rows): p.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
def atomic_json(p:Path,o):
    q=p.with_name(p.name+'.new'); write_json(q,o); os.replace(q,p)
def append_jsonl_fsync(p:Path,o):
    with p.open('a',encoding='utf-8',newline='\n') as f:
        f.write(canonical(o)+'\n'); f.flush(); os.fsync(f.fileno())

def check_manifest(root:Path, expected_sha:str, expected_count:int|None=None):
    m=root/'SHA256SUMS.txt'; req(m.is_file(),'missing manifest '+str(m)); req(sha(m)==expected_sha,'manifest hash '+str(m))
    exp={}
    for ln in m.read_text(encoding='utf-8').splitlines():
        if not ln.strip(): continue
        a=ln.split('  ',1); req(len(a)==2,'malformed manifest '+str(m)); exp[a[1]]=a[0].upper()
    files=[p for p in root.iterdir() if p.is_file()]
    req(set(p.name for p in files)==set(exp)|{'SHA256SUMS.txt'},'manifest file set '+str(root))
    if expected_count is not None: req(len(files)==expected_count,'package count '+str(root))
    req(not any(p.is_dir() for p in root.iterdir()),'unexpected subdir '+str(root))
    for n,h in exp.items(): req(sha(root/n)==h,'hash '+str(root/n))
    return exp

def verify_inputs(a):
    mc10a=Path(a.mc10a); act=Path(a.activation); v1=Path(a.v1); h11=Path(a.h11)
    check_manifest(mc10a,MC10A_MANIFEST_SHA,25); check_manifest(act,ACT_MANIFEST_SHA,9)
    req(sha(v1/'e_inf_candidate_generation_contract_v1.json')==EINF_CONTRACT_SHA,'EINF contract bytes')
    req(sha(h11/'candidate_generator_judge_independence_policy_v1.json')==GENJUDGE_SHA,'generator/judge policy bytes')
    ec=jload(v1/'e_inf_candidate_generation_contract_v1.json'); gp=jload(h11/'candidate_generator_judge_independence_policy_v1.json')
    req(ec['candidate_pool']['generation_modes']==METHODS,'EINF methods')
    req(ec['candidate_pool']['seeds_per_mode']==3 and ec['candidate_pool']['raw_candidates']==12 and ec['candidate_pool']['unknown_null_candidate']==1,'EINF pool')
    req(ec['candidate_pool']['minimum_distinct_non_null_candidates_after_dedup']==4,'EINF distinct floor')
    req(ec['eligibility']['minimum_independent_E0_anchor_families_high_impact']==3 and ec['eligibility']['direct_E0_anchor_required'] is True,'EINF high-impact floor')
    req(ec['eligibility']['synthetic_only_ancestry_allowed'] is False and ec['eligibility']['unknown_null_candidate_required'] is True,'EINF provenance/unknown')
    req(gp['candidate_generation']['generator_may_see_heldout_E0'] is False and gp['candidate_generation']['minimum_distinct_generation_methods']==4,'generator firewall')
    req(gp['evaluation']['judge_may_not_be_same_sampling_trace_as_generator'] is True and gp['evaluation']['single_LLM_judge_sufficient'] is False,'judge independence')
    ar=jload(act/'current_mc10_execution_authority_v1.json')
    req(ar['MC10B_activation_complete'] is True and ar['MC10B_generation_start_allowed'] is True,'activation authority')
    req(ar['E_INF_generation_enabled'] is True and ar['A_SYN_generation_enabled'] is False and ar['model_training_enabled'] is False,'scoped authority')
    s=jload(mc10a/'mc10a_summary_v1.json'); req(s['post_freeze_E_INF_eligible_packets']==65,'MC10A EINF count')
    req(s['candidate_visible_MC8_hidden_evaluator_material_loaded']==0,'hidden evaluator in MC10A')
    packets={x['packet_id']:x for x in jlines(mc10a/'mc10a_candidate_frozen_evidence_packets_v1.jsonl')}
    eq=jlines(mc10a/'mc10a_einf_generation_queue_v1.jsonl'); req(len(eq)==65,'EINF queue')
    ids=[x['packet_id'] for x in eq]; req(len(set(ids))==65 and all(i in packets for i in ids),'EINF queue ids')
    elig=[packets[i] for i in ids]
    req(all(p['post_freeze_E_INF_eligible'] is True and p['candidate_relevant_independent_E0_family_count']>=3 for p in elig),'packet eligibility')
    req(all(p['unknown_remains_competitor'] is True and p['historical_Elaina_claim_allowed'] is False for p in elig),'packet safeguards')
    unitreg={x['unit_id']:x for x in jlines(mc10a/'mc10a_frozen_unit_registry_v1.jsonl')}
    return mc10a,act,v1,h11,elig,unitreg

def verify_bound_sources(a, mc10a:Path):
    sm=jload(mc10a/'mc10a_frozen_source_manifest_v1.json'); H=sm['source_hashes']
    mapping={
      'router_v2_entries':Path(a.router)/'eipm_global_fidelity_router_v2_entries.jsonl',
      'semantic_component_masks_v2':Path(a.semantic)/'e0_semantic_component_masks_v2.jsonl',
      'semantic_repair_evidence_map_v2':Path(a.semantic)/'e0_semantic_repair_evidence_map_v2.jsonl',
      'evidence_family_registry_v1':Path(a.leak)/'evidence_family_registry_v1.jsonl',
      'current_high_impact_inventory_v3':Path(a.recon)/'mc5b_high_impact_behavioral_system_inventory_v3.jsonl',
      'MC6_core_interactions':Path(a.mc6)/'mc6b_core_interaction_probe_registry_v1.jsonl',
      'MC6_dynamic_interactions':Path(a.mc6)/'mc6e_dynamic_conditioned_four_way_interactions_v1.jsonl',
      'MC6_priority_frontier':Path(a.mc6)/'mc6g_priority_interaction_gap_frontier_v1.jsonl',
      'MC7_axis_challenges':Path(a.mc7)/'mc7b_axis_counterfactual_challenges_v1.jsonl',
      'MC7_completion_frontier':Path(a.mc7)/'mc7g_priority_gap_candidate_frontier_v1.jsonl',
      'MC8_handoff':Path(a.mc8)/'mc8g_mc9_mc10_handoff_contract_v1.json',
      'MC9_eligibility':Path(a.mc9)/'mc9e_completion_candidate_eligibility_v1.jsonl',
      'MC9_handoff':Path(a.mc9)/'mc9g_mc10_frozen_eligibility_handoff_v1.json',
      'MC9_closure':Path(a.mc9)/'mc9h_closure_receipt_v1.json',
      'owner_ratified_doctrine_summary':Path(a.doctrine)/'owner_ratified_doctrine_summary_v1.json',
      'MC10_recursive_discovery_policy_v2':Path(a.v1)/'mc10_recursive_gap_discovery_policy_v2.json',
      'EINF_generation_contract':Path(a.v1)/'e_inf_candidate_generation_contract_v1.json',
      'ASYN_generation_contract':Path(a.v1)/'a_syn_candidate_generation_contract_v1.json',
      'hardening_dual_frontier':Path(a.h11)/'dual_frontier_gap_discovery_policy_v1.json',
      'hardening_evidence_independence':Path(a.h11)/'evidence_independence_recursive_lineage_policy_v2.json',
      'hardening_pseudogap':Path(a.h11)/'heldout_pseudogap_calibration_protocol_v1.json',
      'repo_README':Path(a.repo)/'README.md',
      'repo_identity_formation_architecture':Path(a.repo)/'docs'/'MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md',
      'repo_clone_standard':Path(a.repo)/'docs'/'ALICE_CLONE_AWARE_IDENTITY_STANDARD.md',
      'repo_phase2_migration':Path(a.repo)/'docs'/'PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md',
    }
    req(set(mapping)==set(H),'frozen source manifest key set')
    for k,p in mapping.items(): req(p.is_file() and sha(p)==H[k],'bound source drift '+k)
    return {k:sha(p) for k,p in mapping.items()}

def verify_qualification_receipt(path:Path):
    req(path.is_file(),'missing generator portfolio qualification receipt'); req(sha(path)==QUAL_RECEIPT_SHA,'qualification receipt SHA mismatch')
    q=jload(path); req(q['artifact_id']=='alice.MC10B.generator-portfolio-qualification.v1','qualification receipt id')
    req(q['canonical_pool_contract']['unchanged'] is True and q['canonical_pool_contract']['raw_EINF_candidates_per_packet']==12,'qualification contract drift')
    req(q['canonical_pool_contract']['challenger_outputs_count_as_EINF_candidates'] is False and q['canonical_pool_contract']['challenger_outputs_may_be_promoted'] is False,'challenger authority drift')
    req(q['authority_invariants']['generator_is_Alice_identity_model'] is False and q['authority_invariants']['generator_has_acceptance_authority'] is False,'portfolio authority drift')
    p=q['portfolio']['primary']; cs=q['portfolio']['challengers']; r=q['portfolio']['reserve']
    req(p['tag']=='gpt-oss:20b' and len(cs)==2 and cs[0]['tag']=='gemma4:31b-it-q4_K_M' and cs[1]['tag']=='glm-4.7-flash:q4_K_M','portfolio tags')
    req(r['tag']=='qwen3.8:27b-q4_K_M' and r['execution_scope'].startswith('Zero calls'),'reserve semantics')
    return q

def local_url_ok(url:str)->bool:
    s=url.lower().rstrip('/')
    return s.startswith('http://127.0.0.1:') or s.startswith('http://localhost:') or s in ('http://127.0.0.1','http://localhost')

def http_json(url:str, method='GET', body=None, timeout=120):
    req(local_url_ok(url.split('/api/')[0] if '/api/' in url else url),'non-local generator endpoint forbidden')
    data=None; headers={}
    if body is not None:
        data=canonical(body).encode('utf-8'); headers['Content-Type']='application/json'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rq=urllib.request.Request(url,data=data,headers=headers,method=method)
    try:
        with opener.open(rq,timeout=timeout) as r: return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        b=e.read().decode('utf-8','replace'); raise RuntimeError(f'HTTP {e.code} {url}: {b[:500]}')
    except Exception as e: raise RuntimeError(f'HTTP failure {url}: {e}')

def runtime_info(base:str, spec:dict):
    base=base.rstrip('/'); req(local_url_ok(base),'Only localhost/SSH-tunneled Ollama is allowed')
    ver=http_json(base+'/api/version',timeout=20); tags=http_json(base+'/api/tags',timeout=30); ms=tags.get('models') or []
    tag=spec['tag']; m=next((x for x in ms if x.get('name')==tag or x.get('model')==tag),None); req(m is not None,'Qualified model not installed: '+tag)
    digest=m.get('digest'); req(isinstance(digest,str),'model digest unavailable '+tag); req(digest.lower()==spec['digest'].lower(),'QUALIFIED MODEL DIGEST DRIFT '+tag)
    return {'base_url':base,'ollama_version':ver.get('version'),'model_name':tag,'model_digest':digest,'model_details':m.get('details') or {},'model_size':m.get('size'),'model_modified_at':m.get('modified_at'),'qualified_profile':spec['qualified_profile'],'qualified_role':spec['role']}

def response_diag(r:dict, budget:int)->dict:
    msg=(r.get('message') or {}) if isinstance(r,dict) else {}
    content=msg.get('content') if isinstance(msg,dict) else None
    thinking=msg.get('thinking') if isinstance(msg,dict) else None
    return {
      'num_predict':budget,
      'done':r.get('done') if isinstance(r,dict) else None,
      'done_reason':r.get('done_reason') if isinstance(r,dict) else None,
      'eval_count':r.get('eval_count') if isinstance(r,dict) else None,
      'prompt_eval_count':r.get('prompt_eval_count') if isinstance(r,dict) else None,
      'content_chars':len(content) if isinstance(content,str) else 0,
      'thinking_chars':len(thinking) if isinstance(thinking,str) else 0,
    }

def smoke_model(rt:dict, timeout:int):
    sch={'type':'object','additionalProperties':False,'required':['status'],'properties':{'status':{'type':'string','enum':['ok']}}}
    schema_text=canonical(sch)
    last=None
    for budget in SMOKE_PREDICT_BUDGETS:
        body={'model':rt['model_name'],'messages':[{'role':'user','content':'Return exactly one JSON object matching this schema: '+schema_text+' Set status to ok. No markdown.'}],'stream':False,'think':rt['qualified_profile']['think'],'format':sch,'options':{'seed':7,'temperature':0,'num_predict':budget}}
        try:
            r=http_json(rt['base_url']+'/api/chat','POST',body,timeout=min(timeout,300))
            d=response_diag(r,budget); content=((r.get('message') or {}).get('content'))
            if isinstance(content,str) and content.strip():
                try:
                    x=json.loads(content)
                except Exception as e:
                    last={**d,'failure':'json_parse','detail':type(e).__name__}
                else:
                    if str(x.get('status','')).lower()=='ok': return {**d,'status':'PASS'}
                    last={**d,'failure':'status_mismatch'}
            else:
                last={**d,'failure':'empty_content'}
        except Exception as e:
            last={'num_predict':budget,'failure':'request_error','detail':str(e)[:300]}
        if budget != SMOKE_PREDICT_BUDGETS[-1]: time.sleep(1)
    raise RuntimeError('structured output smoke '+rt['model_name']+' diagnostics='+canonical(last or {'failure':'unknown'}))

def unload_model(rt:dict):
    try: http_json(rt['base_url']+'/api/generate','POST',{'model':rt['model_name'],'keep_alive':0},timeout=60)
    except Exception: pass

def select_pilot(eligible):
    def tie(p): return hashlib.sha256(('MC10B1-PILOT|'+p['packet_id']).encode()).hexdigest()
    sel=[]; asyn=[p for p in eligible if p.get('post_freeze_A_SYN_eligible')]; einf_only=[p for p in eligible if not p.get('post_freeze_A_SYN_eligible')]
    if asyn: sel.append(sorted(asyn,key=tie)[0])
    if einf_only:
        q=sorted(einf_only,key=tie)[0]
        if q['packet_id'] not in {x['packet_id'] for x in sel}: sel.append(q)
    fields=['candidate_kind','system','MC4_priority_tier','scope_factor_kind']
    while len(sel)<PILOT_N:
        seen={f:{str(x.get(f)) for x in sel} for f in fields}; seen_asyn={bool(x.get('post_freeze_A_SYN_eligible')) for x in sel}; best=None; bestkey=None
        for p in eligible:
            if p['packet_id'] in {x['packet_id'] for x in sel}: continue
            novelty=sum(str(p.get(f)) not in seen[f] for f in fields)+(bool(p.get('post_freeze_A_SYN_eligible')) not in seen_asyn)
            key=(-novelty,tie(p))
            if bestkey is None or key<bestkey: best,bestkey=p,key
        req(best is not None,'pilot selection exhausted'); sel.append(best)
    return sel

def compact_record(x):
    return {'unit_id':x['unit_id'],'family_id':x['family_id'],'leakage_cluster_id':x['leakage_cluster_id'],'router_record':x['router_record'],'semantic_mask_record':x['semantic_mask_record'],'semantic_evidence_map_record':x['semantic_evidence_map_record']}

def schema(provenance='E-INF'):
    # The wire schema carries universal bounds; the conditional UNKNOWN/non-UNKNOWN
    # minimum-length rule is duplicated in the prompt and enforced by the local validator.
    # UNKNOWN/defer is a first-class competitor and may use very short non-empty text.
    return {'type':'object','additionalProperties':False,'required':['hypothesis_type','hypothesis_text','hypothesis_probability','scope_conditions','alternative_hypotheses','counterevidence_interpretation','uncertainty_reasons','unknown_preferred','historical_claim','provenance_class'],'properties':{
        'hypothesis_type':{'type':'string','enum':['conditional_behavior_rule','microfacet_parameter','preference_prior','decision_policy','relationship_policy','emotion_regulation_policy','communication_style_rule','dynamic_update_rule','other_abstract_behavioral_hypothesis']},
        'hypothesis_text':{'type':'string','minLength':1,'maxLength':1600},
        'hypothesis_probability':{'type':'number','minimum':0,'maximum':1},
        'scope_conditions':{'type':'array','maxItems':16,'items':{'type':'string','maxLength':300}},
        'alternative_hypotheses':{'type':'array','maxItems':5,'items':{'type':'string','maxLength':600}},
        'counterevidence_interpretation':{'type':'string','maxLength':1200},
        'uncertainty_reasons':{'type':'array','maxItems':10,'items':{'type':'string','maxLength':400}},
        'unknown_preferred':{'type':'boolean'},'historical_claim':{'type':'boolean'},'provenance_class':{'type':'string','enum':[provenance]}}}

def payload_diag(x):
    if not isinstance(x,dict): return {'payload_type':type(x).__name__}
    def slen(v): return len(v) if isinstance(v,str) else None
    return {
      'unknown_preferred':x.get('unknown_preferred') if isinstance(x.get('unknown_preferred'),bool) else None,
      'hypothesis_chars':slen(x.get('hypothesis_text')),
      'scope_count':len(x.get('scope_conditions')) if isinstance(x.get('scope_conditions'),list) else None,
      'alternatives_count':len(x.get('alternative_hypotheses')) if isinstance(x.get('alternative_hypotheses'),list) else None,
      'counterevidence_chars':slen(x.get('counterevidence_interpretation')),
      'uncertainty_count':len(x.get('uncertainty_reasons')) if isinstance(x.get('uncertainty_reasons'),list) else None,
    }

def validate_payload(x, provenance='E-INF'):
    req(isinstance(x,dict),'payload not object'); req(set(x)==set(schema(provenance)['required']),'payload key set')
    req(x['provenance_class']==provenance,'payload provenance'); req(x['historical_claim'] is False,'historical claim'); req(isinstance(x['unknown_preferred'],bool),'unknown flag')
    ht=x['hypothesis_text']; req(isinstance(ht,str) and 1<=len(ht)<=1600 and (x['unknown_preferred'] is True or len(ht)>=10),'hypothesis length')
    req(isinstance(x['hypothesis_probability'],(int,float)) and 0<=x['hypothesis_probability']<=1,'probability')
    req(isinstance(x['scope_conditions'],list) and len(x['scope_conditions'])<=16 and all(isinstance(z,str) and len(z)<=300 for z in x['scope_conditions']),'scope')
    req(isinstance(x['alternative_hypotheses'],list) and len(x['alternative_hypotheses'])<=5 and all(isinstance(z,str) and len(z)<=600 for z in x['alternative_hypotheses']),'alternatives')
    req(isinstance(x['counterevidence_interpretation'],str) and len(x['counterevidence_interpretation'])<=1200,'counterevidence')
    req(isinstance(x['uncertainty_reasons'],list) and len(x['uncertainty_reasons'])<=10 and all(isinstance(z,str) and len(z)<=400 for z in x['uncertainty_reasons']),'uncertainty')
    text=' '.join([x['hypothesis_text'],x['counterevidence_interpretation']]+x['alternative_hypotheses'])
    req(not re.search(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',text),'date-like historical detail')
    req('i remember when' not in text.lower() and 'she remembers when' not in text.lower(),'memory-like historical narrative')


def parse_unique_schema_json_object(content:str, provenance:str):
    """Recover exactly one schema-valid JSON object from model content.

    Challenger models are non-authoritative and may wrap otherwise valid structured
    JSON in markdown/prose despite Ollama's schema request. We therefore ignore
    wrappers only when there is exactly one object that independently passes the
    existing strict local payload validator. Zero or multiple valid objects fail
    closed. The canonical primary E-INF generation path intentionally remains on
    direct json.loads() and is unchanged by this transport hotfix.
    """
    req(isinstance(content,str) and content.strip(),'empty model content')
    text=content.lstrip('\ufeff').strip()
    valid=[]
    # Fast path: preserve exact-object behavior.
    try:
        x=json.loads(text)
    except Exception:
        x=None
    if x is not None:
        try:
            validate_payload(x,provenance)
        except Exception:
            pass
        else:
            return x
    # Recovery path: scan for JSON object starts and accept only one schema-valid object.
    dec=json.JSONDecoder()
    for i,ch in enumerate(text):
        if ch!='{':
            continue
        try:
            obj,end=dec.raw_decode(text,i)
        except Exception:
            continue
        if not isinstance(obj,dict):
            continue
        try:
            validate_payload(obj,provenance)
        except Exception:
            continue
        valid.append(obj)
    req(len(valid)==1,'challenger JSON envelope recovery requires exactly one schema-valid object; valid_object_count='+str(len(valid)))
    return valid[0]

def method_instruction(method):
    return {
      'nearest_behavior_extrapolation':'Infer the narrowest behavioral hypothesis by extrapolating only from the nearest directly supported E0 behavior patterns. Prefer UNKNOWN when the target differs materially from observed conditions.',
      'constraint_satisfaction':'Treat supplied E0 as constraints. Propose the narrowest hypothesis that satisfies the most directly relevant constraints while preserving real tensions and counterevidence. Do not average away contradictory context-dependent behavior.',
      'causal_analogical':'Use cautious causal/analogical reasoning across the supplied independent E0 families. State only an abstract behavioral hypothesis. Do not invent an event, motive, date, place, dialogue, diagnosis, or private third-party state.',
      'counterfactual_consistency':'Propose a hypothesis that remains coherent under the target factor combination and plausible counterfactual changes. Penalize generic niceness/neutrality when unsupported; preserve evidence-supported sharpness, intensity, coldness, anger, contradiction, or warmth when present.'
    }[method]

def prompt_for(packet,unitreg,method):
    rel=[compact_record(unitreg[u]) for u in packet['candidate_relevant_E0_unit_ids']]; con=[compact_record(unitreg[u]) for u in packet['related_constraint_E0_unit_ids_no_family_floor_credit']]
    payload={'task':'Generate one raw E-INF hypothesis for a known gap in the Elaina-derived Identity Completion Graph.','provenance_rule':'E-INF is an inference/hypothesis, never historical Elaina truth.','objective':OBJ,'method':method,'method_instruction':method_instruction(method),'target':{'packet_id':packet['packet_id'],'cluster_id':packet['cluster_id'],'graph_id':packet['graph_id'],'system':packet['system'],'candidate_kind':packet['candidate_kind'],'priority':packet['MC4_priority_tier'],'target_kind':packet['target_kind'],'target_probe':packet['target_probe_snapshot'],'target_challenge':packet.get('target_challenge'),'scope_factor_kind':packet.get('scope_factor_kind')},'frozen_relevant_E0':rel,'related_E0_constraints_no_eligibility_credit':con,'response_schema':schema('E-INF'),'hard_rules':['Use only the supplied frozen evidence. Do not request or assume additional evidence.','Do not invent historical events, dates, places, dialogue, diagnoses, ethnicity, credentials, hidden intent, intimate history, or third-party private mental states.','Do not claim a synthetic scenario actually happened.','Do not optimize toward generic safety, niceness, politeness, moderation, calmness, or neutrality. Fidelity to supplied Elaina evidence is the behavioral objective.','Preserve evidence-supported contextual contradictions instead of flattening them.','UNKNOWN remains a valid competitor. Set unknown_preferred=true if the evidence does not justify a specific hypothesis. A calibrated UNKNOWN may use a short non-empty hypothesis_text; do not invent detail just to make it longer.','For a non-UNKNOWN hypothesis, hypothesis_text must be 10-1600 characters. All other array/string limits in response_schema are hard output limits.','Return only the requested JSON object. No reasoning trace.']}
    return canonical(payload)

def challenge_prompt_for(packet,unitreg,role):
    rel=[compact_record(unitreg[u]) for u in packet['candidate_relevant_E0_unit_ids']]; con=[compact_record(unitreg[u]) for u in packet['related_constraint_E0_unit_ids_no_family_floor_credit']]
    payload={'task':'Generate one independent shadow challenger hypothesis for audit. This output is NOT an E-INF candidate and can never enter the canonical 12-candidate pool.','provenance_rule':'SHADOW-CHALLENGE is an audit hypothesis, never historical Elaina truth and never promotion authority.','objective':OBJ,'challenge_role':role,'method':'independent_evidence_conditioned_challenge','target':{'packet_id':packet['packet_id'],'cluster_id':packet['cluster_id'],'graph_id':packet['graph_id'],'system':packet['system'],'candidate_kind':packet['candidate_kind'],'priority':packet['MC4_priority_tier'],'target_kind':packet['target_kind'],'target_probe':packet['target_probe_snapshot'],'target_challenge':packet.get('target_challenge'),'scope_factor_kind':packet.get('scope_factor_kind')},'frozen_relevant_E0':rel,'related_E0_constraints_no_eligibility_credit':con,'response_schema':schema('SHADOW-CHALLENGE'),'hard_rules':['Work independently. You may not inspect or imitate the primary generator outputs.','Use only the supplied frozen evidence. Do not request or assume additional evidence.','Do not invent historical events, dates, places, dialogue, diagnoses, ethnicity, credentials, hidden intent, intimate history, or third-party private mental states.','Do not claim a synthetic scenario actually happened.','Preserve evidence-supported contextual contradictions instead of flattening them.','UNKNOWN is valid when evidence is insufficient. A calibrated UNKNOWN may use a short non-empty hypothesis_text; do not invent detail just to make it longer.','For a non-UNKNOWN hypothesis, hypothesis_text must be 10-1600 characters. All other array/string limits in response_schema are hard output limits.','This output is SHADOW-CHALLENGE only and has zero acceptance, identity, training, or saturation authority.','Return only the requested JSON object. No reasoning trace.']}
    return canonical(payload)

def obligation_id(packet_id,method,seed): return sha_bytes((packet_id+'|'+method+'|'+str(seed)).encode())[:32]
def challenge_id(packet_id,model,digest,canonical_payload): return 'CHALLENGE-'+sha_bytes((packet_id+'|'+model+'|'+digest+'|'+canonical_payload).encode())[:24]
def normalized_text(s): return re.sub(r'\W+',' ',s.lower()).strip()

def generate_primary(rt,packet,unitreg,method,seed,timeout):
    pr=prompt_for(packet,unitreg,method); sch=schema('E-INF'); last=None
    for attempt,budget in enumerate(GENERATION_PREDICT_BUDGETS,1):
        d=None; x=None
        body={'model':rt['model_name'],'messages':[{'role':'system','content':'You are a non-authoritative E-INF proposal generator. Do not reveal chain-of-thought. Return only the schema-conforming final JSON candidate.'},{'role':'user','content':pr}],'stream':False,'think':rt['qualified_profile']['think'],'format':sch,'options':{'seed':seed,'temperature':0.35,'num_ctx':16384,'num_predict':budget}}
        try:
            r=http_json(rt['base_url']+'/api/chat','POST',body,timeout); d=response_diag(r,budget); content=((r.get('message') or {}).get('content'))
            req(isinstance(content,str) and content.strip(),'empty model content diagnostics='+canonical(d)); x=json.loads(content); validate_payload(x,'E-INF'); cc=canonical(x)
            return {'candidate_id':'EINF-'+sha_bytes((packet['packet_id']+'|'+method+'|'+str(seed)+'|'+cc).encode())[:24],'generation_obligation_id':obligation_id(packet['packet_id'],method,seed),'candidate_state':'RAW_UNEVALUATED','provenance_class':'E-INF','historical_Elaina_truth':False,'source_person_historical_claim_allowed':False,'packet_id':packet['packet_id'],'packet_content_sha256':packet['packet_content_sha256'],'cluster_id':packet['cluster_id'],'graph_id':packet['graph_id'],'system':packet['system'],'priority':packet['MC4_priority_tier'],'candidate_kind':packet['candidate_kind'],'generation_method':method,'seed':seed,'generator_backend':'OLLAMA_LOCALHOST_OR_SSH_TUNNEL_PROPOSAL_ONLY','generator_model':rt['model_name'],'generator_model_digest':rt['model_digest'],'generator_profile':rt['qualified_profile'],'generator_portfolio_role':'CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','generator_is_Alice_identity_model':False,'generator_has_acceptance_authority':False,'generator_may_see_hidden_E0':False,'prompt_template_version':PROMPT_VERSION,'prompt_sha256':sha_bytes(pr.encode()),'response_canonical_sha256':sha_bytes(cc.encode()),'E0_anchor_unit_ids':packet['candidate_relevant_E0_unit_ids'],'E0_anchor_family_ids':packet['candidate_relevant_E0_family_ids'],'E0_leakage_cluster_ids':packet['candidate_relevant_leakage_cluster_ids'],'related_constraint_E0_unit_ids_no_family_floor_credit':packet['related_constraint_E0_unit_ids_no_family_floor_credit'],'candidate_specific_relevant_E0_family_count':packet['candidate_relevant_independent_E0_family_count'],'unknown_remains_competitor':True,'candidate_visible_MC8_hidden_evaluator_material_loaded':0,'future_completion_objective':OBJ,'generic_safe_neutral_default_allowed':False,'payload':x,'generation_attempts':attempt,'generation_num_predict':budget,'ollama_done_reason':r.get('done_reason'),'prompt_eval_count':r.get('prompt_eval_count'),'eval_count':r.get('eval_count')}
        except Exception as e:
            diag = locals().get('d') if isinstance(locals().get('d'),dict) else {'num_predict':budget}
            px = locals().get('x')
            last = {'error':str(e)[:160], 'response':diag, 'payload':payload_diag(px)}
            if attempt < len(GENERATION_PREDICT_BUDGETS): time.sleep(attempt)
    raise RuntimeError(f'primary generation failed packet={packet["packet_id"]} method={method} seed={seed}: '+canonical(last or {'error':'unknown'}))

def generate_challenge(rt,spec,packet,unitreg,timeout):
    pr=challenge_prompt_for(packet,unitreg,spec['role']); sch=schema('SHADOW-CHALLENGE'); seed=int(spec['challenge_seed']); last=None
    for attempt,budget in enumerate(GENERATION_PREDICT_BUDGETS,1):
        d=None; x=None
        body={'model':rt['model_name'],'messages':[{'role':'system','content':'You are an independent non-authoritative shadow challenger. Do not reveal chain-of-thought. Return only the schema-conforming final JSON audit hypothesis.'},{'role':'user','content':pr}],'stream':False,'think':rt['qualified_profile']['think'],'format':sch,'options':{'seed':seed,'temperature':0.25,'num_ctx':16384,'num_predict':budget}}
        try:
            r=http_json(rt['base_url']+'/api/chat','POST',body,timeout); d=response_diag(r,budget); content=((r.get('message') or {}).get('content'))
            req(isinstance(content,str) and content.strip(),'empty challenger content diagnostics='+canonical(d)); x=parse_unique_schema_json_object(content,'SHADOW-CHALLENGE'); validate_payload(x,'SHADOW-CHALLENGE'); cc=canonical(x); cid=challenge_id(packet['packet_id'],rt['model_name'],rt['model_digest'],cc)
            return {'challenge_id':cid,'record_type':'MC10B1_SHADOW_CHALLENGE','candidate_state':'SHADOW_UNEVALUATED_NOT_EINF','provenance_class':'SHADOW-CHALLENGE','counts_as_EINF_candidate':False,'eligible_for_promotion':False,'historical_Elaina_truth':False,'packet_id':packet['packet_id'],'packet_content_sha256':packet['packet_content_sha256'],'cluster_id':packet['cluster_id'],'graph_id':packet['graph_id'],'challenge_role':spec['role'],'challenge_method':spec['challenge_method'],'seed':seed,'generator_model':rt['model_name'],'generator_model_digest':rt['model_digest'],'generator_profile':rt['qualified_profile'],'generator_is_Alice_identity_model':False,'generator_has_acceptance_authority':False,'generator_may_see_hidden_E0':False,'primary_outputs_visible_to_challenger':False,'prompt_template_version':PROMPT_VERSION,'prompt_sha256':sha_bytes(pr.encode()),'response_canonical_sha256':sha_bytes(cc.encode()),'E0_anchor_unit_ids':packet['candidate_relevant_E0_unit_ids'],'E0_anchor_family_ids':packet['candidate_relevant_E0_family_ids'],'candidate_visible_MC8_hidden_evaluator_material_loaded':0,'future_completion_objective':OBJ,'payload':x,'generation_attempts':attempt,'generation_num_predict':budget,'ollama_done_reason':r.get('done_reason'),'prompt_eval_count':r.get('prompt_eval_count'),'eval_count':r.get('eval_count')}
        except Exception as e:
            diag = locals().get('d') if isinstance(locals().get('d'),dict) else {'num_predict':budget}
            px = locals().get('x')
            last = {'error':str(e)[:160], 'response':diag, 'payload':payload_diag(px)}
            if attempt < len(GENERATION_PREDICT_BUDGETS): time.sleep(attempt)
    raise RuntimeError(f'challenger generation failed packet={packet["packet_id"]} model={rt["model_name"]}: '+canonical(last or {'error':'unknown'}))

def validate_primary_record(rec,packets,unitreg,rt):
    p=packets[rec['packet_id']]; method=rec['generation_method']; seed=rec['seed']; req(method in METHODS and seed in SEEDS,'candidate method/seed'); validate_payload(rec['payload'],'E-INF'); cc=canonical(rec['payload']); pr=prompt_for(p,unitreg,method)
    req(rec['generation_obligation_id']==obligation_id(p['packet_id'],method,seed),'obligation id'); req(rec['prompt_sha256']==sha_bytes(pr.encode()),'prompt hash'); req(rec['response_canonical_sha256']==sha_bytes(cc.encode()),'response hash'); req(rec['candidate_id']=='EINF-'+sha_bytes((p['packet_id']+'|'+method+'|'+str(seed)+'|'+cc).encode())[:24],'candidate id')
    req(rec['generator_model']==rt['model_name'] and rec['generator_model_digest']==rt['model_digest'],'generator binding'); req(rec['generator_portfolio_role']=='CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','primary role'); req(rec['E0_anchor_unit_ids']==p['candidate_relevant_E0_unit_ids'] and rec['E0_anchor_family_ids']==p['candidate_relevant_E0_family_ids'],'E0 lineage'); req(rec['candidate_specific_relevant_E0_family_count']>=3 and rec['unknown_remains_competitor'] is True,'eligibility/unknown'); req(rec['generator_is_Alice_identity_model'] is False and rec['generator_has_acceptance_authority'] is False and rec['candidate_visible_MC8_hidden_evaluator_material_loaded']==0,'authority/firewall'); req(rec['generation_num_predict'] in GENERATION_PREDICT_BUDGETS and rec['generation_attempts']==GENERATION_PREDICT_BUDGETS.index(rec['generation_num_predict'])+1,'primary generation budget provenance')

def validate_challenge_record(rec,packets,unitreg,rt,spec):
    p=packets[rec['packet_id']]; validate_payload(rec['payload'],'SHADOW-CHALLENGE'); cc=canonical(rec['payload']); pr=challenge_prompt_for(p,unitreg,spec['role']); req(rec['prompt_sha256']==sha_bytes(pr.encode()),'challenge prompt hash'); req(rec['response_canonical_sha256']==sha_bytes(cc.encode()),'challenge response hash'); req(rec['challenge_id']==challenge_id(p['packet_id'],rt['model_name'],rt['model_digest'],cc),'challenge id'); req(rec['generator_model']==rt['model_name'] and rec['generator_model_digest']==rt['model_digest'],'challenge generator'); req(rec['counts_as_EINF_candidate'] is False and rec['eligible_for_promotion'] is False and rec['primary_outputs_visible_to_challenger'] is False,'challenge isolation'); req(rec['generator_has_acceptance_authority'] is False and rec['generator_is_Alice_identity_model'] is False and rec['candidate_visible_MC8_hidden_evaluator_material_loaded']==0,'challenge authority'); req(rec['generation_num_predict'] in GENERATION_PREDICT_BUDGETS and rec['generation_attempts']==GENERATION_PREDICT_BUDGETS.index(rec['generation_num_predict'])+1,'challenge generation budget provenance')

def manifest(root:Path):
    names=sorted(p.name for p in root.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt'); (root/'SHA256SUMS.txt').write_text(''.join(f'{sha(root/n)}  {n}\n' for n in names),encoding='utf-8',newline='\n')

def validator_source():
    return r'''from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
def req(c,m):
    if not c: raise RuntimeError(m)
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def jl(p):return json.loads(p.read_text(encoding='utf-8'))
def jls(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def main(s):
    r=Path(s);req(r.is_dir(),'root');req(not any(p.is_dir() for p in r.iterdir()),'subdirs');exp={}
    for ln in (r/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if ln.strip():a=ln.split('  ',1);req(len(a)==2,'manifest');exp[a[1]]=a[0].upper()
    req(set(exp)=={p.name for p in r.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt'},'manifest names')
    for n,h in exp.items():req(sh(r/n)==h,'hash '+n)
    req(not (r/'mc10b1_generation_checkpoint_v1.json').exists(),'checkpoint leaked');req(not (r/'mc10b1_einf_raw_candidates_v1.partial.jsonl').exists(),'primary partial leaked');req(not (r/'mc10b1_shadow_challenges_v1.partial.jsonl').exists(),'challenge partial leaked')
    sel=jls(r/'mc10b1_selected_pilot_packets_v1.jsonl');c=jls(r/'mc10b1_einf_raw_candidates_v1.jsonl');u=jls(r/'mc10b1_unknown_competitors_v1.jsonl');b=jls(r/'mc10b1_blinded_future_evaluation_handoff_v1.jsonl');ch=jls(r/'mc10b1_shadow_challenges_v1.jsonl');cb=jls(r/'mc10b1_blinded_challenger_handoff_v1.jsonl');sm=jl(r/'mc10b1_summary_v1.json');z=jl(r/'mc10b1_pilot_closure_receipt_v1.json');rt=jl(r/'mc10b1_generator_runtime_manifest_v1.json');cr=jls(r/'mc10b1_challenger_runtime_manifests_v1.jsonl');pf=jl(r/'mc10b1_generator_portfolio_receipt_v1.json');cp=jl(r/'mc10b1_generation_checkpoint_receipt_v1.json')
    req(len(sel)==5 and len(c)==60 and len(u)==5 and len(b)==65 and len(ch)==10 and len(cb)==10,'counts');req(len(cr)==2,'challenger runtimes');req(pf['primary']['tag']=='gpt-oss:20b' and [x['tag'] for x in pf['challengers']]==['gemma4:31b-it-q4_K_M','glm-4.7-flash:q4_K_M'],'portfolio')
    methods={'nearest_behavior_extrapolation','constraint_satisfaction','causal_analogical','counterfactual_consistency'};seeds={101,202,303};req(len({x['candidate_id'] for x in c})==60 and len({x['generation_obligation_id'] for x in c})==60,'candidate unique')
    for p in {x['packet_id'] for x in sel}:
        q=[x for x in c if x['packet_id']==p];req(len(q)==12,'packet candidates');req({x['generation_method'] for x in q}==methods and {x['seed'] for x in q}==seeds,'pool contract');req(len({re.sub(r'\W+',' ',x['payload']['hypothesis_text'].lower()).strip() for x in q})>=4,'exact diversity');req(len([x for x in ch if x['packet_id']==p])==2,'packet challenges')
    budgets=[1024,2048,4096]
    for x in c:
        req(x['provenance_class']=='E-INF' and x['generator_portfolio_role']=='CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY' and x['generator_is_Alice_identity_model'] is False and x['generator_has_acceptance_authority'] is False and x['candidate_visible_MC8_hidden_evaluator_material_loaded']==0 and x['generation_num_predict'] in budgets and x['generation_attempts']==budgets.index(x['generation_num_predict'])+1,'candidate authority/budget')
        q=x['payload'];h=q['hypothesis_text'];req(isinstance(q.get('unknown_preferred'),bool) and isinstance(h,str) and 1<=len(h)<=1600 and (q['unknown_preferred'] is True or len(h)>=10),'candidate hypothesis length/UNKNOWN')
    for x in ch:
        req(x['provenance_class']=='SHADOW-CHALLENGE' and x['counts_as_EINF_candidate'] is False and x['eligible_for_promotion'] is False and x['primary_outputs_visible_to_challenger'] is False and x['generator_has_acceptance_authority'] is False and x['generation_num_predict'] in budgets and x['generation_attempts']==budgets.index(x['generation_num_predict'])+1,'challenge authority/budget')
        q=x['payload'];h=q['hypothesis_text'];req(isinstance(q.get('unknown_preferred'),bool) and isinstance(h,str) and 1<=len(h)<=1600 and (q['unknown_preferred'] is True or len(h)>=10),'challenge hypothesis length/UNKNOWN')
    req(rt['model_name']=='gpt-oss:20b' and rt['backend_role']=='CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','primary runtime');req(all(x['backend_role'].startswith('SHADOW_') for x in cr),'challenge runtime roles');req(all(x['generator_metadata_visible_to_judge'] is False for x in b+cb),'blinding')
    req(sm['raw_EINF_candidates_generated']==60 and sm['shadow_challenge_proposals_generated']==10 and sm['reserve_model_calls']==0 and sm['E_INF_accepted_count']==0 and sm['A_SYN_generated_count']==0 and sm['MC10_saturation_rounds_credited']==0,'summary');req(z['MC10B_pilot_audit_start_allowed'] is True and z['MC10B_full_generation_start_allowed'] is False and z['MC10C_start_allowed'] is False,'closure');req(cp['completed_primary_generation_obligations']==60 and cp['completed_shadow_challenge_obligations']==10,'checkpoint receipt')
    print('alice_mc10b1_einf_portfolio_pilot_v1_1_valid=true');print('pilot_packets=5');print('raw_EINF_candidates_generated=60');print('shadow_challenge_proposals_generated=10');print('reserve_model_calls=0');print('E_INF_accepted_count=0');print('A_SYN_generated_count=0');print('MC10_saturation_rounds_credited=0');print('MC10B_pilot_audit_start_allowed=true')
if __name__=='__main__':main(sys.argv[1])
'''

def init_or_resume(out:Path,a,source_manifest,runtimes,selected):
    state=out/CHECKPOINT; pp=out/PARTIAL; pf=out/FINAL_CAND; cp=out/CHALLENGE_PARTIAL; cf=out/CHALLENGE_FINAL
    basis={'source_manifest_hash':rhash(source_manifest),'qualification_receipt_sha256':QUAL_RECEIPT_SHA,'primary_model':runtimes['primary']['model_name'],'primary_digest':runtimes['primary']['model_digest'],'challenger_models':[r['model_name'] for r in runtimes['challengers']],'challenger_digests':[r['model_digest'] for r in runtimes['challengers']],'ollama_base_url':runtimes['primary']['base_url'],'prompt_version':PROMPT_VERSION,'schema_version':SCHEMA_VERSION,'selected_packet_ids':[p['packet_id'] for p in selected],'methods':METHODS,'seeds':SEEDS}
    if not a.resume:
        req(not out.exists(),'output exists without --resume'); out.mkdir(parents=True); write_json(out/'mc10b1_generation_source_manifest_v1.json',source_manifest); write_json(out/'mc10b1_generator_portfolio_receipt_v1.json',jload(Path(a.qualification_receipt))); write_json(out/'mc10b1_generator_runtime_manifest_v1.json',{'artifact_id':'alice.MC10B1.primary-generator-runtime.v1.1',**runtimes['primary'],'backend_role':'CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','private_evidence_transport':'LOCALHOST_OR_SSH_TUNNEL_ENDPOINT_ONLY','generator_has_acceptance_authority':False,'generator_is_Alice_identity_model':False,'A_SYN_generation_enabled':False,'model_training_enabled':False}); write_jsonl(out/'mc10b1_challenger_runtime_manifests_v1.jsonl',[{'artifact_id':'alice.MC10B1.shadow-challenger-runtime.v1.1',**r,'backend_role':spec['role'],'counts_as_EINF_candidate':False,'eligible_for_promotion':False,'generator_has_acceptance_authority':False,'generator_is_Alice_identity_model':False} for r,spec in zip(runtimes['challengers'],jload(Path(a.qualification_receipt))['portfolio']['challengers'])]); write_jsonl(out/'mc10b1_selected_pilot_packets_v1.jsonl',[{'packet_id':p['packet_id'],'cluster_id':p['cluster_id'],'graph_id':p['graph_id'],'system':p['system'],'candidate_kind':p['candidate_kind'],'priority':p['MC4_priority_tier'],'family_count':p['candidate_relevant_independent_E0_family_count'],'post_freeze_A_SYN_eligible':p['post_freeze_A_SYN_eligible'],'packet_content_sha256':p['packet_content_sha256']} for p in selected]); atomic_json(state,{'artifact_id':'alice.MC10B1.portfolio-generation-checkpoint.v1.1','basis':basis,'completed_primary_generation_obligations':0,'completed_shadow_challenge_obligations':0,'resume_count':0}); return [],[],0
    req(out.is_dir() and state.is_file(),'resume checkpoint missing'); old=jload(state); req(old['basis']==basis,'resume basis drift'); req(jload(out/'mc10b1_generation_source_manifest_v1.json')==source_manifest,'resume source drift'); primary=jlines(pf if pf.exists() else pp) if (pf.exists() or pp.exists()) else []; challenges=jlines(cf if cf.exists() else cp) if (cf.exists() or cp.exists()) else []; rc=int(old.get('resume_count',0))+1; old['resume_count']=rc; atomic_json(state,old); return primary,challenges,rc

def build(a):
    mc10a,act,v1,h11,eligible,unitreg=verify_inputs(a); bound=verify_bound_sources(a,mc10a); q=verify_qualification_receipt(Path(a.qualification_receipt)); primary_spec=q['portfolio']['primary']; challenger_specs=q['portfolio']['challengers']; rt_primary=runtime_info(a.ollama_base_url,primary_spec); rt_ch=[runtime_info(a.ollama_base_url,s) for s in challenger_specs]
    # All active models must pass exact-profile structured output before any private generation starts.
    for rt in [rt_primary]+rt_ch:
        smoke_model(rt,a.timeout_seconds); unload_model(rt)
    selected=select_pilot(eligible); req(len(selected)==PILOT_N,'pilot count'); packets={p['packet_id']:p for p in selected}
    source_manifest={'artifact_id':'alice.MC10B1.einf-portfolio-pilot.source-manifest.v1.1','MC10A_manifest_sha256':MC10A_MANIFEST_SHA,'activation_manifest_sha256':ACT_MANIFEST_SHA,'EINF_contract_sha256':EINF_CONTRACT_SHA,'generator_judge_policy_sha256':GENJUDGE_SHA,'generator_portfolio_qualification_sha256':QUAL_RECEIPT_SHA,'bound_source_hashes':bound,'selected_packet_ids':[p['packet_id'] for p in selected],'selected_packet_content_sha256':{p['packet_id']:p['packet_content_sha256'] for p in selected},'candidate_visible_MC8_hidden_evaluator_material_loaded':0,'canonical_pool_unchanged':True,'challengers_outside_canonical_pool':True,'reserve_model_calls':0}
    out=Path(a.out); runtimes={'primary':rt_primary,'challengers':rt_ch}; primary_rows,challenge_rows,resume_count=init_or_resume(out,a,source_manifest,runtimes,selected)
    seen=set()
    for rec in primary_rows: req(rec['packet_id'] in packets,'resume unknown primary packet'); validate_primary_record(rec,packets,unitreg,rt_primary); req(rec['generation_obligation_id'] not in seen,'resume duplicate primary'); seen.add(rec['generation_obligation_id'])
    challenge_seen=set()
    spec_by_tag={s['tag']:s for s in challenger_specs}; rt_by_tag={r['model_name']:r for r in rt_ch}
    for rec in challenge_rows:
        req(rec['packet_id'] in packets and rec['generator_model'] in spec_by_tag,'resume unknown challenge'); validate_challenge_record(rec,packets,unitreg,rt_by_tag[rec['generator_model']],spec_by_tag[rec['generator_model']]); key=(rec['packet_id'],rec['generator_model']); req(key not in challenge_seen,'resume duplicate challenge'); challenge_seen.add(key)
    pp=out/PARTIAL; pf=out/FINAL_CAND; cp=out/CHALLENGE_PARTIAL; cf=out/CHALLENGE_FINAL
    if not pf.exists():
        for pi,p in enumerate(selected,1):
            print(f'mc10b1_primary_packet={pi}/{PILOT_N} packet_id={p["packet_id"]}')
            for method in METHODS:
                for seed in SEEDS:
                    oid=obligation_id(p['packet_id'],method,seed)
                    if oid in seen: continue
                    rec=generate_primary(rt_primary,p,unitreg,method,seed,a.timeout_seconds); validate_primary_record(rec,packets,unitreg,rt_primary); append_jsonl_fsync(pp,rec); primary_rows.append(rec); seen.add(oid); st=jload(out/CHECKPOINT); st['completed_primary_generation_obligations']=len(primary_rows); atomic_json(out/CHECKPOINT,st); print(f'mc10b1_primary_generated packet={pi}/{PILOT_N} method={method} seed={seed} candidate_id={rec["candidate_id"]}')
        req(len(primary_rows)==60,'primary total'); os.replace(pp,pf)
    unload_model(rt_primary)
    if not cf.exists():
        for spec,rt in zip(challenger_specs,rt_ch):
            print('mc10b1_challenger_lane='+spec['role']+' model='+rt['model_name'])
            for pi,p in enumerate(selected,1):
                key=(p['packet_id'],rt['model_name'])
                if key in challenge_seen: continue
                rec=generate_challenge(rt,spec,p,unitreg,a.timeout_seconds); validate_challenge_record(rec,packets,unitreg,rt,spec); append_jsonl_fsync(cp,rec); challenge_rows.append(rec); challenge_seen.add(key); st=jload(out/CHECKPOINT); st['completed_shadow_challenge_obligations']=len(challenge_rows); atomic_json(out/CHECKPOINT,st); print(f'mc10b1_shadow_challenge_generated packet={pi}/{PILOT_N} model={rt["model_name"]} challenge_id={rec["challenge_id"]}')
            unload_model(rt)
        req(len(challenge_rows)==10,'challenge total'); os.replace(cp,cf)
    primary_rows=jlines(pf); challenge_rows=jlines(cf); req(len(primary_rows)==60 and len(challenge_rows)==10,'final counts')
    for rec in primary_rows: validate_primary_record(rec,packets,unitreg,rt_primary)
    for rec in challenge_rows: validate_challenge_record(rec,packets,unitreg,rt_by_tag[rec['generator_model']],spec_by_tag[rec['generator_model']])
    unknown=[{'candidate_id':'UNKNOWN-'+sha_bytes((p['packet_id']+'|UNKNOWN').encode())[:24],'candidate_state':'RAW_NULL_COMPETITOR','candidate_type':'UNKNOWN','packet_id':p['packet_id'],'cluster_id':p['cluster_id'],'graph_id':p['graph_id'],'provenance_class':'UNKNOWN','historical_Elaina_truth':False,'unknown_remains_competitor':True,'candidate_visible_MC8_hidden_evaluator_material_loaded':0} for p in selected]; write_jsonl(out/'mc10b1_unknown_competitors_v1.jsonl',unknown)
    blind=[{'blinded_candidate_id':'BLIND-'+sha_bytes(('MC10B1|'+r['candidate_id']).encode())[:24],'packet_id':r['packet_id'],'cluster_id':r['cluster_id'],'graph_id':r['graph_id'],'candidate_state':'RAW_UNEVALUATED','provenance_class':'E-INF','historical_Elaina_truth':False,'E0_anchor_family_ids':r['E0_anchor_family_ids'],'payload':r['payload'],'generator_metadata_visible_to_judge':False} for r in primary_rows]+[{'blinded_candidate_id':'BLIND-'+sha_bytes(('MC10B1|'+u['candidate_id']).encode())[:24],'packet_id':u['packet_id'],'cluster_id':u['cluster_id'],'graph_id':u['graph_id'],'candidate_state':'RAW_NULL_COMPETITOR','provenance_class':'UNKNOWN','historical_Elaina_truth':False,'payload':{'unknown':True},'generator_metadata_visible_to_judge':False} for u in unknown]; write_jsonl(out/'mc10b1_blinded_future_evaluation_handoff_v1.jsonl',blind)
    cblind=[{'blinded_challenge_id':'BLIND-CHALLENGE-'+sha_bytes(('MC10B1|'+r['challenge_id']).encode())[:24],'packet_id':r['packet_id'],'cluster_id':r['cluster_id'],'graph_id':r['graph_id'],'candidate_state':'SHADOW_UNEVALUATED_NOT_EINF','provenance_class':'SHADOW-CHALLENGE','payload':r['payload'],'generator_metadata_visible_to_judge':False,'counts_as_EINF_candidate':False,'eligible_for_promotion':False} for r in challenge_rows]; write_jsonl(out/'mc10b1_blinded_challenger_handoff_v1.jsonl',cblind)
    distinct={p['packet_id']:len({normalized_text(r['payload']['hypothesis_text']) for r in primary_rows if r['packet_id']==p['packet_id']}) for p in selected}; req(all(v>=4 for v in distinct.values()),'exact diversity floor'); retry_primary=sum(r['generation_attempts']-1 for r in primary_rows); retry_ch=sum(r['generation_attempts']-1 for r in challenge_rows); unknown_pref=sum(1 for r in primary_rows if r['payload']['unknown_preferred']); primary_budgets=sorted({r['generation_num_predict'] for r in primary_rows}); challenge_budgets=sorted({r['generation_num_predict'] for r in challenge_rows})
    write_json(out/'mc10b1_generation_checkpoint_receipt_v1.json',{'artifact_id':'alice.MC10B1.portfolio-generation-checkpoint-receipt.v1.1','resume_supported':True,'resume_count':resume_count,'completed_primary_generation_obligations':60,'completed_shadow_challenge_obligations':10,'partial_files_finalized':True,'checkpoint_state_removed_after_completion':True})
    summary={'artifact_id':'alice.MC10B1.einf-portfolio-pilot-summary.v1.1','pilot_packets':5,'eligible_EINF_packets_total':65,'raw_EINF_candidates_generated':60,'unknown_competitors_created':5,'shadow_challenge_proposals_generated':10,'reserve_model_calls':0,'generation_methods':4,'seeds_per_method':3,'candidate_ensemble_size_per_packet':12,'primary_model':rt_primary['model_name'],'primary_model_digest':rt_primary['model_digest'],'challenger_models':[r['model_name'] for r in rt_ch],'reserve_model':q['portfolio']['reserve']['tag'],'normalized_distinct_non_null_candidates_by_packet':distinct,'exact_normalized_diversity_failures':0,'semantic_dedup_not_yet_claimed':True,'primary_generation_retry_count':retry_primary,'challenger_generation_retry_count':retry_ch,'primary_generation_predict_budgets_used':primary_budgets,'challenger_generation_predict_budgets_used':challenge_budgets,'raw_candidates_preferring_UNKNOWN':unknown_pref,'candidate_visible_MC8_hidden_evaluator_material_loaded':0,'challenger_outputs_count_as_EINF_candidates':False,'challenger_outputs_eligible_for_promotion':False,'generator_is_Alice_identity_model':False,'generator_has_acceptance_authority':False,'A_SYN_generation_enabled':False,'autonomous_A_SYN_promotion_enabled':False,'model_training_enabled':False,'E_INF_generated_count':60,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'MC10_saturation_rounds_credited':0,'MC10B_started':True,'MC10B_complete':False,'MC10C_start_allowed':False,'stage_g_closed':False,'stage_h_activated':False,'phase2_replaced':False,'future_completion_objective':OBJ,'generic_safe_neutral_default_allowed':False,'unknown_remains_competitor':True,'pilot_only_no_full_generation_authority_inferred':True}; write_json(out/'mc10b1_summary_v1.json',summary)
    write_json(out/'mc10b1_pilot_closure_receipt_v1.json',{'artifact_id':'alice.MC10B1.einf-portfolio-pilot-closure.v1.1','pilot_generation_complete':True,'pilot_packets':5,'raw_EINF_candidates_generated':60,'shadow_challenge_proposals_generated':10,'unknown_competitors_created':5,'all_primary_candidates_raw_unevaluated':True,'all_challenges_shadow_unevaluated_not_EINF':True,'generator_judge_separation_preserved':True,'candidate_blinded_handoff_created':True,'challenger_blinded_handoff_created':True,'MC10B_pilot_audit_start_allowed':True,'MC10B_full_generation_start_allowed':False,'MC10B_complete':False,'MC10C_start_allowed':False,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_performed':False,'MC10_saturation_rounds_credited':0,'stage_g_closed':False,'phase2_replaced':False})
    (out/'ALICE_MC10B1_EINF_PORTFOLIO_PILOT_V1_1.md').write_text('# A.L.I.C.E. MC10B1 — E-INF Portfolio Pilot v1.1\n\nGPT-OSS 20B is the frozen canonical proposal generator for the ratified 12-candidate pool (4 methods × 3 seeds) on five stratified MC10A packets. Gemma 4 31B and GLM-4.7-Flash each create one independent SHADOW-CHALLENGE per packet. Shadow challenges are explicitly outside the E-INF pool and cannot be promoted. Qwen3.8 27B is recorded as reserve and receives zero calls. No output has identity, acceptance, A-SYN, training, saturation, Stage-G-closure, or Phase-2 authority.\n',encoding='utf-8',newline='\n')
    (out/'validate_alice_mc10b1_einf_portfolio_pilot_v1_1.py').write_text(validator_source(),encoding='utf-8',newline='\n'); (out/CHECKPOINT).unlink(missing_ok=True); req(not pp.exists() and not cp.exists(),'partial survived'); manifest(out)
    print('alice_mc10b1_einf_portfolio_pilot_v1_1_materialized=true');print('pilot_packets=5');print('raw_EINF_candidates_generated=60');print('shadow_challenge_proposals_generated=10');print('reserve_model_calls=0');print('primary_model='+rt_primary['model_name']);print('challengers='+','.join(r['model_name'] for r in rt_ch));print('generator_is_Alice_identity_model=false');print('generator_has_acceptance_authority=false');print('E_INF_accepted_count=0');print('A_SYN_generated_count=0');print('model_training_performed=false');print('MC10_saturation_rounds_credited=0');print('MC10B_pilot_audit_start_allowed=true');print('MC10B_full_generation_start_allowed=false');print('MC10C_start_allowed=false');print('stage_g_closed=false')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['mc10a','activation','v1','h11','router','semantic','leak','recon','mc6','mc7','mc8','mc9','doctrine','repo','out','ollama_base_url','qualification_receipt']:p.add_argument('--'+n,required=True)
    p.add_argument('--timeout_seconds',type=int,default=1800);p.add_argument('--resume',action='store_true');build(p.parse_args())
