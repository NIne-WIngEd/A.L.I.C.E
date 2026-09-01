from __future__ import annotations
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
