from __future__ import annotations
import hashlib, json, os, zipfile, re, statistics, difflib
from pathlib import Path
from typing import Any

FULL_VERSION = "MC10B_FULL_EINF_FRONTIER_V1_1_0"
AUDIT_ZIP_NAME = "ALICE_MC10B1_CHALLENGER_AUDIT_INPUT_mc10b1-challengers-20260830T152032Z-4870b223.zip"
AUDIT_ZIP_SHA256 = "F6E2E54157AFBFEF0B7214F1DF0C0FDA3CC698A69CCCD4C964E923D91BFD8788"
CLOSURE_NAME = "ALICE_MC10B1_PILOT_AUDIT_CLOSURE_AND_FULL_GENERATION_AUTHORIZATION_v1.json"
CLOSURE_SHA256 = "402115CAC129473EA7D74F3E56E5702BF95B1F569E4C2CAB81BF9DD5D8BE50FD"
CHALLENGER_AUDIT_MD_SHA256 = "1870FC2245E4B12F18D04D82580DDA74DC2C6C4B0BD1390141E9FE141A91FD61"
CHALLENGER_VERDICTS_SHA256 = "8C91587CA45B03F90498BE94A8EA4763085B30F11132A861C072D246FE45990F"
POST_CHALLENGER_SET_SHA256 = "D9533F5139C5573FD0DCED7BD9FE2E0A67EEC136263C01006388F9C0FBC429CA"
PILOT_PACKET_COUNT = 5
REMAINING_PACKET_COUNT = 60
CANDIDATES_PER_PACKET = 12
TOTAL_REMAINING_CANDIDATES = 720
TOTAL_FRONTIER_PACKETS = 65
TOTAL_FRONTIER_CANDIDATES = 780
TELEMETRY_BLOCK_PACKET_COUNT = 10
TELEMETRY_BLOCK_COUNT = 6
TELEMETRY_BLOCK_CANDIDATE_COUNT = 120
FINAL_OUTPUT_FILENAMES = {
    'ALICE_MC10B_FULL_EINF_FRONTIER_V1.md',
    'mc10b_full_block_telemetry_v1.jsonl',
    'mc10b_full_blinded_future_evaluation_handoff_v1.jsonl',
    'mc10b_full_einf_raw_candidates_v1.jsonl',
    'mc10b_full_frontier_coverage_receipt_v1.json',
    'mc10b_full_generation_checkpoint_v1.json',
    'mc10b_full_generation_closure_receipt_v1.json',
    'mc10b_full_generation_source_manifest_v1.json',
    'mc10b_full_generator_portfolio_receipt_v1.json',
    'mc10b_full_generator_runtime_manifest_v1.json',
    'mc10b_full_remaining_packets_v1.jsonl',
    'mc10b_full_resume_integrity_receipt_v1.json',
    'mc10b_full_summary_v1.json',
    'mc10b_full_telemetry_acknowledgements_v1.json',
    'mc10b_full_telemetry_pilot_baseline_v1.json',
    'mc10b_full_unknown_competitors_v1.jsonl',
    'validate_alice_mc10b_full_einf_frontier_v1.py',
    'SHA256SUMS.txt',
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest().upper()


def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canon(obj)+'\n', encoding='utf-8', newline='\n')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canon(r)+'\n' for r in rows), encoding='utf-8', newline='\n')


def append_jsonl_fsync(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(canon(row)+'\n')
        f.flush(); os.fsync(f.fileno())


def _safe_zip_name(name: str) -> bool:
    p=name.replace('\\','/')
    return bool(p) and not p.startswith('/') and '..' not in Path(p).parts


def verify_pilot_audit_gate(src_root: Path, builder) -> dict[str, Any]:
    gate=src_root/'pilot_audit'
    closure=gate/CLOSURE_NAME
    audit_zip=gate/AUDIT_ZIP_NAME
    audit_md=gate/'ALICE_MC10B1_CHALLENGER_FIDELITY_AUDIT_20260830.md'
    verdicts=gate/'ALICE_MC10B1_CHALLENGER_AUDIT_VERDICTS_20260830.jsonl'
    postset=gate/'ALICE_MC10B1_POST_CHALLENGER_PROVISIONAL_EINF_SET_20260830.md'
    require(closure.is_file() and sha_file(closure)==CLOSURE_SHA256, 'pilot audit closure bytes')
    require(audit_zip.is_file() and sha_file(audit_zip)==AUDIT_ZIP_SHA256, 'challenger audit input ZIP bytes')
    require(audit_md.is_file() and sha_file(audit_md)==CHALLENGER_AUDIT_MD_SHA256, 'challenger audit report bytes')
    require(verdicts.is_file() and sha_file(verdicts)==CHALLENGER_VERDICTS_SHA256, 'challenger verdict bytes')
    require(postset.is_file() and sha_file(postset)==POST_CHALLENGER_SET_SHA256, 'post-challenger set bytes')

    c=read_json(closure)
    require(c.get('artifact_id')=='alice.MC10B1.pilot-challenger-audit-closure.v1','pilot audit closure id')
    require(c.get('pilot_challenger_audit_passed') is True,'pilot challenger audit not passed')
    require(c.get('challenger_audit_status')=='PASS_WITH_FILTERING','pilot challenger audit status')
    require(c.get('owner_primary_review_status')=='APPROVED_BY_OWNER','owner primary review gate')
    require(c.get('primary_packet_decisions_overturned')==0,'pilot primary decisions overturned')
    require(c.get('challenger_outputs_audited')==10 and c.get('gemma_outputs_audited')==5 and c.get('glm_outputs_audited')==5,'challenger audit counts')
    require(c.get('MC10B_full_generation_start_allowed') is True,'full MC10B generation not authorized')
    require(c.get('next_gate')=='MC10B_FULL_CANDIDATE_GENERATION','wrong next gate')
    require(c.get('E_INF_accepted_count')==0 and c.get('A_SYN_generated_count')==0,'authority drift before full generation')
    require(c.get('model_training_performed') is False and c.get('MC10B_complete') is False and c.get('MC10C_start_allowed') is False,'downstream authority drift')
    require(c.get('stage_g_closed') is False and c.get('stage_h_activated') is False and c.get('phase2_replaced') is False,'stage authority drift')
    require(str(c.get('input_challenger_audit_zip_sha256','')).upper()==AUDIT_ZIP_SHA256,'closure/audit zip binding')

    with zipfile.ZipFile(audit_zip,'r') as z:
        names=z.namelist()
        require(all(_safe_zip_name(n) for n in names),'unsafe challenger audit ZIP path')
        required={
            'final_package/SHA256SUMS.txt',
            'final_package/mc10b1_selected_pilot_packets_v1.jsonl',
            'final_package/mc10b1_einf_raw_candidates_v1.jsonl',
            'final_package/mc10b1_shadow_challenges_v1.jsonl',
            'final_package/mc10b1_challenger_runtime_manifests_v1.jsonl',
            'final_package/mc10b1_summary_v1.json',
            'final_package/mc10b1_pilot_closure_receipt_v1.json',
        }
        require(required.issubset(set(names)), 'challenger audit ZIP missing required final-package artifacts')
        sums=z.read('final_package/SHA256SUMS.txt').decode('utf-8').splitlines()
        seen={}
        for ln in sums:
            if not ln.strip(): continue
            parts=ln.split('  ',1); require(len(parts)==2,'malformed internal SHA256SUMS')
            h,n=parts[0].upper(), parts[1]
            zp='final_package/'+n
            require(zp in names, 'internal SHA target missing '+zp)
            require(sha_bytes(z.read(zp))==h, 'internal SHA mismatch '+zp)
            seen[n]=h
        require(len(seen)==15,'expected 15 internally manifested challenger artifacts')
        selected=[json.loads(x) for x in z.read('final_package/mc10b1_selected_pilot_packets_v1.jsonl').decode('utf-8').splitlines() if x.strip()]
        challenges=[json.loads(x) for x in z.read('final_package/mc10b1_shadow_challenges_v1.jsonl').decode('utf-8').splitlines() if x.strip()]
        primary=[json.loads(x) for x in z.read('final_package/mc10b1_einf_raw_candidates_v1.jsonl').decode('utf-8').splitlines() if x.strip()]
        runtimes=[json.loads(x) for x in z.read('final_package/mc10b1_challenger_runtime_manifests_v1.jsonl').decode('utf-8').splitlines() if x.strip()]
        summary=json.loads(z.read('final_package/mc10b1_summary_v1.json').decode('utf-8'))
        pclose=json.loads(z.read('final_package/mc10b1_pilot_closure_receipt_v1.json').decode('utf-8'))
    require(len(selected)==5 and len(primary)==60 and len(challenges)==10 and len(runtimes)==2,'challenger audit final counts')
    require(summary.get('raw_EINF_candidates_generated')==60 and summary.get('shadow_challenge_proposals_generated')==10,'challenger audit summary counts')
    require(summary.get('E_INF_accepted_count')==0 and summary.get('A_SYN_generated_count')==0 and summary.get('model_training_enabled') is False,'challenger audit authority')
    require(pclose.get('MC10B_complete') is False and pclose.get('MC10C_start_allowed') is False,'pilot closure downstream authority')
    return {'closure':c,'pilot_selected_rows':selected,'pilot_primary_rows':primary,'audit_zip_sha256':AUDIT_ZIP_SHA256}


def select_remaining_packets(eligible: list[dict], builder, pilot_selected_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    require(len(eligible)==65,'expected 65 E-INF-eligible packets')
    deterministic_pilot=builder.select_pilot(eligible)
    require(len(deterministic_pilot)==5,'deterministic pilot count')
    audit_by_id={r['packet_id']:r for r in pilot_selected_rows}
    require(len(audit_by_id)==5,'audit pilot ids unique')
    det_ids=[p['packet_id'] for p in deterministic_pilot]
    require(set(det_ids)==set(audit_by_id),'audited pilot packet set differs from deterministic selection')
    for p in deterministic_pilot:
        a=audit_by_id[p['packet_id']]
        require(str(a.get('packet_content_sha256','')).upper()==str(p.get('packet_content_sha256','')).upper(), 'pilot packet content hash drift '+p['packet_id'])
    pilot_ids=set(det_ids)
    remaining=[p for p in eligible if p['packet_id'] not in pilot_ids]
    require(len(remaining)==60 and len({p['packet_id'] for p in remaining})==60,'remaining frontier count/uniqueness')
    require(set(p['packet_id'] for p in remaining).isdisjoint(pilot_ids),'pilot/remaining overlap')
    return deterministic_pilot, remaining


def source_manifest(bound: dict[str,str], pilot: list[dict], remaining: list[dict], builder) -> dict:
    return {
        'artifact_id':'alice.MC10B.full-einf-frontier.source-manifest.v1',
        'version':FULL_VERSION,
        'MC10A_manifest_sha256':builder.MC10A_MANIFEST_SHA,
        'activation_manifest_sha256':builder.ACT_MANIFEST_SHA,
        'EINF_contract_sha256':builder.EINF_CONTRACT_SHA,
        'generator_judge_policy_sha256':builder.GENJUDGE_SHA,
        'generator_portfolio_qualification_sha256':builder.QUAL_RECEIPT_SHA,
        'pilot_audit_closure_sha256':CLOSURE_SHA256,
        'pilot_challenger_audit_zip_sha256':AUDIT_ZIP_SHA256,
        'bound_source_hashes':bound,
        'pilot_packet_ids':[p['packet_id'] for p in pilot],
        'pilot_packet_content_sha256':{p['packet_id']:p['packet_content_sha256'] for p in pilot},
        'remaining_packet_ids':[p['packet_id'] for p in remaining],
        'remaining_packet_content_sha256':{p['packet_id']:p['packet_content_sha256'] for p in remaining},
        'eligible_EINF_packets_total':65,
        'pilot_packets_already_generated_and_audited':5,
        'remaining_packets_generation_authorized':60,
        'candidate_visible_MC8_hidden_evaluator_material_loaded':0,
        'canonical_pool_unchanged':True,
        'unknown_remains_competitor':True,
        'challengers_outside_canonical_pool':True,
        'reserve_model_calls':0,
        'A_SYN_generation_enabled':False,
        'model_training_enabled':False,
        'execution_backend':'KAGGLE_T4X2_PINNED_OLLAMA_RUNTIME',
    }


def synthetic_runtime(primary_spec: dict) -> dict:
    return {
        'base_url':'http://127.0.0.1:11434',
        'model_name':primary_spec['tag'],
        'model_digest':primary_spec['digest'],
        'qualified_profile':primary_spec['qualified_profile'],
    }


def validate_resume_rows(rows: list[dict], remaining: list[dict], unitreg: dict, builder, primary_spec: dict) -> list[dict]:
    packets={p['packet_id']:p for p in remaining}
    rt=synthetic_runtime(primary_spec)
    seen=set(); valid=[]
    allowed={(p['packet_id'],m,s) for p in remaining for m in builder.METHODS for s in builder.SEEDS}
    for rec in rows:
        builder.validate_primary_record(rec,packets,unitreg,rt)
        key=(rec['packet_id'],rec['generation_method'],int(rec['seed']))
        require(key in allowed,'resume row obligation not in remaining frontier')
        require(key not in seen,'duplicate resume obligation')
        seen.add(key); valid.append(rec)
    require(len(valid)<=TOTAL_REMAINING_CANDIDATES,'too many resume candidates')
    return valid





def read_recoverable_partial_jsonl(path: Path) -> tuple[list[dict], dict[str, Any]]:
    """Read a checkpoint JSONL file and recover only a torn final record.

    Appends are fsync'd, but a process or filesystem failure can still leave the last
    physical record truncated, including in the middle of a UTF-8 code point. Middle
    corruption is never tolerated. Only the final non-empty physical record may be
    discarded as an uncommitted obligation, and its exact bytes are SHA-bound in the
    private recovery receipt.
    """
    raw=path.read_bytes()
    physical=raw.splitlines()
    nonempty=[i for i,line in enumerate(physical) if line.strip()]
    last_nonempty=nonempty[-1] if nonempty else -1
    rows=[]
    recovered_tail=None
    for i,line_bytes in enumerate(physical):
        if not line_bytes.strip():
            continue
        try:
            line=line_bytes.decode('utf-8', errors='strict')
            obj=json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            require(i==last_nonempty, f'JSONL corruption before final line at physical line {i+1}')
            recovered_tail={
                'tail_recovered':True,
                'physical_line':i+1,
                'discarded_tail_sha256':sha_bytes(line_bytes),
                'discarded_tail_bytes':len(line_bytes),
                'parse_error_class':type(e).__name__,
            }
            break
        require(isinstance(obj,dict), f'JSONL row is not object at physical line {i+1}')
        rows.append(obj)
    return rows, (recovered_tail or {'tail_recovered':False})


def remaining_packet_projection(remaining: list[dict]) -> list[dict]:
    return [{
        'packet_id':p['packet_id'],'cluster_id':p['cluster_id'],'graph_id':p['graph_id'],
        'system':p['system'],'candidate_kind':p['candidate_kind'],'priority':p['MC4_priority_tier'],
        'family_count':p['candidate_relevant_independent_E0_family_count'],
        'post_freeze_A_SYN_eligible':p['post_freeze_A_SYN_eligible'],
        'packet_content_sha256':p['packet_content_sha256'],
    } for p in remaining]


def blinded_handoff_rows(rows: list[dict], unknowns: list[dict]) -> list[dict]:
    out=[]
    for r in rows:
        out.append({
            'blinded_candidate_id':'BLIND-FULL-'+sha_bytes(('MC10B-FULL|'+r['candidate_id']).encode())[:24],
            'packet_id':r['packet_id'],'cluster_id':r['cluster_id'],'graph_id':r['graph_id'],
            'candidate_state':'RAW_UNEVALUATED','provenance_class':'E-INF','historical_Elaina_truth':False,
            'E0_anchor_family_ids':r['E0_anchor_family_ids'],'payload':r['payload'],
            'generator_metadata_visible_to_judge':False,
        })
    for u in unknowns:
        out.append({
            'blinded_candidate_id':'BLIND-FULL-'+sha_bytes(('MC10B-FULL|'+u['candidate_id']).encode())[:24],
            'packet_id':u['packet_id'],'cluster_id':u['cluster_id'],'graph_id':u['graph_id'],
            'candidate_state':'RAW_NULL_COMPETITOR','provenance_class':'UNKNOWN','historical_Elaina_truth':False,
            'payload':{'unknown':True},'generator_metadata_visible_to_judge':False,
        })
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs=sorted(float(x) for x in values)
    if len(xs)==1:
        return round(xs[0],6)
    pos=(len(xs)-1)*q
    lo=int(pos); hi=min(len(xs)-1,lo+1); frac=pos-lo
    return round(xs[lo]*(1-frac)+xs[hi]*frac,6)


def _specificity_proxy(rec: dict, packet: dict) -> tuple[bool,list[str]]:
    payload=rec.get('payload') or {}
    text=str(payload.get('hypothesis_text',''))
    packet_text=canon(packet)
    numeric_re=re.compile(r'(?<![A-Za-z0-9])(?:19\d{2}|20\d{2}|\d{1,2}:\d{2}|\d+(?:\.\d+)?)(?![A-Za-z0-9])')
    generated=set(numeric_re.findall(text))
    supported=set(numeric_re.findall(packet_text))
    reasons=[]
    if generated-supported: reasons.append('new_numeric_or_time_token')
    if len(text)>700: reasons.append('long_hypothesis_text')
    if len(payload.get('scope_conditions') or [])>8: reasons.append('many_scope_conditions')
    return bool(reasons),reasons


def build_block_telemetry(block_index: int, packets: list[dict], rows: list[dict], builder, *, scope: str='FULL_FRONTIER_BLOCK') -> dict:
    require(len(packets)>0,'telemetry packets empty')
    packet_ids=[p['packet_id'] for p in packets]
    packet_set=set(packet_ids)
    br=[r for r in rows if r.get('packet_id') in packet_set]
    expected=len(packets)*CANDIDATES_PER_PACKET
    require(len(br)==expected,f'telemetry candidate count expected {expected} got {len(br)}')
    by_packet={pid:[r for r in br if r['packet_id']==pid] for pid in packet_ids}
    for pid,pr in by_packet.items():
        require(len(pr)==CANDIDATES_PER_PACKET,'telemetry packet candidate count '+pid)
        require(len({(x['generation_method'],int(x['seed'])) for x in pr})==CANDIDATES_PER_PACKET,'telemetry obligation uniqueness '+pid)
    probs=[float((r.get('payload') or {}).get('hypothesis_probability',0.0)) for r in br]
    unknown=[r for r in br if bool((r.get('payload') or {}).get('unknown_preferred'))]
    nonunknown=[r for r in br if not bool((r.get('payload') or {}).get('unknown_preferred'))]
    normalized=[builder.normalized_text((r.get('payload') or {}).get('hypothesis_text','')) for r in nonunknown]
    exact_dups=max(0,len(normalized)-len(set(normalized)))
    near_pairs=0; total_pairs=0
    for i in range(len(normalized)):
        for j in range(i+1,len(normalized)):
            total_pairs+=1
            sm=difflib.SequenceMatcher(None,normalized[i],normalized[j],autojunk=False)
            # quick_ratio() is an upper bound on ratio(). If it cannot reach the
            # registered 0.94 threshold, skip the much more expensive full diff.
            if sm.quick_ratio()>=0.94 and sm.ratio()>=0.94:
                near_pairs+=1
    packet_distinct={pid:len({builder.normalized_text((r.get('payload') or {}).get('hypothesis_text','')) for r in pr}) for pid,pr in by_packet.items()}
    packet_unknown={pid:sum(1 for r in pr if (r.get('payload') or {}).get('unknown_preferred')) for pid,pr in by_packet.items()}
    method_stats={}
    for m in builder.METHODS:
        mr=[r for r in br if r['generation_method']==m]
        mn=[r for r in mr if not (r.get('payload') or {}).get('unknown_preferred')]
        mt=[builder.normalized_text((r.get('payload') or {}).get('hypothesis_text','')) for r in mn]
        mp=[float((r.get('payload') or {}).get('hypothesis_probability',0.0)) for r in mr]
        method_stats[m]={
            'candidate_count':len(mr),'unknown_preferred_count':len(mr)-len(mn),
            'unknown_preferred_rate':round((len(mr)-len(mn))/len(mr),6) if mr else 0.0,
            'nonunknown_unique_ratio':round(len(set(mt))/len(mt),6) if mt else 1.0,
            'retry_obligation_count':sum(1 for r in mr if int(r.get('generation_attempts',1))>1),
            'probability_mean':round(statistics.mean(mp),6) if mp else 0.0,
        }
    seed_stats={}
    for seed in builder.SEEDS:
        sr=[r for r in br if int(r['seed'])==int(seed)]
        sp=[float((r.get('payload') or {}).get('hypothesis_probability',0.0)) for r in sr]
        seed_stats[str(seed)]={
            'candidate_count':len(sr),'unknown_preferred_count':sum(1 for r in sr if (r.get('payload') or {}).get('unknown_preferred')),
            'retry_obligation_count':sum(1 for r in sr if int(r.get('generation_attempts',1))>1),
            'probability_mean':round(statistics.mean(sp),6) if sp else 0.0,
        }
    packet_lookup={p['packet_id']:p for p in packets}
    specificity=[]; specificity_reason_counts={}
    for r in br:
        flagged,reasons=_specificity_proxy(r,packet_lookup[r['packet_id']])
        if flagged:
            specificity.append(r['candidate_id'])
            for reason in reasons:specificity_reason_counts[reason]=specificity_reason_counts.get(reason,0)+1
    retry_obligations=sum(1 for r in br if int(r.get('generation_attempts',1))>1)
    retry_extra=sum(max(0,int(r.get('generation_attempts',1))-1) for r in br)
    return {
        'artifact_id':'alice.MC10B.full-einf.block-telemetry.v1',
        'scope':scope,'block_index':int(block_index),'packet_count':len(packets),'candidate_count':len(br),
        'candidate_contents_in_telemetry':False,'telemetry_has_acceptance_authority':False,
        'unknown_preferred_count':len(unknown),'unknown_preferred_rate':round(len(unknown)/len(br),6),
        'retry_obligation_count':retry_obligations,'retry_obligation_rate':round(retry_obligations/len(br),6),
        'retry_extra_attempt_count':retry_extra,
        'probability':{
            'min':round(min(probs),6),'p10':_pct(probs,.10),'p25':_pct(probs,.25),'median':_pct(probs,.50),
            'mean':round(statistics.mean(probs),6),'p75':_pct(probs,.75),'p90':_pct(probs,.90),'max':round(max(probs),6),
            'low_le_0_40_count':sum(1 for x in probs if x<=0.40),'high_ge_0_85_count':sum(1 for x in probs if x>=0.85),
        },
        'exact_duplicate_nonunknown_count':exact_dups,
        'exact_duplicate_nonunknown_rate':round(exact_dups/len(normalized),6) if normalized else 0.0,
        'near_duplicate_nonunknown_pair_count':near_pairs,
        'near_duplicate_nonunknown_pair_rate':round(near_pairs/total_pairs,6) if total_pairs else 0.0,
        'packet_distinct_normalized_hypotheses':packet_distinct,
        'packet_distinct_min':min(packet_distinct.values()),'packet_distinct_median':_pct(list(packet_distinct.values()),.5),'packet_distinct_max':max(packet_distinct.values()),
        'packet_unknown_preferred_counts':packet_unknown,
        'specificity_proxy_flag_count':len(specificity),'specificity_proxy_rate':round(len(specificity)/len(br),6),
        'specificity_proxy_reason_counts':specificity_reason_counts,
        'specificity_proxy_candidate_ids_private_only':specificity,
        'method_stats':method_stats,'seed_stats':seed_stats,
        'structural_invariants_passed':all(v>=4 for v in packet_distinct.values()),
        'quality_metrics_are_diagnostic_not_truth':True,
    }


def evaluate_block_telemetry_gate(block: dict, pilot_baseline: dict) -> dict:
    reasons=[]
    structural=[]
    if int(block.get('packet_count',0))!=TELEMETRY_BLOCK_PACKET_COUNT: structural.append('packet_count')
    if int(block.get('candidate_count',0))!=TELEMETRY_BLOCK_CANDIDATE_COUNT: structural.append('candidate_count')
    if int(block.get('packet_distinct_min',0))<4: structural.append('packet_diversity_floor')
    # Conservative review thresholds. These pause generation; they do not accept/reject any candidate.
    baseline_retry=float(pilot_baseline.get('retry_obligation_rate',0.0))
    if float(block.get('retry_obligation_rate',0.0))>=max(0.55,baseline_retry+0.30): reasons.append('retry_rate_spike')
    if float(block.get('unknown_preferred_rate',0.0))>=0.60: reasons.append('unknown_rate_extreme')
    if float(block.get('exact_duplicate_nonunknown_rate',0.0))>=0.15: reasons.append('exact_template_collapse')
    if float(block.get('near_duplicate_nonunknown_pair_rate',0.0))>=0.05: reasons.append('near_template_collapse')
    if float(block.get('specificity_proxy_rate',0.0))>=0.80: reasons.append('specificity_proxy_extreme')
    bmean=float((pilot_baseline.get('probability') or {}).get('mean',0.0)); mean=float((block.get('probability') or {}).get('mean',0.0))
    if abs(mean-bmean)>=0.30: reasons.append('confidence_distribution_shift')
    for method,stats in (block.get('method_stats') or {}).items():
        if int(stats.get('candidate_count',0))!=TELEMETRY_BLOCK_PACKET_COUNT*len((101,202,303)): structural.append('method_count_'+method)
        if float(stats.get('nonunknown_unique_ratio',1.0))<0.70 and int(stats.get('candidate_count',0))-int(stats.get('unknown_preferred_count',0))>=10:
            reasons.append('method_collapse_'+method)
    return {
        'artifact_id':'alice.MC10B.full-einf.telemetry-gate.v1',
        'block_index':int(block['block_index']),'structural_failures':sorted(set(structural)),
        'review_reasons':sorted(set(reasons)),'review_required':bool(reasons),
        'gate_passed':not structural and not reasons,'telemetry_gate_has_acceptance_authority':False,
    }


def validate_resume_telemetry_state(resume_rows: list[dict], remaining: list[dict], pilot_baseline: dict, builder,
                                    prior_telemetry: list[dict], persisted_ack_blocks: set[int], requested_ack_blocks: set[int]) -> dict[str, Any]:
    prior_by_block={}
    for t in prior_telemetry:
        bi=int(t['block_index']); require(1<=bi<=TELEMETRY_BLOCK_COUNT,'prior telemetry block range')
        require(bi not in prior_by_block,'duplicate prior telemetry block')
        prior_by_block[bi]=t
    require(all(1<=x<=TELEMETRY_BLOCK_COUNT for x in persisted_ack_blocks),'persisted telemetry ack block range')
    require(all(1<=x<=TELEMETRY_BLOCK_COUNT for x in requested_ack_blocks),'requested telemetry ack block range')
    fresh_review=set(); fresh_complete=set(); fresh_by_block={}
    for bi in range(1,TELEMETRY_BLOCK_COUNT+1):
        bp=remaining[(bi-1)*TELEMETRY_BLOCK_PACKET_COUNT:bi*TELEMETRY_BLOCK_PACKET_COUNT]
        ids={p['packet_id'] for p in bp}; br=[r for r in resume_rows if r['packet_id'] in ids]
        if len(br)!=TELEMETRY_BLOCK_CANDIDATE_COUNT:
            continue
        fresh_complete.add(bi)
        fresh=build_block_telemetry(bi,bp,br,builder); gate=evaluate_block_telemetry_gate(fresh,pilot_baseline)
        fresh_by_block[bi]={'telemetry':fresh,'gate':gate}
        if gate['review_required']: fresh_review.add(bi)
        prior=prior_by_block.get(bi); require(prior is not None,'resume telemetry record missing for completed block '+str(bi))
        prior_core={k:v for k,v in prior.items() if k not in {'gate','gate_status','review_acknowledged'}}
        require(canon(prior_core)==canon(fresh),'resume telemetry metrics drift block '+str(bi))
        require(canon(prior.get('gate') or {})==canon(gate),'resume telemetry gate drift block '+str(bi))
    require(set(prior_by_block).issubset(fresh_complete),'prior telemetry exists for incomplete/nonexistent resume block')
    prior_ack_status={bi for bi,t in prior_by_block.items() if t.get('gate_status')=='ACKNOWLEDGED_REVIEW'}
    require(persisted_ack_blocks==prior_ack_status,'persisted telemetry acknowledgement/status binding')
    require(persisted_ack_blocks.issubset(fresh_review),'persisted acknowledgement no longer corresponds to a review-required block')
    require(requested_ack_blocks.issubset(fresh_review),'telemetry acknowledgement may only target a freshly recomputed review-required block')
    require(requested_ack_blocks.issubset(set(prior_by_block)),'telemetry acknowledgement requires a prior completed telemetry block')
    return {
        'acknowledged_blocks':persisted_ack_blocks|requested_ack_blocks,
        'fresh_review_blocks':fresh_review,
        'fresh_complete_blocks':fresh_complete,
        'fresh_by_block':fresh_by_block,
    }


def public_block_telemetry(block: dict, gate: dict, gate_status: str) -> dict:
    # Public-safe projection: aggregate metrics only, never packet ids, candidate ids, prompts, or hypothesis text.
    return {
        'artifact_id':'alice.MC10B.full-einf.block-telemetry-public.v1','block_index':block['block_index'],
        'packet_count':block['packet_count'],'candidate_count':block['candidate_count'],'gate_status':gate_status,
        'review_reasons':gate.get('review_reasons',[]),'unknown_preferred_rate':block['unknown_preferred_rate'],
        'retry_obligation_rate':block['retry_obligation_rate'],'probability':block['probability'],
        'exact_duplicate_nonunknown_rate':block['exact_duplicate_nonunknown_rate'],
        'near_duplicate_nonunknown_pair_rate':block['near_duplicate_nonunknown_pair_rate'],
        'packet_distinct_min':block['packet_distinct_min'],'packet_distinct_median':block['packet_distinct_median'],
        'specificity_proxy_rate':block['specificity_proxy_rate'],
        'method_stats':block['method_stats'],'seed_stats':block['seed_stats'],
        'candidate_contents_published':False,'candidate_ids_published':False,'packet_ids_published':False,
        'telemetry_has_acceptance_authority':False,
    }


def unknown_rows(remaining: list[dict]) -> list[dict]:
    return [{
        'candidate_id':'UNKNOWN-'+sha_bytes((p['packet_id']+'|UNKNOWN').encode())[:24],
        'candidate_state':'RAW_NULL_COMPETITOR','candidate_type':'UNKNOWN',
        'packet_id':p['packet_id'],'cluster_id':p['cluster_id'],'graph_id':p['graph_id'],
        'provenance_class':'UNKNOWN','historical_Elaina_truth':False,
        'unknown_remains_competitor':True,'candidate_visible_MC8_hidden_evaluator_material_loaded':0,
    } for p in remaining]


def write_manifest(root: Path) -> None:
    files=[p for p in sorted(root.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']
    (root/'SHA256SUMS.txt').write_text(''.join(f'{sha_file(p)}  {p.name}\n' for p in files),encoding='utf-8',newline='\n')


def validator_source() -> str:
    # Standalone final validator; no model inference and no hidden evidence access.
    expected=sorted(FINAL_OUTPUT_FILENAMES-{'SHA256SUMS.txt'})
    return f"""from pathlib import Path
import hashlib,json,sys
EXPECTED_FILES={expected!r}

def req(c,m):
    if not c: raise RuntimeError(m)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def jl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def j(p): return json.loads(p.read_text(encoding='utf-8'))
def main(root):
    r=Path(root); req(r.is_dir(),'root')
    actual={{p.name for p in r.iterdir() if p.is_file()}}
    req(actual==set(EXPECTED_FILES)|{{'SHA256SUMS.txt'}},'exact final file set')
    req(not any(p.is_dir() for p in r.iterdir()),'unexpected final subdirectory')
    sums={{}}
    for ln in (r/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not ln.strip():continue
        a=ln.split('  ',1);req(len(a)==2,'manifest format');req(a[1] not in sums,'manifest duplicate');sums[a[1]]=a[0].upper()
    req(set(sums)==set(EXPECTED_FILES),'manifest file set')
    for n,h in sums.items():req(sha(r/n)==h,'hash '+n)
    rows=jl(r/'mc10b_full_einf_raw_candidates_v1.jsonl'); unk=jl(r/'mc10b_full_unknown_competitors_v1.jsonl'); pk=jl(r/'mc10b_full_remaining_packets_v1.jsonl'); blind=jl(r/'mc10b_full_blinded_future_evaluation_handoff_v1.jsonl')
    s=j(r/'mc10b_full_summary_v1.json'); c=j(r/'mc10b_full_generation_closure_receipt_v1.json'); cov=j(r/'mc10b_full_frontier_coverage_receipt_v1.json'); tele=jl(r/'mc10b_full_block_telemetry_v1.jsonl'); baseline=j(r/'mc10b_full_telemetry_pilot_baseline_v1.json'); ack=j(r/'mc10b_full_telemetry_acknowledgements_v1.json'); ri=j(r/'mc10b_full_resume_integrity_receipt_v1.json'); cp=j(r/'mc10b_full_generation_checkpoint_v1.json')
    req(len(pk)==60 and len(rows)==720 and len(unk)==60 and len(blind)==780,'counts')
    packet_ids={{x['packet_id'] for x in pk}};req(len(packet_ids)==60,'packet ids')
    req(all(x.get('packet_id') in packet_ids for x in rows),'candidate packet membership')
    req({{x.get('packet_id') for x in unk}}==packet_ids,'unknown packet membership')
    req(all(sum(1 for x in rows if x['packet_id']==pid)==12 for pid in packet_ids),'twelve candidates per packet')
    req(len({{(x['packet_id'],x['generation_method'],int(x['seed'])) for x in rows}})==720,'obligation uniqueness')
    req(all(x['provenance_class']=='E-INF' and x['candidate_state']=='RAW_UNEVALUATED' and x['historical_Elaina_truth'] is False for x in rows),'raw candidate authority')
    req(all(x['provenance_class']=='UNKNOWN' and x['historical_Elaina_truth'] is False for x in unk),'unknown authority')
    req(len({{x['candidate_id'] for x in rows}})==720 and len({{x['candidate_id'] for x in unk}})==60,'candidate id uniqueness')
    req(len({{x['blinded_candidate_id'] for x in blind}})==780,'blind id uniqueness')
    req(all(x.get('packet_id') in packet_ids for x in blind),'blinded packet membership')
    req(sum(1 for x in blind if x['provenance_class']=='E-INF')==720 and sum(1 for x in blind if x['provenance_class']=='UNKNOWN')==60,'blind provenance counts')
    req(s['raw_EINF_candidates_generated_this_stage']==720 and s['E_INF_accepted_count']==0 and s['A_SYN_generated_count']==0 and s['model_training_enabled'] is False,'summary authority')
    req(c['MC10B_full_EINF_frontier_generation_complete'] is True and c['broader_canon_falsification_required_before_acceptance'] is True and c['MC10C_start_allowed'] is False,'closure gate')
    req(cov['eligible_EINF_packets_total']==65 and cov['pilot_packets_generated_and_audited']==5 and cov['remaining_packets_generated']==60 and cov['total_raw_EINF_candidates_across_pilot_and_full_generation']==780,'coverage')
    req(len(tele)==6 and [int(x['block_index']) for x in tele]==[1,2,3,4,5,6],'telemetry block count/order')
    req(all(int(x['packet_count'])==10 and int(x['candidate_count'])==120 and x['gate_status'] in {{'PASS','ACKNOWLEDGED_REVIEW'}} for x in tele),'telemetry gate resolution')
    req(all(x.get('candidate_contents_in_telemetry') is False and x.get('telemetry_has_acceptance_authority') is False for x in tele),'telemetry authority')
    req(baseline.get('scope')=='AUDITED_PILOT_BASELINE' and int(baseline.get('candidate_count',0))==60,'pilot telemetry baseline')
    req(ack.get('artifact_id')=='alice.MC10B.full-einf.telemetry-acknowledgements.v1','telemetry ack artifact')
    expected_ack=sorted(int(x['block_index']) for x in tele if x['gate_status']=='ACKNOWLEDGED_REVIEW'); req(sorted(int(x) for x in ack.get('acknowledged_review_blocks',[]))==expected_ack,'telemetry acknowledgement binding')
    req(s.get('telemetry_blocks_completed')==6 and s.get('telemetry_all_gates_resolved') is True,'summary telemetry closure'); req(c.get('telemetry_blocks_completed')==6 and c.get('telemetry_all_gates_resolved') is True,'closure telemetry gate')
    req(ri.get('artifact_id')=='alice.MC10B.full-einf.resume-integrity.v1' and isinstance(ri.get('tail_recovered'),bool),'resume integrity receipt')
    req(int(cp.get('completed_candidate_obligations',-1))==720 and int(cp.get('remaining_candidate_obligations',-1))==0 and cp.get('generation_soft_stop_reached') is False,'final checkpoint completion')
    print('MC10B_FULL_EINF_FRONTIER_VALID=true')
if __name__=='__main__':main(sys.argv[1])
"""
