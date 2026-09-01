from __future__ import annotations
import argparse, importlib.util, json, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest().upper()

def require(c,m):
    if not c: raise RuntimeError(m)

def load_exact(name, expected, module_name):
    xs=sorted(p for p in Path('/kaggle/input').rglob(name) if p.is_file())
    require(len(xs)==1,f'expected exactly one {name}, found {len(xs)}')
    require(sha256_file(xs[0])==expected.upper(),f'{name} SHA mismatch')
    spec=importlib.util.spec_from_file_location(module_name,xs[0]);require(spec and spec.loader,f'cannot load {name}')
    m=importlib.util.module_from_spec(spec);sys.modules[module_name]=m;spec.loader.exec_module(m);return m

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write_jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for x in rows),encoding='utf-8',newline='\n')


def safe_remove_model(base, binary, env, tag, live):
    # Model cleanup is best-effort only. A cleanup warning must never convert a
    # successfully persisted checkpoint/final package into a failed generation run.
    try:
        base.remove_model(binary, env, tag, live)
    except Exception as e:
        try:
            live.event('MODEL_CLEANUP_WARNING', {'model':tag,'cleanup_failed':True,'private_candidate_content_published':False})
        except Exception:
            pass

def main():
    cfgs=sorted(p for p in Path('/kaggle/input').rglob('mc10b-full-run-config.json') if p.is_file())
    require(len(cfgs)==1,f'expected one full run config, found {len(cfgs)}')
    cfg=read_json(cfgs[0])
    require(cfg.get('artifact_id')=='alice.MC10B.full-einf-generation.run-config.v1.1.0','run config artifact id')
    require(str(cfg.get('controller_version'))=='1.1.0','controller version mismatch')
    run_id=str(cfg['run_id']);branch=str(cfg['github_branch'])
    base=load_exact('mc10b1_kaggle_worker.py',str(cfg['base_worker_sha256']),'mc10b1_worker_base')
    b=load_exact('build_mc10b1_portfolio_pilot_v1_1.py',str(cfg['builder_sha256']),'mc10b1_builder_base')
    tc=load_exact('mc10b1_transport_common.py',str(cfg['transport_common_sha256']),'mc10b1_transport_common_runtime')
    fc=load_exact('mc10b_full_common.py',str(cfg['full_common_sha256']),'mc10b_full_common_runtime')
    require(str(cfg.get('private_blob_name'))==str(tc.PRIVATE_BLOB_NAME),'private blob contract drift')
    require(int(cfg.get('expected_eligible_packets',-1))==65,'expected eligible packet config')
    require(int(cfg.get('expected_pilot_packets',-1))==5,'expected pilot packet config')
    require(int(cfg.get('expected_remaining_packets',-1))==60,'expected remaining packet config')
    require(int(cfg.get('expected_candidate_obligations',-1))==720,'expected candidate obligation config')
    base.b=b;base.tc=tc
    # Preserve the proven telemetry privacy firewall; only change stage/count labels.
    class FullLive(base.GitHubLive):
        def __init__(self, token, repo, branch, run_id):
            super().__init__(token,repo,branch,run_id,'full-einf-generation')
            self.current.update({
                'artifact_id':'alice.MC10B.full-einf-generation.live.current.v1',
                'packet_total':60,'primary_candidates_generated':0,'shadow_challenges_generated':0,
                'pilot_packets_already_audited':5,'remaining_packets_total':60,
                'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_enabled':False,'controller_version':'1.1.0',
            })
        def publish_current(self):
            with self.lock:
                self.current['updated_at_utc']=base.utcnow();snapshot=dict(self.current)
            self.put_json('mc10b/full/current.json',snapshot,f'mc10b-full: {self.run_id} heartbeat');self._telemetry_success()
        def event(self,kind,payload):
            import hashlib
            # Public event telemetry is allow-redacted. Base MC10B1 helpers sometimes
            # include transient executable/input paths or raw warning strings; those are
            # useful privately but must never land on the public live branch.
            blocked={'path','argv0','arg1','warning','failure_message','message','traceback','token'}
            clean={}
            for k,v in dict(payload).items():
                key=str(k)
                if key in blocked or key.endswith('_path') or 'token' in key.lower():
                    continue
                if isinstance(v,str) and ('/kaggle/' in v or '/tmp/' in v or ':\\' in v):
                    continue
                clean[key]=v
            event={'run_id':self.run_id,'stage':self.stage,'kind':kind,'at_utc':base.utcnow(),**clean}
            token=hashlib.sha256((base.canon(event)+str(time.time_ns())).encode()).hexdigest()[:12]
            self.put_json(f'mc10b/full/runs/{self.run_id}/events/{int(time.time()*1000)}-{token}.json',event,f'mc10b-full: {self.run_id} {kind}')
        def observation(self,relpath,obj):
            self.put_json(f'mc10b/full/runs/{self.run_id}/{relpath}',obj,f'mc10b-full: {self.run_id} {relpath}')
        def poll_control(self,force=False):
            now=time.monotonic()
            with self._control_poll_lock:
                if not force and now-self._last_control_poll_monotonic<base.CONTROL_POLL_SECONDS:return
                self._last_control_poll_monotonic=now
            try:
                text,_=self.get_file('mc10b/full/control.json');self._telemetry_success()
                if not text:return
                ctl=json.loads(text)
                if ctl.get('run_id')!=self.run_id:return
                if str(ctl.get('action','run')).lower() in {'stop','abort','cancel'}:self.stop_event.set()
            except Exception as e:
                with self.lock:self.current['control_poll_warning_class']=type(e).__name__
                self._telemetry_failure(e)

    output_root=Path('/kaggle/working/output');out=output_root/'full_result';out.mkdir(parents=True,exist_ok=True)
    fatal=output_root/'mc10b_full_kaggle_failure.json';live=None;serve=None;work=Path(tempfile.mkdtemp(prefix='alice-mc10b-full-',dir='/tmp'))
    try:
        token=base.get_github_token();live=FullLive(token,base.GITHUB_REPO,branch,run_id)
        live.publish_current();live.poll_control(force=True);live.check_stop();live.start_heartbeat();live.event('FULL_GENERATION_WORKER_STARTED',{'max_generation_minutes':int(cfg['max_generation_minutes'])})
        gpu_count=0;gpu_inventory=[]
        try:
            import torch
            gpu_count=torch.cuda.device_count();gpu_inventory=[torch.cuda.get_device_name(i) for i in range(gpu_count)]
        except Exception:
            p=subprocess.run(['bash','-lc','nvidia-smi --query-gpu=name --format=csv,noheader'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            gpu_inventory=[x.strip() for x in p.stdout.splitlines() if x.strip()];gpu_count=len(gpu_inventory)
        live.update(gpu_count=gpu_count,gpu_inventory=gpu_inventory,checkpoint='GPU_PREFLIGHT',status='PREFLIGHT')
        require(gpu_count>=2,f'GPU_COUNT expected >=2 got {gpu_count}')
        src=base.locate_private_source(work,live,str(cfg['private_input_sha256']).upper())
        ns=base.source_namespace(src)
        mc10a,act,v1,h11,eligible,unitreg=b.verify_inputs(ns);bound=b.verify_bound_sources(ns,mc10a);q=b.verify_qualification_receipt(Path(ns.qualification_receipt))
        gate=fc.verify_pilot_audit_gate(src,b);pilot,remaining=fc.select_remaining_packets(eligible,b,gate['pilot_selected_rows'])
        require(len(eligible)==int(cfg['expected_eligible_packets']),'eligible packet drift')
        require(len(pilot)==int(cfg['expected_pilot_packets']),'pilot packet drift')
        require(len(remaining)==int(cfg['expected_remaining_packets']),'remaining packet drift')
        pilot_baseline=fc.build_block_telemetry(0,pilot,gate['pilot_primary_rows'],b,scope='AUDITED_PILOT_BASELINE')
        write_json(out/'mc10b_full_telemetry_pilot_baseline_v1.json',pilot_baseline)
        live.update(checkpoint='FULL_GENERATION_AUTHORITY_VERIFIED',status='PREFLIGHT',packet_total=len(remaining),telemetry_block_packet_count=fc.TELEMETRY_BLOCK_PACKET_COUNT)
        # Resume input, if any, is always revalidated against current frozen packets before model work.
        resume_rows=[];persisted_ack_blocks=set();prior_telemetry_by_block={};tail_recovery={'tail_recovered':False}
        rr=src/'resume_result'
        if rr.is_dir():
            candidates=[]
            for n in ['mc10b_full_einf_raw_candidates_v1.partial.jsonl','mc10b_full_einf_raw_candidates_v1.jsonl']:
                p=rr/n
                if p.is_file():
                    if n.endswith('.partial.jsonl'):
                        candidates,tail_recovery=fc.read_recoverable_partial_jsonl(p)
                    else:
                        candidates=read_jsonl(p)
                    break
            if candidates:
                resume_rows=fc.validate_resume_rows(candidates,remaining,unitreg,b,q['portfolio']['primary'])
            ackp=rr/'mc10b_full_telemetry_acknowledgements_v1.json'
            if ackp.is_file():
                ao=read_json(ackp);require(ao.get('artifact_id')=='alice.MC10B.full-einf.telemetry-acknowledgements.v1','telemetry ack artifact id')
                persisted_ack_blocks={int(x) for x in ao.get('acknowledged_review_blocks',[])};require(all(1<=x<=fc.TELEMETRY_BLOCK_COUNT for x in persisted_ack_blocks),'telemetry ack block range')
            tp=rr/'mc10b_full_block_telemetry_v1.jsonl'
            if tp.is_file():
                for t in read_jsonl(tp):
                    bi=int(t['block_index']);require(bi not in prior_telemetry_by_block,'duplicate prior telemetry block');prior_telemetry_by_block[bi]=t
        # Recompute every complete resume telemetry block from candidate rows. Persisted
        # metrics, gates, and acknowledgements are accepted only when they still match the
        # freshly recomputed state. This central helper is shared with regression tests.
        requested_ack_blocks={int(x) for x in cfg.get('acknowledged_telemetry_blocks',[])}
        tele_state=fc.validate_resume_telemetry_state(
            resume_rows,remaining,pilot_baseline,b,list(prior_telemetry_by_block.values()),
            persisted_ack_blocks,requested_ack_blocks,
        )
        acknowledged_blocks=set(tele_state['acknowledged_blocks'])
        write_json(out/'mc10b_full_resume_integrity_receipt_v1.json',{'artifact_id':'alice.MC10B.full-einf.resume-integrity.v1','resume_source_present':rr.is_dir(),'resume_candidates_validated':len(resume_rows),**tail_recovery,'regeneration_of_discarded_uncommitted_obligation_allowed':bool(tail_recovery.get('tail_recovered'))})
        if tail_recovery.get('tail_recovered'):
            live.event('RESUME_TORN_TAIL_RECOVERED',{'recovered_valid_candidate_rows':len(resume_rows),'discarded_tail_bytes':int(tail_recovery.get('discarded_tail_bytes',0))})
        live.update(resume_candidates_validated=len(resume_rows),checkpoint='RESUME_VALIDATED',telemetry_acknowledged_blocks=sorted(acknowledged_blocks))
        binary=base.install_runtime(live,work);serve,env=base.start_ollama(binary,work,live)
        spec=q['portfolio']['primary'];rt=base.pull_and_verify_model(binary,env,spec,live,int(cfg.get('model_pull_timeout_seconds',5400)))
        packets={p['packet_id']:p for p in remaining}
        source_manifest=fc.source_manifest(bound,pilot,remaining,b)
        write_json(out/'mc10b_full_generation_source_manifest_v1.json',source_manifest)
        write_json(out/'mc10b_full_generator_portfolio_receipt_v1.json',q)
        write_json(out/'mc10b_full_generator_runtime_manifest_v1.json',{
            'artifact_id':'alice.MC10B.full-einf-primary-generator-runtime.v1',**rt,
            'backend_role':'CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','generator_has_acceptance_authority':False,
            'generator_is_Alice_identity_model':False,'A_SYN_generation_enabled':False,'model_training_enabled':False,
        })
        rows=list(resume_rows);partial=out/'mc10b_full_einf_raw_candidates_v1.partial.jsonl'
        if rows:write_jsonl(partial,rows)
        selected_rows=fc.remaining_packet_projection(remaining)
        write_jsonl(out/'mc10b_full_remaining_packets_v1.jsonl',selected_rows)
        def refresh_telemetry(current_rows):
            records=[]
            for bi in range(1,fc.TELEMETRY_BLOCK_COUNT+1):
                bp=remaining[(bi-1)*fc.TELEMETRY_BLOCK_PACKET_COUNT:bi*fc.TELEMETRY_BLOCK_PACKET_COUNT]
                ids={p['packet_id'] for p in bp};br=[r for r in current_rows if r['packet_id'] in ids]
                if len(br)!=fc.TELEMETRY_BLOCK_CANDIDATE_COUNT:continue
                tele=fc.build_block_telemetry(bi,bp,br,b);gate_result=fc.evaluate_block_telemetry_gate(tele,pilot_baseline)
                acknowledged=bi in acknowledged_blocks
                gate_status='PASS' if not gate_result['review_required'] else ('ACKNOWLEDGED_REVIEW' if acknowledged else 'REVIEW_REQUIRED')
                tele['gate']=gate_result;tele['gate_status']=gate_status;tele['review_acknowledged']=acknowledged
                records.append(tele)
            write_jsonl(out/'mc10b_full_block_telemetry_v1.jsonl',records)
            write_json(out/'mc10b_full_telemetry_acknowledgements_v1.json',{'artifact_id':'alice.MC10B.full-einf.telemetry-acknowledgements.v1','acknowledged_review_blocks':sorted(acknowledged_blocks),'acknowledgements_do_not_accept_or_reject_candidates':True})
            return records
        telemetry_records=refresh_telemetry(rows)
        seen={(r['packet_id'],r['generation_method'],int(r['seed'])) for r in rows}
        started=time.monotonic();max_seconds=int(cfg['max_generation_minutes'])*60
        # generate_primary can make one attempt for each frozen prediction budget. Do not
        # begin a new obligation unless the configured generation window can accommodate
        # the worst-case timeout of the complete obligation. This bounds the generation
        # phase rather than overrunning it by up to three request timeouts.
        obligation_worst_case_seconds=int(cfg.get('generation_timeout_seconds',1800))*len(b.GENERATION_PREDICT_BUDGETS)+10
        live.update(status='RUNNING',checkpoint='FULL_EINF_GENERATION',model=spec['tag'],primary_candidates_generated=len(rows))
        soft_stop=False
        for pi,p in enumerate(remaining,1):
            live.check_stop(); packet_rows=[r for r in rows if r['packet_id']==p['packet_id']]
            for method in b.METHODS:
                for seed in b.SEEDS:
                    key=(p['packet_id'],method,seed)
                    if key in seen:continue
                    if time.monotonic()-started+obligation_worst_case_seconds>max_seconds:
                        soft_stop=True;break
                    live.check_stop();live.update(packet_index=pi,checkpoint='FULL_EINF_PACKET',primary_candidates_generated=len(rows))
                    rec=b.generate_primary(rt,p,unitreg,method,seed,int(cfg.get('generation_timeout_seconds',1800)))
                    b.validate_primary_record(rec,packets,unitreg,rt);fc.append_jsonl_fsync(partial,rec);rows.append(rec);packet_rows.append(rec);seen.add(key)
                    live.observation(f'primary/packet-{pi:02d}/{method}-{seed}.json',{
                        'run_id':run_id,'stage':'full-einf-generation','packet_index':pi,'method':method,'seed':seed,
                        'generation_attempts':rec['generation_attempts'],'generation_num_predict':rec['generation_num_predict'],
                        'ollama_done_reason':rec.get('ollama_done_reason'),'prompt_eval_count':rec.get('prompt_eval_count'),'eval_count':rec.get('eval_count'),
                        'unknown_preferred':bool(rec['payload'].get('unknown_preferred')),'prompt_sha256':rec['prompt_sha256'],'response_canonical_sha256':rec['response_canonical_sha256'],
                        'private_prompt_published':False,'private_candidate_payload_published':False,
                    })
                    live.update(primary_candidates_generated=len(rows))
                if soft_stop:break
            if len(packet_rows)==12:
                distinct=len({b.normalized_text(r['payload']['hypothesis_text']) for r in packet_rows})
                live.observation(f'primary/packet-{pi:02d}/summary.json',{'packet_index':pi,'candidate_count':12,'distinct_normalized_hypotheses':distinct,'unknown_preferred_count':sum(1 for r in packet_rows if r['payload'].get('unknown_preferred')),'retry_count':sum(int(r.get('generation_attempts',1))-1 for r in packet_rows)})
                require(distinct>=4,f'DIVERSITY_FLOOR packet {pi} has {distinct}')
            if not soft_stop and pi%fc.TELEMETRY_BLOCK_PACKET_COUNT==0 and len(packet_rows)==12:
                telemetry_records=refresh_telemetry(rows);current=[x for x in telemetry_records if int(x['block_index'])==pi//fc.TELEMETRY_BLOCK_PACKET_COUNT]
                require(len(current)==1,'completed telemetry block missing');tele=current[0];g=tele['gate'];status=tele['gate_status']
                checkpoint={'artifact_id':'alice.MC10B.full-einf-generation.checkpoint.v1','run_id':run_id,'completed_candidate_obligations':len(rows),'remaining_candidate_obligations':fc.TOTAL_REMAINING_CANDIDATES-len(rows),'generation_soft_stop_reached':False,'resume_supported':True,'last_completed_telemetry_block':tele['block_index'],'telemetry_gate_status':status,'telemetry_review_reasons':g['review_reasons'],'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_performed':False,'MC10B_complete':False,'MC10C_start_allowed':False,'stage_g_closed':False}
                write_json(out/'mc10b_full_generation_checkpoint_v1.json',checkpoint)
                live.observation(f'telemetry/block-{tele["block_index"]:02d}.json',fc.public_block_telemetry(tele,g,status))
                live.update(checkpoint='TELEMETRY_BLOCK_'+str(tele['block_index']),telemetry_block_index=tele['block_index'],telemetry_gate_status=status,primary_candidates_generated=len(rows))
                live.event('FULL_EINF_TELEMETRY_BLOCK_CHECKPOINT',{'block_index':tele['block_index'],'candidate_count':tele['candidate_count'],'gate_status':status,'review_reasons':g['review_reasons'],'candidate_contents_published':False})
                if g['structural_failures']:
                    raise RuntimeError('TELEMETRY_STRUCTURAL_FAILURE block='+str(tele['block_index'])+' '+','.join(g['structural_failures']))
                if status=='REVIEW_REQUIRED':
                    live.update(status='TELEMETRY_REVIEW_REQUIRED',checkpoint='TELEMETRY_REVIEW_REQUIRED',packet_index=pi,primary_candidates_generated=len(rows))
                    write_json(output_root/'mc10b_full_kaggle_status.json',{'run_id':run_id,'stage':'full-einf-generation','status':'TELEMETRY_REVIEW_REQUIRED','telemetry_block_index':tele['block_index'],'review_reasons':g['review_reasons'],'completed_candidate_obligations':len(rows),'resume_required':True,'stage_g_closed':False})
                    safe_remove_model(base,binary,env,spec['tag'],live);return 0
            if soft_stop:break
        telemetry_records=refresh_telemetry(rows)
        checkpoint={'artifact_id':'alice.MC10B.full-einf-generation.checkpoint.v1','run_id':run_id,'completed_candidate_obligations':len(rows),'remaining_candidate_obligations':fc.TOTAL_REMAINING_CANDIDATES-len(rows),'generation_soft_stop_reached':soft_stop,'resume_supported':True,'completed_telemetry_blocks':len(telemetry_records),'last_completed_telemetry_block':telemetry_records[-1]['block_index'] if telemetry_records else 0,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_performed':False,'MC10B_complete':False,'MC10C_start_allowed':False,'stage_g_closed':False}
        write_json(out/'mc10b_full_generation_checkpoint_v1.json',checkpoint)
        if soft_stop:
            live.update(status='PARTIAL_CHECKPOINT_READY',checkpoint='SOFT_STOP',primary_candidates_generated=len(rows),packet_index=min(60,1+len({r['packet_id'] for r in rows})))
            live.event('FULL_GENERATION_PARTIAL_CHECKPOINT',{'completed_candidate_obligations':len(rows),'remaining_candidate_obligations':720-len(rows)})
            write_json(output_root/'mc10b_full_kaggle_status.json',{'run_id':run_id,'stage':'full-einf-generation','status':'PARTIAL_CHECKPOINT','completed_candidate_obligations':len(rows),'resume_required':True,'stage_g_closed':False})
            safe_remove_model(base,binary,env,spec['tag'],live);return 0
        require(len(rows)==int(cfg['expected_candidate_obligations']),'full remaining candidate total mismatch')
        telemetry_records=refresh_telemetry(rows);require(len(telemetry_records)==fc.TELEMETRY_BLOCK_COUNT,'expected six completed telemetry blocks')
        require(all(t['gate_status'] in {'PASS','ACKNOWLEDGED_REVIEW'} for t in telemetry_records),'unresolved telemetry review gate')
        # Revalidate complete pool and per-packet diversity before finalizing.
        rows=fc.validate_resume_rows(rows,remaining,unitreg,b,spec)
        for p in remaining:
            pr=[r for r in rows if r['packet_id']==p['packet_id']];require(len(pr)==12,'packet candidate count '+p['packet_id']);require(len({b.normalized_text(r['payload']['hypothesis_text']) for r in pr})>=4,'final diversity '+p['packet_id'])
        final=out/'mc10b_full_einf_raw_candidates_v1.jsonl';write_jsonl(final,rows);partial.unlink(missing_ok=True)
        unk=fc.unknown_rows(remaining);write_jsonl(out/'mc10b_full_unknown_competitors_v1.jsonl',unk)
        blind=fc.blinded_handoff_rows(rows,unk)
        write_jsonl(out/'mc10b_full_blinded_future_evaluation_handoff_v1.jsonl',blind)
        distinct={p['packet_id']:len({b.normalized_text(r['payload']['hypothesis_text']) for r in rows if r['packet_id']==p['packet_id']}) for p in remaining}
        summary={'artifact_id':'alice.MC10B.full-einf-frontier-summary.v1','eligible_EINF_packets_total':65,'pilot_packets_generated_and_audited':5,'remaining_packets_generated_this_stage':60,'raw_EINF_candidates_generated_this_stage':720,'total_raw_EINF_candidates_across_pilot_and_full_generation':780,'unknown_competitors_created_this_stage':60,'generation_methods':4,'seeds_per_method':3,'candidate_ensemble_size_per_packet':12,'primary_model':spec['tag'],'primary_model_digest':spec['digest'],'reserve_model_calls':0,'normalized_distinct_non_null_candidates_by_packet':distinct,'exact_normalized_diversity_failures':0,'candidate_visible_MC8_hidden_evaluator_material_loaded':0,'generator_is_Alice_identity_model':False,'generator_has_acceptance_authority':False,'challenger_outputs_count_as_EINF_candidates':False,'A_SYN_generation_enabled':False,'autonomous_A_SYN_promotion_enabled':False,'model_training_enabled':False,'E_INF_generated_count_this_stage':720,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'MC10_saturation_rounds_credited':0,'MC10B_full_EINF_frontier_generation_complete':True,'MC10B_complete':False,'MC10C_start_allowed':False,'stage_g_closed':False,'stage_h_activated':False,'phase2_replaced':False,'broader_canon_falsification_required_before_acceptance':True,'unknown_remains_competitor':True,'telemetry_block_packet_count':fc.TELEMETRY_BLOCK_PACKET_COUNT,'telemetry_blocks_completed':len(telemetry_records),'telemetry_review_blocks_acknowledged':sorted(acknowledged_blocks),'telemetry_all_gates_resolved':all(t['gate_status'] in {'PASS','ACKNOWLEDGED_REVIEW'} for t in telemetry_records)}
        write_json(out/'mc10b_full_summary_v1.json',summary)
        write_json(out/'mc10b_full_frontier_coverage_receipt_v1.json',{'artifact_id':'alice.MC10B.full-einf-frontier-coverage.v1','eligible_EINF_packets_total':65,'pilot_packets_generated_and_audited':5,'remaining_packets_generated':60,'pilot_raw_EINF_candidates':60,'remaining_raw_EINF_candidates':720,'total_raw_EINF_candidates_across_pilot_and_full_generation':780,'all_65_EINF_eligible_packets_have_raw_candidate_pools':True,'E_INF_accepted_count':0})
        write_json(out/'mc10b_full_generation_closure_receipt_v1.json',{'artifact_id':'alice.MC10B.full-einf-generation-closure.v1','MC10B_full_EINF_frontier_generation_complete':True,'raw_EINF_candidates_generated_this_stage':720,'remaining_packets_generated':60,'unknown_competitors_created':60,'all_candidates_raw_unevaluated':True,'broader_canon_falsification_required_before_acceptance':True,'challenger_or_independent_evaluation_required_before_acceptance':True,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_performed':False,'MC10_saturation_rounds_credited':0,'MC10B_complete':False,'MC10C_start_allowed':False,'stage_g_closed':False,'stage_h_activated':False,'phase2_replaced':False,'telemetry_blocks_completed':len(telemetry_records),'telemetry_review_blocks_acknowledged':sorted(acknowledged_blocks),'telemetry_all_gates_resolved':True,'next_gate':'MC10B_FULL_FRONTIER_FIDELITY_EVALUATION'})
        (out/'ALICE_MC10B_FULL_EINF_FRONTIER_V1.md').write_text('# A.L.I.C.E. MC10B — Full E-INF Frontier Generation v1\n\nThis package contains the raw canonical 12-candidate E-INF pools for the 60 MC10A E-INF-eligible packets not used in the five-packet audited pilot. GPT-OSS 20B remains the frozen proposal-only generator. UNKNOWN remains a competitor for every packet. No E-INF is accepted here; no A-SYN is generated; no model is trained; MC10B/MC10C/Stage G remain open. Broader-canon/hidden-E0 falsification is mandatory before acceptance.\n',encoding='utf-8',newline='\n')
        (out/'validate_alice_mc10b_full_einf_frontier_v1.py').write_text(fc.validator_source(),encoding='utf-8',newline='\n')
        fc.write_manifest(out)
        proc=subprocess.run([sys.executable,str(out/'validate_alice_mc10b_full_einf_frontier_v1.py'),str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        require(proc.returncode==0,'generated full validator failed: '+proc.stdout[-3000:])
        live.update(status='FULL_EINF_FRONTIER_GENERATION_COMPLETE_AWAITING_FIDELITY_EVALUATION',checkpoint='FULL_GENERATION_COMPLETE',packet_index=60,primary_candidates_generated=720)
        live.event('FULL_EINF_FRONTIER_GENERATION_COMPLETE',{'raw_EINF_candidates_generated_this_stage':720,'remaining_packets_generated':60,'E_INF_accepted_count':0})
        write_json(output_root/'mc10b_full_kaggle_status.json',{'run_id':run_id,'stage':'full-einf-generation','status':'COMPLETE','raw_EINF_candidates_generated':720,'stage_g_closed':False})
        safe_remove_model(base,binary,env,spec['tag'],live);return 0
    except base.StopRequested as e:
        write_json(fatal,{'run_id':run_id,'status':'STOPPED_BY_CONTROL','message':str(e),'stage_g_closed':False});
        if live:
            try:live.update(status='STOPPED_BY_CONTROL',checkpoint='STOPPED',failure_class='STOP_REQUESTED',failure_message=str(e))
            except Exception:pass
        return 75
    except Exception as e:
        msg=str(e);payload={'run_id':run_id,'status':'FAILED','failure_class':type(e).__name__,'message':msg[:4000],'traceback':traceback.format_exc()[-12000:],'stage_g_closed':False,'MC10B_complete':False,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_performed':False}
        # Full error detail remains only in the private Kaggle output. The public live
        # branch receives class/state metadata and never raw exception text/tracebacks.
        write_json(fatal,payload)
        if live:
            try:
                public_failure={'run_id':run_id,'stage':'full-einf-generation','status':'FAILED','failure_class':type(e).__name__,'private_failure_details_published':False,'stage_g_closed':False}
                live.update(status='FAILED',checkpoint='FAILED',failure_class=type(e).__name__,failure_message='See private Kaggle failure artifact.')
                live.event('FAILURE',public_failure)
            except Exception:pass
        return 1
    finally:
        if live:live.close()
        if serve is not None and serve.poll() is None:
            try:serve.terminate();serve.wait(timeout=10)
            except Exception:
                try:serve.kill()
                except Exception:pass
        shutil.rmtree(work,ignore_errors=True)

if __name__=='__main__':raise SystemExit(main())
