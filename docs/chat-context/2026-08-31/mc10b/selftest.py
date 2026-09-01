#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest().upper()
def req(c,m):
    if not c: raise RuntimeError(m)
def load(name,mod):
    p=ROOT/name;s=importlib.util.spec_from_file_location(mod,p);req(s and s.loader,'spec '+name)
    m=importlib.util.module_from_spec(s);sys.modules[mod]=m;s.loader.exec_module(m);return m

def compile_in_memory(p:Path):
    compile(p.read_text(encoding='utf-8'),str(p),'exec')

def main():
    checks=[]
    def ok(name): checks.append(name+'=true')

    for p in sorted(ROOT.glob('*.py')): compile_in_memory(p)
    ok('python_compile_all_in_memory')

    ctl=load('mc10b_full_controller.py','ctl_test_v110')
    tc=load('mc10b1_transport_common.py','tc_test_v110')
    b=load('build_mc10b1_portfolio_pilot_v1_1.py','builder_test_v110')
    fc=load('mc10b_full_common.py','fc_test_v110')

    for n,h in ctl.FROZEN_HASHES.items(): req(sha(ROOT/n)==h,'frozen hash '+n)
    ok('frozen_generation_and_authority_hashes')

    req(tc.PRIVATE_BLOB_NAME=='mc10b1-private-input.bin','transport private blob name')
    expected=ctl.dataset_expected_files(tc.PRIVATE_BLOB_NAME)
    req(tc.PRIVATE_BLOB_NAME in expected,'dataset contract includes transport blob')
    req('mc10b-private-input.bin' not in expected,'old mismatched blob forbidden')
    controller_text=(ROOT/'mc10b_full_controller.py').read_text(encoding='utf-8')
    probe_text=(ROOT/'mc10b_full_mount_probe.py').read_text(encoding='utf-8')
    worker_text=(ROOT/'mc10b_full_kaggle_worker.py').read_text(encoding='utf-8')
    req('transport.PRIVATE_BLOB_NAME' in controller_text,'controller derives blob name from transport')
    req('tc.PRIVATE_BLOB_NAME' in probe_text,'probe checks transport blob constant')
    req('tc.PRIVATE_BLOB_NAME' in worker_text,'worker checks transport blob constant')
    req("fc.validate_resume_telemetry_state(" in worker_text,'worker uses shared resume telemetry validator')
    req("read_recoverable_partial_jsonl" in probe_text,'probe uses torn-tail recovery')
    ok('cross_file_contract_and_resume_unification')

    with tempfile.TemporaryDirectory(prefix='alice-mc10b-v110-selftest-') as tmp:
        td=Path(tmp)
        # Deterministic transport round trip.
        src=td/'src';src.mkdir()
        for d in tc.SOURCE_ROOT_DIR_MARKERS:
            (src/d).mkdir(parents=True);(src/d/'marker.txt').write_text('x\n',encoding='utf-8')
        (src/tc.SOURCE_ROOT_FILE_MARKER).write_text('{}\n',encoding='utf-8')
        arc=td/'input.zip';info=tc.pack_private_source(src,arc,prefix='source')
        req(info['source_prefix']=='source/','source prefix')
        extracted=tc.safe_extract_private_archive(arc,td/'extract');tc.validate_source_tree(extracted)
        ok('deterministic_private_transport_roundtrip')

        # Exact local dataset contract.
        data=td/'dataset';data.mkdir();shutil.copy2(arc,data/tc.PRIVATE_BLOB_NAME)
        for n in ['build_mc10b1_portfolio_pilot_v1_1.py','mc10b1_kaggle_worker.py','mc10b1_transport_common.py','mc10b_full_common.py','ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json']:
            shutil.copy2(ROOT/n,data/n)
        cfg={'private_blob_name':tc.PRIVATE_BLOB_NAME,'private_input_sha256':sha(arc),
             'builder_sha256':sha(ROOT/'build_mc10b1_portfolio_pilot_v1_1.py'),'base_worker_sha256':sha(ROOT/'mc10b1_kaggle_worker.py'),
             'transport_common_sha256':sha(ROOT/'mc10b1_transport_common.py'),'full_common_sha256':sha(ROOT/'mc10b_full_common.py'),
             'qualification_sha256':sha(ROOT/'ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json'),
             'expected_remaining_packets':60,'expected_candidate_obligations':720}
        (data/'mc10b-full-run-config.json').write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        (data/'alice-github-token.txt').write_text('x'*40+'\n',encoding='utf-8')
        ctl.validate_dataset_contract(data,cfg,tc)
        ok('local_dataset_contract_validator')

        # Kaggle listing parser: documented CSV and conservative table fallback.
        listing='\n'.join(f'{n}  123B  2026-08-30' for n in sorted(expected))
        req(expected.issubset(ctl.parse_dataset_file_listing(listing)),'dataset table listing parser')
        csv_listing='name,size,creationDate\n'+'\n'.join(f'{n},123,2026-08-30' for n in sorted(expected))
        req(expected==ctl.parse_dataset_file_listing(csv_listing),'dataset CSV listing parser')
        ok('kaggle_dataset_listing_parser')

        # Real pilot gate verifies exact owner/challenger artifacts and internal 15-file manifest.
        gate_root=td/'gate';pa=gate_root/'pilot_audit';pa.mkdir(parents=True)
        for n in [fc.CLOSURE_NAME,fc.AUDIT_ZIP_NAME,'ALICE_MC10B1_CHALLENGER_FIDELITY_AUDIT_20260830.md','ALICE_MC10B1_CHALLENGER_AUDIT_VERDICTS_20260830.jsonl','ALICE_MC10B1_POST_CHALLENGER_PROVISIONAL_EINF_SET_20260830.md']:
            shutil.copy2(ROOT/n,pa/n)
        gate=fc.verify_pilot_audit_gate(gate_root,b)
        req(len(gate['pilot_selected_rows'])==5 and len(gate['pilot_primary_rows'])==60,'pilot audit counts')
        real_baseline=fc.build_block_telemetry(0,gate['pilot_selected_rows'],gate['pilot_primary_rows'],b,scope='AUDITED_PILOT_BASELINE')
        req(real_baseline['candidate_count']==60 and real_baseline['packet_count']==5,'pilot telemetry baseline')
        ok('real_pilot_audit_and_telemetry_baseline')

        # Crash-safe JSONL recovery: a final torn UTF-8 record is recoverable; middle corruption is not.
        good=fc.canon({'x':'ok'}).encode()+b'\n'
        torn='{"x":"caf\u00e9'.encode('utf-8')[:-1]
        part=td/'partial.jsonl';part.write_bytes(good+torn)
        recovered,receipt=fc.read_recoverable_partial_jsonl(part)
        req(recovered==[{'x':'ok'}] and receipt['tail_recovered'] is True and receipt['discarded_tail_bytes']==len(torn),'torn utf8 final recovery')
        middle=td/'middle.jsonl';middle.write_bytes(good+b'{broken}\n'+good)
        try:
            fc.read_recoverable_partial_jsonl(middle)
            raise RuntimeError('middle corruption accepted')
        except RuntimeError as e:
            req('before final line' in str(e),'wrong middle corruption failure')
        ok('crash_safe_jsonl_tail_recovery')

        class FakeBuilder:
            METHODS=tuple(b.METHODS);SEEDS=tuple(int(x) for x in b.SEEDS)
            @staticmethod
            def normalized_text(x): return ' '.join(str(x).lower().split())
            @staticmethod
            def validate_primary_record(rec,packets,unitreg,rt):
                req(rec['packet_id'] in packets,'fake packet membership')
                req(rec['generation_method'] in FakeBuilder.METHODS,'fake method')
                req(int(rec['seed']) in FakeBuilder.SEEDS,'fake seed')
                req(rec['candidate_state']=='RAW_UNEVALUATED' and rec['provenance_class']=='E-INF','fake authority')
        fb=FakeBuilder()
        methods=list(fb.METHODS);seeds=list(fb.SEEDS)
        packets=[];rows=[]
        for i in range(60):
            pid=f'PACKETX{i:02d}'
            p={'packet_id':pid,'cluster_id':f'C{i:02d}','graph_id':f'G{i:02d}','system':'synthetic-system',
               'candidate_kind':'synthetic-kind','MC4_priority_tier':'P1','candidate_relevant_independent_E0_family_count':2,
               'post_freeze_A_SYN_eligible':False,'packet_content_sha256':hashlib.sha256(pid.encode()).hexdigest().upper()}
            packets.append(p)
            for mi,m in enumerate(methods):
                for si,seed in enumerate(seeds):
                    cid=f'EINF-{i:02d}-{mi}-{si}'
                    text=f'{pid} {m} seed{seed} distinct behavioral hypothesis {mi}-{si}'
                    rows.append({'candidate_id':cid,'packet_id':pid,'cluster_id':p['cluster_id'],'graph_id':p['graph_id'],
                                 'generation_method':m,'seed':seed,'generation_attempts':1,'candidate_state':'RAW_UNEVALUATED',
                                 'provenance_class':'E-INF','historical_Elaina_truth':False,'E0_anchor_family_ids':['F1','F2'],
                                 'payload':{'hypothesis_text':text,'hypothesis_probability':0.65,'unknown_preferred':False}})
        req(len(rows)==720,'synthetic rows')
        qualification={'portfolio':{'primary':{'tag':'gpt-oss:20b','digest':'abc123','qualified_profile':'synthetic-profile'}}}
        unitreg={}
        req(len(fc.validate_resume_rows(rows,packets,unitreg,fb,qualification['portfolio']['primary']))==720,'synthetic resume validation')

        # Six complete blocks with deliberately shifted baseline so every review path is exercised and acknowledged.
        baseline={'scope':'AUDITED_PILOT_BASELINE','candidate_count':60,'retry_obligation_rate':0.0,'probability':{'mean':0.10}}
        tele=[];acked=[]
        for bi in range(1,7):
            bp=packets[(bi-1)*10:bi*10]
            t=fc.build_block_telemetry(bi,bp,rows,fb);g=fc.evaluate_block_telemetry_gate(t,baseline)
            req(not g['structural_failures'],'telemetry structural '+str(bi));req(g['review_required'],'synthetic review expected')
            t['gate']=g;t['gate_status']='ACKNOWLEDGED_REVIEW';t['review_acknowledged']=True;tele.append(t);acked.append(bi)
        req(len(tele)==6 and acked==[1,2,3,4,5,6],'six telemetry blocks')
        ok('six_block_telemetry_scale_test')

        # Resume telemetry is recomputed and cryptographically/semantically bound; tampering and invalid ack are rejected.
        state=fc.validate_resume_telemetry_state(rows,packets,baseline,fb,tele,set(acked),set())
        req(state['acknowledged_blocks']==set(acked),'resume ack state')
        tampered=[dict(x) for x in tele];tampered[0]=dict(tampered[0]);tampered[0]['candidate_count']=119
        try:
            fc.validate_resume_telemetry_state(rows,packets,baseline,fb,tampered,set(acked),set())
            raise RuntimeError('tampered telemetry accepted')
        except RuntimeError as e: req('drift' in str(e) or 'candidate' in str(e),'unexpected telemetry tamper error')
        try:
            fc.validate_resume_telemetry_state(rows[:60],packets,baseline,fb,[],set(),{2})
            raise RuntimeError('future ack accepted')
        except RuntimeError: pass
        ok('resume_telemetry_recompute_and_ack_binding')

        # Build a complete synthetic final package and run both standalone + independent controller validation.
        final=td/'full_result';final.mkdir()
        unknowns=fc.unknown_rows(packets);blind=fc.blinded_handoff_rows(rows,unknowns)
        fc.write_jsonl(final/'mc10b_full_einf_raw_candidates_v1.jsonl',rows)
        fc.write_jsonl(final/'mc10b_full_unknown_competitors_v1.jsonl',unknowns)
        fc.write_jsonl(final/'mc10b_full_remaining_packets_v1.jsonl',fc.remaining_packet_projection(packets))
        fc.write_jsonl(final/'mc10b_full_blinded_future_evaluation_handoff_v1.jsonl',blind)
        fc.write_jsonl(final/'mc10b_full_block_telemetry_v1.jsonl',tele)
        fc.write_json(final/'mc10b_full_telemetry_pilot_baseline_v1.json',baseline)
        fc.write_json(final/'mc10b_full_telemetry_acknowledgements_v1.json',{'artifact_id':'alice.MC10B.full-einf.telemetry-acknowledgements.v1','acknowledged_review_blocks':acked,'acknowledgements_do_not_accept_or_reject_candidates':True})
        source_manifest={'artifact_id':'synthetic-source-manifest','remaining_packet_ids':[p['packet_id'] for p in packets]}
        fc.write_json(final/'mc10b_full_generation_source_manifest_v1.json',source_manifest)
        fc.write_json(final/'mc10b_full_generator_portfolio_receipt_v1.json',qualification)
        fc.write_json(final/'mc10b_full_generator_runtime_manifest_v1.json',{'artifact_id':'alice.MC10B.full-einf-primary-generator-runtime.v1','model_name':'gpt-oss:20b','model_digest':'abc123','qualified_profile':'synthetic-profile','backend_role':'CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY','generator_has_acceptance_authority':False,'generator_is_Alice_identity_model':False})
        fc.write_json(final/'mc10b_full_resume_integrity_receipt_v1.json',{'artifact_id':'alice.MC10B.full-einf.resume-integrity.v1','resume_source_present':False,'resume_candidates_validated':0,'tail_recovered':False,'regeneration_of_discarded_uncommitted_obligation_allowed':False})
        fc.write_json(final/'mc10b_full_generation_checkpoint_v1.json',{'artifact_id':'alice.MC10B.full-einf-generation.checkpoint.v1','completed_candidate_obligations':720,'remaining_candidate_obligations':0,'generation_soft_stop_reached':False})
        fc.write_json(final/'mc10b_full_summary_v1.json',{'raw_EINF_candidates_generated_this_stage':720,'E_INF_accepted_count':0,'A_SYN_generated_count':0,'model_training_enabled':False,'telemetry_blocks_completed':6,'telemetry_all_gates_resolved':True})
        fc.write_json(final/'mc10b_full_generation_closure_receipt_v1.json',{'MC10B_full_EINF_frontier_generation_complete':True,'broader_canon_falsification_required_before_acceptance':True,'MC10C_start_allowed':False,'telemetry_blocks_completed':6,'telemetry_all_gates_resolved':True})
        fc.write_json(final/'mc10b_full_frontier_coverage_receipt_v1.json',{'eligible_EINF_packets_total':65,'pilot_packets_generated_and_audited':5,'remaining_packets_generated':60,'total_raw_EINF_candidates_across_pilot_and_full_generation':780})
        (final/'ALICE_MC10B_FULL_EINF_FRONTIER_V1.md').write_text('# synthetic full frontier\n',encoding='utf-8',newline='\n')
        (final/'validate_alice_mc10b_full_einf_frontier_v1.py').write_text(fc.validator_source(),encoding='utf-8',newline='\n')
        fc.write_manifest(final)
        standalone=load_validator=final/'validate_alice_mc10b_full_einf_frontier_v1.py'
        ns={};exec(compile(standalone.read_text(encoding='utf-8'),str(standalone),'exec'),ns);ns['main'](str(final))
        source_info={'remaining':packets,'unitreg':unitreg,'qualification':qualification,'expected_source_manifest':source_manifest,
                     'pilot_baseline':baseline,'resume_source_present':False,'resume_count':0}
        ctl.independent_validate_completed_result(final,sys.executable,source_info,fc,fb)
        ok('synthetic_720_candidate_dual_final_validation')

        # Exact final file set is fail-closed against extras.
        extra=td/'extra_result';shutil.copytree(final,extra);(extra/'unexpected.txt').write_text('x\n',encoding='utf-8')
        try:
            ns['main'](str(extra));raise RuntimeError('extra file accepted')
        except RuntimeError as e: req('file set' in str(e),'wrong extra-file failure')
        ok('final_exact_file_set_fail_closed')

        # Atomic publication/idempotency logic is tested separately from the already
        # completed full independent validation to avoid repeating its O(block-pair)
        # telemetry recomputation five additional times in every package qualification.
        audit=td/'audit';audit.mkdir();orig_validate=ctl.independent_validate_completed_result
        atomic_calls=[]
        def shallow_validate(result,*_args,**_kwargs):
            req((result/'SHA256SUMS.txt').is_file(),'atomic shallow manifest');atomic_calls.append(str(result))
        ctl.independent_validate_completed_result=shallow_validate
        try:
            published=ctl.verify_and_publish_canonical(final,audit,sys.executable,'testrun',source_info,fc,fb)
            req(published.is_dir(),'atomic publish')
            published2=ctl.verify_and_publish_canonical(final,audit,sys.executable,'testrun2',source_info,fc,fb)
            req(published2==published,'idempotent canonical publish')
            req(len(atomic_calls)>=4,'atomic validation hooks not exercised')
        finally:
            ctl.independent_validate_completed_result=orig_validate
        ok('atomic_canonical_publish_and_idempotency')

    # Structured preflight retry: deterministic failure once; approved transient mount miss exactly one retry.
    with tempfile.TemporaryDirectory(prefix='alice-mc10b-preflight-policy-') as tmp2:
        td2=Path(tmp2);probe_dir=td2/'probe';probe_dir.mkdir();outdir=td2/'out';runroot=td2/'run';runroot.mkdir()
        orig=(ctl.run_cmd,ctl.wait_kernel,ctl.download_kernel_output,ctl.locate_one,ctl.time.sleep)
        try:
            ctl.time.sleep=lambda _s: None
            pushes=[];payloads=[{'pass':False,'retryable':False,'failure_code':'PRIVATE_BLOB_HASH_MISMATCH','failure_class':'ProbeError','failure_message':'hash mismatch'}]
            def fake_run(args,**kw):
                if 'push' in args: pushes.append(1)
                return ctl.CmdResult(0,'')
            def fake_wait(*a,**k): return 'COMPLETE'
            def fake_download(kaggle,ref,dst):
                dst.mkdir(parents=True,exist_ok=True);(dst/'mc10b_full_mount_probe.json').write_text(json.dumps(payloads[min(len(pushes)-1,len(payloads)-1)])+'\n',encoding='utf-8');return ctl.CmdResult(0,'')
            def fake_locate(root,name): return root/name if (root/name).is_file() else None
            ctl.run_cmd,ctl.wait_kernel,ctl.download_kernel_output,ctl.locate_one=fake_run,fake_wait,fake_download,fake_locate
            try: ctl.run_cpu_preflight('k',probe_dir,'ref',outdir,runroot,True)
            except Exception: pass
            req(len(pushes)==1,'deterministic preflight failure retried')
            pushes.clear();payloads[:]=[{'pass':False,'retryable':True,'failure_code':'MOUNT_PRIVATE_BLOB_MISSING','failure_class':'ProbeError','failure_message':'missing'},{'pass':True,'retryable':False,'checkpoint':'COMPLETE'}]
            result=ctl.run_cpu_preflight('k',probe_dir,'ref',outdir,runroot,True)
            req(result.get('pass') is True and len(pushes)==2,'controlled mount retry policy')
        finally:
            ctl.run_cmd,ctl.wait_kernel,ctl.download_kernel_output,ctl.locate_one,ctl.time.sleep=orig
    ok('structured_preflight_retry_policy')

    # Static deep guards around recovery, privacy, scheduling, and launcher delegation.
    ctl_text=(ROOT/'mc10b_full_controller.py').read_text(encoding='utf-8')
    worker_text=(ROOT/'mc10b_full_kaggle_worker.py').read_text(encoding='utf-8')
    probe_text=(ROOT/'mc10b_full_mount_probe.py').read_text(encoding='utf-8')
    req('terminal_output_retrieved' in ctl_text and 'remote_resources_preserved_for_recovery' in ctl_text,'remote recovery preservation guard')
    req('checkpoint == "CANONICAL_FINALIZATION" and remote_stage_status == "COMPLETE"' in ctl_text,'worker failure state clobber guard')
    req('base_worker_sha256' in probe_text and 'qualification_sha256' in probe_text,'preflight deterministic input hashes')
    req("failure_message='See private Kaggle failure artifact.'" in worker_text and "private_failure_details_published':False" in worker_text,'public failure privacy guard')
    req('obligation_worst_case_seconds' in worker_text and 'len(b.GENERATION_PREDICT_BUDGETS)' in worker_text,'generation soft deadline guard')
    req('safe_remove_model' in worker_text and worker_text.count('base.remove_model(binary,env,spec')==0,'cleanup must be best effort')
    req('sys.version_info >= (3, 11)' in ctl_text,'python version gate')
    ok('deep_failure_privacy_and_deadline_guards')

    ps=(ROOT/'Start-ALICEMC10BFullEInfGeneration-v1.1.0.ps1').read_text(encoding='utf-8')
    req('mc10b-private-input.bin' not in ps,'old blob name in powershell')
    req('[ValidateRange(120,600)]' in ps,'powershell generation minimum mismatch')
    req('mc10b_full_controller.py' in ps and ps.count('& $Python @ArgsList')==1,'powershell delegates once')
    req(ps.count('{')==ps.count('}') and ps.count('(')==ps.count(')'),'powershell delimiter balance')
    ok('minimal_powershell_wrapper_static_sanity')

    req(not (ROOT/'__pycache__').exists(),'selftest created package __pycache__')
    print('\n'.join(checks))
    print(f'selftest_check_count={len(checks)}')
    return checks

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('SELFTEST_FAILED='+repr(e));raise
