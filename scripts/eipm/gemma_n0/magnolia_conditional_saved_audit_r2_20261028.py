"""Read exact saved experiment evidence inside owner isolation; no model calls."""
from contextlib import ExitStack
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import traceback

ROOT=Path('/homes/01/mxrayan/rayan-compute')
REV='333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME='rayan-conditional-saved-audit-r2-20261028'
SUMMARY_SHA='f83ac86c2ab439350cf471718edf6d8689b8f2253e86141f50461752aec72e70'
LOG_SHA='c378363ada61702cb89a6ed48b82d5aa619fdc8cb719d93a2a7a5a776215cc50'
DEPENDENCIES={
 'helper':('helper.py','fbb253cbc09e63fb52069371e87c86811800d6d7776b1a44751e92af38cf5605'),
 'source':('source.py','dcb10bed56daa4d54605e072349b331c84921d42369f81f5a491172b70712689'),
 'analysis':('analysis.py','8e0982e758ad81d735c474101fdd2e271d4dee3e124d33fc7f1e129e0c50b6c7'),
 'live':('live.py','7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533')}


def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def require(condition,message='saved experiment binding differs'):
    if not condition:raise ValueError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def write(path,value):
    with path.open('xb') as out:out.write(canonical(value)+b'\n')
    return sha256(path.read_bytes()).hexdigest()


def seal(value):
    require(sha256(canonical({k:v for k,v in value.items() if k!='receipt_sha256'})).hexdigest()==value['receipt_sha256'])
    return value


def progress_acknowledgments(pinned):
    pinned.stream.seek(0)  # Entry/closing hash verification leaves the descriptor at EOF.
    records=[json.loads(line) for line in pinned.stream if line.startswith(b'{')]
    records=[r for r in records if r['stage']=='comparison_update_completed']
    require([r['completed_update_events'] for r in records]==list(range(1,81)))
    return records


def run(namespace):
    isolation=namespace._p2_guard()  # FIRST: never re-enter after descriptors exist.
    os.umask(0o077);sys.dont_write_bytecode=True
    code=namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'],ROOT,file=False)
    namespace._clean_code(code,REV);sys.path.insert(0,str(code))
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import raw_inference_lineage as raw
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    with ExitStack() as stack:
        held=[]
        def pin(path,digest,limit=512*1024**2):
            path=namespace._bounded_path(str(path),ROOT,file=True)
            require(stat.S_IMODE(path.stat().st_mode)==0o600)
            value=stack.enter_context(_PinnedFile(path,digest,limit));held.append(value);return value
        def read(path,digest):return raw._json(pin(path,digest).bounded_bytes())
        def lines(path,digest):
            value=pin(path,digest);value.stream.seek(0)
            return [raw._json(line) for line in value.stream]
        modules={}
        for key,(suffix,digest) in DEPENDENCIES.items():
            path=ROOT/'jobs'/(NAME+'-'+suffix);pin(path,digest)
            modules[key]=module('conditional_saved_'+key,path)
        helper,source,analyzer,live=(modules[k] for k in ('helper','source','analysis','live'))
        live.verify_active_namespace(namespace,isolation)
        worker_sha=sha256(Path(__file__).read_bytes()).hexdigest();pin(Path(__file__),worker_sha)
        result=seal(read(ROOT/'jobs/rayan-conditional-learning-20261025.json',SUMMARY_SHA))
        require(result['job_id']==579741 and result['complete_source_and_checkpoint_custody_verified'])
        require(result['total_committed_experimental_updates']==80 and result['winner_selected'] is False)
        directory=ROOT/'rayan-personality/private-compile/rayan-conditional-learning-579741'
        # The plan was published by canonical digest, not by raw-file digest.
        plan_path=namespace._bounded_path(str(directory/'conditional-learning-plan.protected.json'),ROOT,file=True)
        require(stat.S_IMODE(plan_path.stat().st_mode)==0o600 and plan_path.stat().st_size<=16*1024**2)
        with plan_path.open('rb') as stream:plan_bytes=stream.read(16*1024**2+1)
        require(len(plan_bytes)<=16*1024**2)
        experiment_plan=raw._json(plan_bytes)
        require(sha256(canonical(experiment_plan)).hexdigest()==result['experiment_plan_sha256'])
        require(plan_bytes==canonical(experiment_plan)+b'\n')
        pin(plan_path,sha256(plan_bytes).hexdigest());del plan_bytes
        manifest=read(directory/'experimental-arms/manifest.json',result['experimental_manifest_file_sha256'])
        require(manifest['plan_sha256']==result['experiment_plan_sha256'] and manifest['total_committed_updates']==80)
        require(manifest['state_file_sha256']==result['experimental_state_file_sha256'])
        pin(directory/'experimental-arms/experimental.pt',result['experimental_state_file_sha256'],4*1024**3)
        stdout=pin(ROOT/'jobs'/(NAME+'-source-stdout.txt'),LOG_SHA)
        acknowledgments=progress_acknowledgments(stdout)
        for row in acknowledgments:
            count=row['completed_update_events']
            cursor=read(directory/'experimental-progress'/f'group-{count:04d}'/'manifest.json',row['snapshot_manifest_sha256'])['cursor']
            require(cursor=={'arm':'control' if count<=40 else 'intervention',
                             'next_group':count if count<=40 else count-40,'total_committed_updates':count})
        original=seal(read(ROOT/'jobs/rayan-cumulative-learning-r3-20261015.json',result['source_summary_sha256']))
        source_plan=read(ROOT/'rayan-personality/private-inputs/rayan-cumulative-learning-r3-20261015/owner-cumulative-run.protected.json',result['source_plan_sha256'])
        first_plan=read(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json',
                        '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
        all_cases={};allocation=None
        for name in ('original','conditional','broader','voice'):
            ref=source_plan['packets'][name]
            folder=Path(first_plan['packet_directory'] if name=='original' else ref['directory'])
            index=read(folder/'packet-index.protected.json',ref['index_sha256'])
            cases=[ReviewedTeachingCase.from_mapping(v) for v in lines(folder/'reviewed-teaching-cases.protected.jsonl',index['files']['reviewed-teaching-cases.protected.jsonl'])]
            require(len(cases)==len({c.case_id for c in cases}) and {c.case_id:c.sha256 for c in cases}==index['case_sha256'])
            all_cases.update({(name,c.case_id):c for c in cases})
            if name=='broader':allocation=read(folder/'construction-allocation.protected.json',index['files']['construction-allocation.protected.json'])
        old_output=ROOT/'rayan-personality/private-compile/rayan-cumulative-learning-r3-577714'
        metadata=original['successor_full_governed_responses']
        base_objectives=[r for r in lines(old_output/'successor-all-reviewed-objectives.protected.jsonl',metadata['all_reviewed_objective_file_sha256']) if r['phase']=='n2']
        require(len(base_objectives)==216 and {r['batch_sha256'] for r in base_objectives}==set(experiment_plan['batch_sha256']))
        selected=[]
        for row in base_objectives:
            case=all_cases[(row['packet'],row['case_id'])]
            require(case.sha256==row['case_sha256'] and case.phase=='n2' and case.split=='TRAIN')
            selected.append(case)
        broad=[all_cases[(r['packet'],r['case_id'])] for r in base_objectives if r['packet']=='broader']
        require(len(broad)==80);by_document={c.document.sha256:c for c in broad}
        pairs=source.source_pairs(broad,allocation)
        output=ROOT/'rayan-personality/private-compile'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
        output.mkdir(mode=0o700,exist_ok=False)
        private={};metrics={};decisions={};bindings={};frozen={};objective_rows={};broad_bindings={}
        target_details,target_summary=helper.target_compatibility(selected)
        private['target_compatibility']=write(output/'target-compatibility.protected.json',target_details)
        for tag in ('baseline','control','intervention'):
            live.verify_active_namespace(namespace,isolation)
            if tag=='baseline':
                rows=lines(old_output/'successor-governed-responses.protected.jsonl',metadata['file_sha256'])
            else:
                arm=result['arms'][tag]
                rows=lines(directory/(tag+'-candidate-native-packets.protected.jsonl'),arm['response_file_sha256'])
                objectives=lines(directory/(tag+'-all-n2-objectives.protected.jsonl'),arm['objective_file_sha256'])
                require(analyzer.compare_objectives(base_objectives,objectives)==arm['objective_comparison'])
                objective_rows[tag]=objectives
                expected_events=manifest['arms'][tag]['events']
                require(len(expected_events)==40 and [e['batch_sha256'] for e in expected_events]==experiment_plan['matched_groups'])
                details,metrics[tag]=helper.trajectory(expected_events,objectives,
                    correction_batches=set(experiment_plan['correction_sha256']),control=tag=='control')
                private[tag+'_trajectory']=write(output/(tag+'-trajectory.protected.json'),details)
            require(len(rows)==len({(r['packet'],r['document_id']) for r in rows})==236)
            current={}
            for row in rows:
                key=(row['packet'],row['document_id']);response=row['response'] if tag=='baseline' else row
                decision=response['decision']
                if tag=='baseline':
                    seal(response);native=seal(response['preference_selection'])
                    require(native['model_state_sha256']==experiment_plan['model_state_sha256'] and native['packet_sha256']==response['packet_sha256'])
                    doc_sha=native['document_sha256']
                    bindings[key]=(doc_sha,native['feature_binding_sha256'],native['closed_session_sha256'])
                else:
                    require(response['arm']==tag and response['model_state_sha256']==result['arms'][tag]['model_state_sha256'])
                    doc_sha=response['document_sha256']
                    require(bindings[key]==(doc_sha,response['frame_binding']['binding_sha256'],response['closed_session_sha256']))
                    require(response['qualification']=='UNQUALIFIED' and all(response[k] is False for k in ('native_checkpoint','native_consumer_admitted','acceptance_authority')))
                unchanged={'sources':decision['sources'],'concept_records':decision['concepts'],
                           'conditional':decision['conditional_concepts'],
                           **{k:decision['native_latent'][k] for k in ('concepts','concept_mask','views','view_masks')}}
                digest=sha256(canonical(unchanged)).hexdigest()
                if tag=='baseline':frozen[key]=digest
                else:require(frozen[key]==digest)
                if row['packet']=='broader':
                    require(doc_sha in by_document);current[by_document[doc_sha].case_id]=decision
                    broad_bindings[doc_sha]=bindings[key][1]
            require(len(current)==80);decisions[tag]=current
            if tag!='baseline':require(source.paired_region_comparison(pairs,decisions['baseline'],current)==result['arms'][tag]['broader_pair_comparison'])
            print(json.dumps({'stage':'saved_natural_arm_verified','arm':tag,'documents':236,'new_forwards':0}),flush=True)
        transitions={}
        for before,after in (('baseline','control'),('baseline','intervention'),('control','intervention')):
            name=before+'_to_'+after
            details,transitions[name]=helper.transitions(broad,decisions[before],decisions[after])
            private[name]=write(output/(name+'.protected.json'),details)
        for tag in ('control','intervention'):
            arm=result['arms'][tag]
            rows=lines(directory/(tag+'-corrected-native-packets.protected.jsonl'),arm['correction_assessment']['private_file_sha256'])
            require(len(rows)==240)
            by_sha={c.sha256:c for c in broad}
            for subset in ('full','first_half','second_half'):
                selected_rows=[r for r in rows if r['subset']==subset]
                require(len(selected_rows)==len({r['case_sha256'] for r in selected_rows})==80)
                patched={};objectives=[]
                for row in selected_rows:
                    require(row['arm']==tag and row['training_targets_supplied'] is True)
                    case=by_sha[row['case_sha256']];obj=row['objectives']
                    require(obj['case_sha256']==case.sha256 and obj['batch_sha256'] in experiment_plan['correction_sha256'])
                    require(row['correction_record_sha256']==experiment_plan['correction_sha256'][obj['batch_sha256']])
                    require(row['frame_binding']['binding_sha256']==broad_bindings[case.document.sha256])
                    patched[case.case_id]=row['decision'];objectives.append(obj)
                expected=arm['correction_assessment']['subsets'][subset]
                require(source.paired_region_comparison(pairs,decisions[tag],patched)==expected['broader_pair_comparison_with_own_natural'])
                natural=[r for r in objective_rows[tag] if r['packet']=='broader']
                require(analyzer.compare_objectives(natural,objectives)==expected['objective_comparison_with_own_natural'])
                name=tag+'_to_'+subset
                details,transitions[name]=helper.transitions(broad,decisions[tag],patched)
                private[name]=write(output/(name+'.protected.json'),details)
        live.verify_active_namespace(namespace,isolation)
        for value in held:value.verify()
        namespace._clean_code(code,REV)
        summary={'schema':'alice-personality-conditional-saved-audit-v1','job_id':int(os.environ['SLURM_JOB_ID']),
            'source_job_id':579741,'source_summary_file_sha256':SUMMARY_SHA,'source_receipt_sha256':result['receipt_sha256'],
            'owner':'mxrayan','worker_sha256':worker_sha,'dependency_sha256':{k:v[1] for k,v in DEPENDENCIES.items()},
            'isolation':isolation,'consumed_file_hashes_verified_at_open_and_close':len(held),
            'all_80_published_progress_manifests_verified':True,'final_export_bytes_verified':True,
            'historical_snapshot_tensor_bodies_reloaded':False,'all_216_objectives_reproduced_each_arm':True,
            'all_236_saved_packets_bound_each_arm':True,'all_480_corrected_probes_reproduced':True,
            'target_compatibility':target_summary,'transitions':transitions,'training_trajectory':metrics,
            'private_detail_file_sha256':private,'new_model_forwards':0,'new_private_gradient_steps':0,
            'new_targets_created':False,'source_acceptance_authority':False,'private_payload_exported':False,
            'qualification':'UNQUALIFIED','n0_approved':False,'personality_approved':False}
        summary['receipt_sha256']=sha256(canonical(summary)).hexdigest()
        write(ROOT/'jobs'/(NAME+'.json'),summary)
        print(json.dumps({'stage':'complete_saved_audit','receipt_sha256':summary['receipt_sha256'],'new_forwards':0}),flush=True)


if __name__=='__main__':
    guard=Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    require(sha256(guard.read_bytes()).hexdigest()==GUARD)
    namespace=module('saved_audit_namespace',guard)
    try:run(namespace)
    except BaseException as error:
        report={'schema':'alice-personality-saved-audit-failure-v1','job_id':os.environ.get('SLURM_JOB_ID'),
                'error_type':type(error).__name__,'public_code_locations':[{'file':Path(f.filename).name,'line':f.lineno}
                for f in traceback.extract_tb(error.__traceback__)],'private_payload_exported':False}
        print(json.dumps(report),flush=True)
        raise SystemExit(2) from None
