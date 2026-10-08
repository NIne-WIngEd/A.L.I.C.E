"""Matched one-step diagnostic of the unchanged, admitted N1 path in owner P2.

Only fixed aggregate timings and checkpoint seals leave the protected process.
This diagnostic cannot qualify or complete the connected personality model.
"""
import cProfile
import importlib.util
import json
import os
from pathlib import Path
from hashlib import sha256
import pstats
import sys
import time
from contextlib import ExitStack

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
PLAN = ROOT/'rayan-personality/private-inputs/rayan-primary-governing-continuation-20261006/owner-connected-run.protected.json'
PLAN_SHA = '1af51526d19901b60debd9a063c0993a149bdc869faecdcd6df80fe380c3e26c'
INVENTORY_SHA = 'd0e6cf55f513ff8ccd7077052bfb3d6935b3622b50b5fdd265317f755e48c97b'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def emit(stage, **fields):
    print(json.dumps({'stage':stage, **fields},sort_keys=True),flush=True)

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def profile(namespace, threads):
    isolation=namespace._p2_guard()  # FIRST protected path/stat/hash/read.
    os.umask(0o077)
    sys.dont_write_bytecode=True
    code=Path(os.environ['PERSONALITY_REPO_ROOT'])
    namespace._clean_code(code,REV)
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest()==GUARD
    output=Path(os.environ['PERSONALITY_PROFILE_ROOT'])
    assert output==ROOT/'rayan-personality/private-compile'/f'rayan-timeout-profile-{os.environ["SLURM_JOB_ID"]}-threads-{threads:02d}'
    namespace._bounded_path(str(output.parent),ROOT,file=False)
    assert not output.exists() and not output.is_symlink()
    output.mkdir(mode=0o700)
    old=namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-connected-personality-576893'),ROOT,file=False)
    overlay=namespace._bounded_path(str(ROOT/'rayan-personality/runs/public-forward-576598/processor-deps'),ROOT,file=False)
    sys.path[:0]=[str(overlay),str(code)]
    import torch
    assert torch.__version__=='2.7.1+cu118'
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import raw_inference_lineage as raw
    from src.alice_personality.identity.codec import CognitiveFrameDocument, fingerprint
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.model import IdentityModel
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    from src.alice_personality.teaching.materialize import materialize_case
    from src.alice_personality.teaching.contracts import TrainingRecipe
    from src.alice_personality.teaching.trainer import ArtifactPin,TeachingAdmission,IdentityTrainer,current_code_hashes,model_fingerprint
    helper=module('unchanged_connected_helper',code/'scripts/eipm/gemma_n0/construct_personality.py')
    measurements={}
    def measured(label, call):
        emit('operation_started',operation=label,threads=threads)
        profiler=cProfile.Profile()
        started=time.monotonic()
        profiler.enable()
        try:
            return call()
        finally:
            profiler.disable()
            elapsed=time.monotonic()-started
            # Functions and aggregate times only; no argument, pointer, source or frame text.
            allowed={'tensor_digest','fingerprint','_hash','_read','produce','validate','verify',
                '_production','materialize_case','verify_materialized_batch','forward','backward',
                'step','_loss','_check_batch','verify_prepared','verify_compiled','_check_optimizer'}
            rows=[]
            for (filename,line,name),(primitive,calls,self_seconds,total_seconds,callers) in pstats.Stats(profiler).stats.items():
                if name in allowed and (filename.startswith(str(code)) or '/torch/' in filename):
                    rows.append({'module':Path(filename).name,'function':name,'calls':calls,
                        'self_seconds':self_seconds,'cumulative_seconds':total_seconds})
            measurements[label]={'wall_seconds':elapsed,'calls':sorted(rows,key=lambda r:-r['cumulative_seconds'])[:25]}
            emit('operation_completed',operation=label,threads=threads,**measurements[label])
    with ExitStack() as stack:
        held=[]
        def pin(path,digest,limit=64*1024**2):
            bounded=namespace._bounded_path(str(path),ROOT,file=True)
            value=stack.enter_context(_PinnedFile(bounded,digest,limit));held.append(value)
            value.stream.seek(0)
            return raw._json(value.stream.read())
        plan=pin(PLAN,PLAN_SHA)
        packet=namespace._bounded_path(plan['packet_directory'],ROOT,file=False)
        index=pin(packet/'packet-index.protected.json',plan['packet_index_sha256'])
        inventory=raw._json((ROOT/'jobs/rayan-connected-576893-inventory-20261008.json').read_bytes())
        assert inventory['receipt_sha256']==INVENTORY_SHA
        assert sha256(canonical({k:v for k,v in inventory.items() if k!='receipt_sha256'})).hexdigest()==INVENTORY_SHA
        docs_path=packet/'target-free-documents.protected.jsonl'
        case_path=packet/'reviewed-teaching-cases.protected.jsonl'
        for path in (docs_path,case_path):
            bounded=namespace._bounded_path(str(path),ROOT,file=True)
            held.append(stack.enter_context(_PinnedFile(bounded,index['files'][path.name],64*1024**2)))
            held[-1].stream.seek(0)
        documents={}
        for line in held[-2].stream:
            row=raw._json(line)
            document=CognitiveFrameDocument.from_mapping(row['document'])
            assert index['document_sha256'][row['document_id']]==document.sha256
            documents[row['document_id']]=document
        cases=[ReviewedTeachingCase.from_mapping(raw._json(line)) for line in held[-1].stream]
        assert {case.case_id:case.sha256 for case in cases}==index['case_sha256']
        first=next(case for case in cases if case.phase=='n1')
        artifacts=index['artifact_sha256']
        store,config,session,bindings,cost=helper.restore_completed_features(plan['completed_features'],
            documents=documents,source_package_sha256=artifacts['source_package'],
            prepared_receipt_file_sha256=artifacts['prepared_receipt'],
            read_pinned_json=lambda path,digest:pin(path,digest),
            bounded_directory=lambda path:namespace._bounded_path(path,ROOT,file=False))
        allocation=pin(old/'construction-allocation.protected.json',inventory['metadata_files']['construction-allocation.protected.json']['file_sha256'])
        assert allocation==pin(packet/'construction-allocation.protected.json',index['files']['construction-allocation.protected.json'])
        reviewed_pin=next(row['file_sha256'] for row in inventory['phase_artifacts']['n1']['entries'] if row['name']=='reviewed-batches.protected.json')
        reviewed=pin(old/'n1/reviewed-batches.protected.json',reviewed_pin)
        # Current authorization is part of the exact original immutable failed run.
        auth_path=namespace._bounded_path(str(old/'n1/current-authorization.protected.json'),ROOT,file=True)
        auth_sha=helper.file_hash(auth_path)
        auth=pin(auth_path,auth_sha)
        assert auth['owner_run_plan_sha256']==PLAN_SHA and auth['primary_packet_index_sha256']==plan['packet_index_sha256']
        assert auth['private_gradient_authorized'] is True and auth['acceptance_authority'] is False
        arguments=dict(plan['recipes']['n1'])
        if 'betas' in arguments:arguments['betas']=tuple(arguments['betas'])
        recipe=TrainingRecipe(**arguments)
        recipe.validate()
        assert auth['recipe_sha256']==recipe.fingerprint() and auth['code_sha256']==current_code_hashes()
        replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=session['receipt_sha256'])
        produced=measured('closed_full_feature_replay',lambda:replay.produce(first.document))
        batch=measured('reviewed_case_materialization',lambda:materialize_case(first,produced,session,config,
            expected_case_sha256=first.sha256,expected_artifact_sha256=artifacts,family_splits=allocation).batch)
        assert batch.fingerprint()==reviewed['batch_order']['TRAIN'][0]
        torch.manual_seed(plan['model_initialization_seed'])
        model=IdentityModel(config)
        assert model_fingerprint(model)==auth['model_initial_state_sha256']
        admitted={'source_package':ArtifactPin(Path(helper.BASE)/'rayan-personality/private-inputs/ALICE_EIPM_CURATED_FRONTIER_v2-3867ff04.zip',artifacts['source_package']),
            'compiled_receipt':ArtifactPin(Path(helper.COMPILED)/'compile_receipt.json',artifacts['compiled_receipt']),
            'prepared_receipt':ArtifactPin(Path(helper.PREPARED),artifacts['prepared_receipt']),
            'reviewed_targets':ArtifactPin(old/'n1/reviewed-batches.protected.json',reviewed_pin),
            'source_families':ArtifactPin(old/'construction-allocation.protected.json',inventory['metadata_files']['construction-allocation.protected.json']['file_sha256']),
            'authorization':ArtifactPin(auth_path,auth_sha)}
        admission=TeachingAdmission(admitted,auth['model_config_sha256'],auth['model_initial_state_sha256'],recipe.fingerprint(),current_code_hashes())
        trainer=measured('actual_deep_admission',lambda:IdentityTrainer(model,recipe,admission,first_batch=batch))
        event=measured('actual_admitted_optimizer_step',lambda:trainer.step(batch))
        manifest=measured('actual_checkpoint_save',lambda:trainer.save_checkpoint(output/'one-step-checkpoint'))
        for value in held:value.verify()
        result={'schema':'alice-personality-matched-step-profile-v1','owner':'mxrayan','failed_job':576893,
            'scientific_revision':REV,'owner_plan_sha256':PLAN_SHA,'original_inventory_sha256':INVENTORY_SHA,
            'authorization_file_sha256':auth_sha,'original_reviewed_batches_file_sha256':reviewed_pin,
            'torch_version':str(torch.__version__),'intraop_threads':threads,'interop_threads':torch.get_num_interop_threads(),
            'complete_features':True,'original_first_batch_matched':True,'actual_optimizer_steps':trainer.steps,
            'checkpoint_manifest_sha256':manifest['manifest_sha256'],'event_sha256':sha256(canonical(event)).hexdigest(),
            'measurements':measurements,'isolation':isolation,'n0_approved':False,'qualification':'UNQUALIFIED',
            'diagnostic_only':True,'full_construction_complete':False,'private_payload_exported':False}
        result['receipt_sha256']=sha256(canonical(result)).hexdigest()
        destination=ROOT/'jobs'/f'rayan-connected-timeout-profile-20261008-{os.environ["SLURM_JOB_ID"]}-threads-{threads:02d}.json'
        with destination.open('xb') as stream:stream.write(canonical(result)+b'\n')
        emit('matched_step_profile_retained',threads=threads,receipt_sha256=result['receipt_sha256'],actual_optimizer_steps=trainer.steps)

code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
namespace=module('owner_namespace_profile',guard)
threads=int(sys.argv[1])
assert threads in (1,16)
try:
    profile(namespace,threads)
except Exception as error:
    # No private exception text or locals enter console logs.
    import traceback
    frames=[{'module':Path(frame.filename).name,'line':frame.lineno,'function':frame.name}
            for frame in traceback.extract_tb(error.__traceback__)]
    emit('PROFILE_REFUSED',exception_type=type(error).__name__,code_frames=frames)
    raise SystemExit(2)
