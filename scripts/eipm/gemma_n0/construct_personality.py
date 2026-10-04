"""Owner-isolated connected construction from externally pinned primary cases.

No teacher generation, source promotion, Gemma weight mutation, FINAL selection,
N0 approval, spoken approval or automatic N3 gold. The existing namespace
guard must succeed before private paths are resolved, profiled or opened.
"""
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import gc
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

BASE='/homes/01/mxrayan/rayan-compute'
GUARD_SHA='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
SOURCE_CONTEXT_SHA='f88a5718b8cdd9c0ebd70c1334cbabd40a198937e96179504970dbf5f566452a'
COMPILE_SEAL='e164cf9e175f1710ea69adacf1b1272213f2368d696e53af86b3e1542d39d16b'
SOURCE_CONTEXT=BASE+'/rayan-personality/private-compile/rayan-source-complete-cases-576679/source-complete-case-authoring.protected.jsonl'
COMPILED=BASE+'/rayan-personality/private-compile/rayan-substrate-construction-576673/compiled-substrate'
PREPARED=BASE+'/rayan-personality/runs/preparation-576597/personality-prepared.json'
OVERLAY=BASE+'/rayan-personality/runs/public-forward-576598/processor-deps'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def refuse():
    raise ValueError('connected personality construction refused')


def file_hash(path):
    before=path.stat();digest=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):refuse()
    return digest.hexdigest()


def document_text_banks(document):
    """Exact current producer rendering; covered against its real tokenizer."""
    document.validate()
    banks={'query':[f.semantic_text() for f in document.fields],
        'sources':[f'Evidence class: {e.record.provenance_class}\n{e.text}' for e in document.sources],
        'concepts':[e.text for e in document.concepts],'candidates':[e.text for e in document.candidates],
        'relations':[e.text for e in document.relations]}
    for family,labels in document.labels.items():
        banks['label:'+family]=[f'{s.name}\n{s.description}\nUnit: {s.unit}'+
            (f'\nRange: {s.minimum} to {s.maximum}' if family=='voice' else '') for s in labels]
    return banks


def token_cost_profile(documents,tokenizer,*,layers,width,max_tokens):
    """Actual token counts; retained values exact, live memory only an estimate."""
    banks=[document_text_banks(d) for d in documents]
    lengths={}
    for text in sorted({text for doc in banks for bank in doc.values() for text in bank}):
        encoded=tokenizer([text],padding=False,truncation=False,add_special_tokens=True,
            return_attention_mask=True,return_tensors='pt')
        if set(encoded)!={'input_ids','attention_mask'}:refuse()
        ids,mask=encoded['input_ids'],encoded['attention_mask']
        if ids.ndim!=2 or ids.shape[0]!=1 or tuple(mask.shape)!=tuple(ids.shape) or not bool(mask.bool().all()):refuse()
        size=ids.shape[1]
        if size<1 or size>max_tokens:refuse()
        lengths[text]=size
    value_bytes=sum(lengths.values())*layers*width*2  # Exact BF16 value payload, before serialization overhead.
    frame_bytes=[sum(max(1,len(bank))*max([lengths[t] for t in bank] or [1])*layers*width*2 for bank in doc.values()) for doc in banks]
    return {'schema':'alice-personality-full-feature-token-cost-v1','document_count':len(documents),
        'unique_entries':len(lengths),'unique_source_tokens':sum(lengths.values()),'largest_entry_tokens':max(lengths.values()),
        'state_value_bytes':value_bytes,'serialization_overhead_budget_bytes':len(lengths)*8192,
        'largest_padded_frame_value_bytes':max(frame_bytes),'sum_unreused_padded_frame_value_bytes':sum(frame_bytes),
        'measured_peak_memory':False,'source_geometry':[layers,width],'feature_dtype':'bfloat16',
        'truncation':False,'qualification':'UNQUALIFIED','acceptance_authority':False}


def verify_reference_families(document,allocation,compiled_families,original_rows,protocol):
    """Cross-check pinned original compiler membership; no new source aliases."""
    families={r['family_id']:r for r in compiled_families}
    if len(families)!=len(compiled_families):refuse()
    refs=set(allocation['reference_family_ids'])
    for family in refs:
        if family not in families or families[family]['split']!='train' or allocation['family_splits'].get(family)!='TRAIN':refuse()
    for entry in document.sources:
        record=entry.record
        if record.source_family_id not in refs:
            if record.source_kind not in {'HOST','RELATIONSHIP','ASELF'} or record.identity_core_allowed \
                    or record.historical_truth_allowed or record.alice_lived_memory or record.autobiographical_recall_allowed:refuse()
            continue
        row=original_rows.get(record.record_id)
        if row is None:refuse()
        work=row['draft_workitem'];original=row['original_source_record'];decl=work['source_declarations']
        body=sha256(canonical(original)).hexdigest()
        if (work['record_id']!=record.record_id or work['source_family_id']!=record.source_family_id
                or record.source_kind!=row['source_kind_for_future_document'] or work['original_split']!='train'
                or record.record_id not in families[record.source_family_id]['record_ids']
                or record.provenance_class!=original['provenance_class']
                or body!=row['original_source_record_sha256'] or protocol['source_body_sha256'].get(record.record_id)!=body
                or protocol['source_render_sha256'].get(record.record_id)!=sha256(entry.text.encode()).hexdigest()
                or record.alice_lived_memory or record.autobiographical_recall_allowed
                or record.runtime_behavioral_prior_allowed!=decl['runtime_behavioral_prior_allowed']
                or (record.historical_truth_allowed and not decl['historical_truth_allowed'])):refuse()
        if record.source_kind=='E0' and record.identity_core_allowed:
            mask=original['loss_mask']
            if ('source_person:elaina' not in original['claim_subject_partitions'] or mask.get('excluded_from_identity_loss',False)
                    or not any(mask.get(x,False) for x in ('conditional_identity','direct_identity','identity_conditional_loss_enabled','identity_core_loss_enabled'))):refuse()


def inventory(directory):
    values={}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():refuse()
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':values[p.relative_to(directory).as_posix()]=file_hash(p)
    if not values:refuse()
    return sha256(canonical(values)).hexdigest()


def write(path,value):
    with path.open('xb') as f:f.write(canonical(value)+b'\n')
    return file_hash(path)


def run(namespace):
    isolation=namespace._p2_guard()  # FIRST: no private path/stat/hash/profile before this.
    os.umask(0o077);sys.dont_write_bytecode=True
    root=Path(BASE)
    if root.is_symlink() or root.resolve(strict=True)!=root or stat.S_IMODE(root.stat().st_mode)!=0o700:refuse()
    revision=os.environ['PERSONALITY_EXPECTED_REVISION']
    code=namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'],root,file=False)
    namespace._clean_code(code,revision)
    output=namespace._bounded_path(os.environ['PERSONALITY_RUN_ROOT'],root,file=False)
    if any(output.iterdir()) or stat.S_IMODE(output.stat().st_mode)!=0o700:refuse()
    helper=Path(__file__).resolve()
    if not helper.is_relative_to(code) or file_hash(helper)!=os.environ['PERSONALITY_CONSTRUCTION_HELPER_SHA256'] \
            or file_hash(Path(namespace.__file__))!=GUARD_SHA:refuse()
    overlay=namespace._bounded_path(OVERLAY,root,file=False);overlay_sha=inventory(overlay)
    sys.path[:0]=[str(overlay),str(code)]
    import PIL,torch,torchvision,transformers
    if (PIL.__version__!='11.3.0' or torch.__version__!='2.7.1+cu118' or torchvision.__version__!='0.22.1+cu118'
            or transformers.__version__!='5.17.0' or not Path(PIL.__file__).resolve().is_relative_to(overlay)
            or not Path(torchvision.__file__).resolve().is_relative_to(overlay)):refuse()
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import compiler,raw_inference_lineage as raw
    from src.alice_personality.gemma_n0.preparation import verify_prepared
    from src.alice_personality.identity.codec import CognitiveFrameDocument,decision_to_mapping,fingerprint
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.feature_store import FrozenFeatureStore
    from src.alice_personality.identity.contracts import IdentityModelConfig
    from src.alice_personality.identity.model import IdentityModel
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    from src.alice_personality.teaching.materialize import materialize_case
    from src.alice_personality.teaching.construction import ConnectedConstruction
    from src.alice_personality.teaching.contracts import TrainingRecipe,AUTH_SCHEMA,REVIEW_SCHEMA,TARGET_SCHEMA
    from src.alice_personality.teaching.lineage import validate_allocation
    from src.alice_personality.teaching.trainer import ArtifactPin,TeachingAdmission,current_code_hashes,model_fingerprint
    with ExitStack() as stack:
        pins=[]
        def pin(value,digest,limit=64*1024*1024):
            p=namespace._bounded_path(value,root,file=True)
            if code in p.parents or output in p.parents or stat.S_IMODE(p.stat().st_mode)!=0o600:refuse()
            value=stack.enter_context(_PinnedFile(p,digest,limit));pins.append(value);return value
        def load(value):value.stream.seek(0);return raw._json(value.stream.read())
        def rows(value):
            value.stream.seek(0);out=[]
            for line in value.stream:
                if not line.strip() or len(line)>raw.MAX_JSONL_LINE_BYTES:refuse()
                out.append(raw._json(line))
            return out
        plan=load(pin(os.environ['PERSONALITY_CONSTRUCTION_PLAN'],os.environ['PERSONALITY_CONSTRUCTION_PLAN_SHA256']))
        if (plan.get('schema')!='alice-personality-owner-connected-run-v1' or plan.get('owner')!='mxrayan'
                or plan.get('purpose')!='development_construction' or plan.get('private_gradient_authorized') is not True
                or plan.get('n3_targets_authorized') is not False or plan.get('acceptance_authority') is not False):refuse()
        torch.set_num_threads(plan['cpu_threads'])
        torch.set_num_interop_threads(1)
        if plan['cpu_threads']!=int(os.environ['SLURM_CPUS_PER_TASK']):refuse()
        packet=namespace._bounded_path(plan['packet_directory'],root,file=False)
        index=load(pin(str(packet/'packet-index.protected.json'),plan['packet_index_sha256']))
        inputs={name:pin(str(packet/name),digest) for name,digest in index['files'].items()}
        artifacts=index['artifact_sha256']
        protocol=load(pin(str(packet/'annotation-protocol.protected.json'),artifacts['annotation_protocol']))
        review=load(pin(str(packet/'author-review.protected.json'),artifacts['review_record']))
        pin(str(packet/'teacher-spec.protected.json'),protocol['teacher_spec_sha256'])
        if protocol.get('other_teacher_used') is not False or protocol.get('source_complete_authoring_sha256')!=SOURCE_CONTEXT_SHA \
                or review.get('annotation_protocol_sha256')!=artifacts['annotation_protocol'] \
                or review.get('development_targets_reviewed') is not True or review.get('owner_acceptance_recorded') is not False:refuse()
        compiled=namespace._bounded_path(COMPILED,root,file=False)
        compile_receipt=pin(str(compiled/'compile_receipt.json'),artifacts['compiled_receipt'])
        parent=compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts['source_package'],expected_receipt_sha256=COMPILE_SEAL)
        source=pin(os.environ['PRIVATE_PACKAGE_PATH'],artifacts['source_package'],raw.MAX_ARCHIVE_BYTES)
        original_list=rows(pin(SOURCE_CONTEXT,SOURCE_CONTEXT_SHA))
        originals={r['draft_workitem']['record_id']:r for r in original_list}
        if len(originals)!=len(original_list):refuse()
        # Actual verified compiler output, not the newly authored allocation.
        compiled_families=[raw._json(line) for line in (compiled/'split_families.jsonl').read_bytes().splitlines()]
        allocation=load(inputs['construction-allocation.protected.json']);validate_allocation(allocation)
        documents={};document_splits={}
        for row in rows(inputs['target-free-documents.protected.jsonl']):
            if set(row)!={'document_id','split','document'} or row['document_id'] in documents:refuse()
            d=CognitiveFrameDocument.from_mapping(row['document']);name=row['document_id']
            if index['document_sha256'].get(name)!=d.sha256:refuse()
            verify_reference_families(d,allocation,compiled_families,originals,protocol)
            documents[name]=d;document_splits[name]=row['split']
        cases=[ReviewedTeachingCase.from_mapping(row) for row in rows(inputs['reviewed-teaching-cases.protected.jsonl'])]
        if len({c.case_id for c in cases})!=len(cases) or {c.case_id:c.sha256 for c in cases}!=index['case_sha256'] \
                or len(documents)!=len(index['document_sha256']):refuse()
        doc_by_sha={d.sha256:d for d in documents.values()}
        for c in cases:
            if c.phase not in {'n1','n2'} or c.document.sha256 not in doc_by_sha or dict(c.artifact_sha256)!=artifacts \
                    or c.source_class!='governed_identity_targets' or c.split!='TRAIN':refuse()
        phase_counts={phase:sum(c.phase==phase for c in cases) for phase in ('n1','n2','calibration')}
        if phase_counts!=index['phase_counts'] or any(not phase_counts[p] for p in ('n1','n2')):refuse()
        prepared=pin(PREPARED,artifacts['prepared_receipt']);preparation=verify_prepared(prepared.path)
        if preparation['runtime']['dtype']!='bfloat16':refuse()
        processor=transformers.AutoProcessor.from_pretrained(preparation['snapshot_path'],local_files_only=True,trust_remote_code=False)
        geometry=preparation['model_geometry']
        profile=token_cost_profile(list(documents.values()),processor.tokenizer,layers=geometry['hidden_state_count'],
            width=geometry['hidden_size'],max_tokens=geometry['max_position_embeddings'])
        # Input float conversion and retained backward intermediates are explicitly
        # budgeted; this is a conservative operating estimate, not a measurement.
        estimate=48*1024**3+profile['largest_padded_frame_value_bytes']*12
        profile['estimated_live_memory_budget_bytes']=estimate
        profile['requested_memory_bytes']=plan['memory_budget_bytes']
        write(output/'token-cost-profile.json',profile)
        if (estimate>plan['memory_budget_bytes'] or profile['state_value_bytes']+profile['serialization_overhead_budget_bytes']>plan['feature_disk_budget_bytes']
                or os.statvfs(output).f_bavail*os.statvfs(output).f_frsize<plan['feature_disk_budget_bytes']+plan['checkpoint_disk_budget_bytes']):refuse()
        del processor;gc.collect()
        store=FrozenFeatureStore.create(output/'feature-store',source_package_sha256=artifacts['source_package'])
        producer=FrozenFrameProducer.load(prepared.path,expected_file_sha256=artifacts['prepared_receipt'],device='cpu',feature_store=store)
        config=producer.config
        bindings={}
        for number,(name,document) in enumerate(documents.items(),1):
            produced=producer.produce(document);bindings[name]=produced.binding;del produced;gc.collect()
            print(json.dumps({'stage':'source_closed_frame_production','completed_documents':number,'total_documents':len(documents)}),flush=True)
        session=producer.close_and_verify()
        write(output/'feature-session.protected.json',session)
        write(output/'frame-bindings.protected.json',bindings)
        write(output/'model-config.protected.json',asdict(config))
        del producer;gc.collect()  # Frozen Gemma is not retained during learned-stage optimization.
        replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=session['receipt_sha256'])
        torch.manual_seed(plan['model_initialization_seed'])
        model=IdentityModel(config);construction=ConnectedConstruction(model)
        allocation_path=output/'construction-allocation.protected.json';write(allocation_path,allocation)
        phase_results={}
        for phase in ('n1','n2'):
            selected=[c for c in cases if c.phase==phase]
            phase_dir=output/phase;phase_dir.mkdir(mode=0o700)
            batch_cases={};reviewed={'schema':REVIEW_SCHEMA,'target_schema':TARGET_SCHEMA,'reviewed':True,
                'source_class':'governed_identity_targets','reviewed_batches':{},'batch_order':{'TRAIN':[]},
                'reviewed_case_artifact_sha256':{c.sha256:dict(c.artifact_sha256) for c in selected}}
            def materialize(case):
                produced=replay.produce(case.document)
                return materialize_case(case,produced,session,config,expected_case_sha256=case.sha256,
                    expected_artifact_sha256=artifacts,family_splits=allocation).batch
            for c in selected:
                batch=materialize(c);digest=batch.fingerprint();batch_cases[digest]=c
                reviewed['reviewed_batches'][digest]={'split':batch.split,'case_ids':list(batch.case_ids),
                    'source_family_ids':[list(x) for x in batch.source_family_ids],'construction_lineage_sha256':fingerprint(batch.lineage)}
                reviewed['batch_order']['TRAIN'].append(digest);del batch;gc.collect()
            if len(batch_cases)!=len(selected):refuse()
            arguments=dict(plan['recipes'][phase])
            if 'betas' in arguments:arguments['betas']=tuple(arguments['betas'])
            recipe=TrainingRecipe(**arguments);recipe.validate()
            if recipe.phase!=phase or recipe.adapt_n1:refuse()
            target_path=phase_dir/'reviewed-batches.protected.json';write(target_path,reviewed)
            admitted={'source_package':ArtifactPin(source.path,artifacts['source_package']),
                'compiled_receipt':ArtifactPin(compile_receipt.path,artifacts['compiled_receipt']),
                'prepared_receipt':ArtifactPin(prepared.path,artifacts['prepared_receipt']),
                'reviewed_targets':ArtifactPin(target_path,file_hash(target_path)),
                'source_families':ArtifactPin(allocation_path,file_hash(allocation_path))}
            config_sha=sha256(canonical(asdict(config))).hexdigest();initial=model_fingerprint(model);code_sha=current_code_hashes()
            auth={'schema':AUTH_SCHEMA,'authorization_id':plan['authorization_id']+':'+phase,'phase':phase,
                'purpose':'development_construction','source_class':'governed_identity_targets','private_gradient_authorized':True,
                'mechanical_steps_authorized':False,'artifact_file_sha256':{k:v.sha256 for k,v in admitted.items()},
                'model_config_sha256':config_sha,'model_initial_state_sha256':initial,'recipe_sha256':recipe.fingerprint(),'code_sha256':code_sha,
                'owner_run_plan_sha256':os.environ['PERSONALITY_CONSTRUCTION_PLAN_SHA256'],'primary_packet_index_sha256':plan['packet_index_sha256'],
                'acceptance_authority':False,'owner_fidelity_approved':False,'n0_approved':False}
            auth_path=phase_dir/'current-authorization.protected.json';write(auth_path,auth)
            admitted['authorization']=ArtifactPin(auth_path,file_hash(auth_path))
            admission=TeachingAdmission(admitted,config_sha,initial,recipe.fingerprint(),code_sha)
            epochs=phase_dir/'epoch-checkpoints';epochs.mkdir(mode=0o700)
            print(json.dumps({'stage':'governing_phase_training','phase':phase,'reviewed_cases':len(selected),'epochs':plan['epochs'][phase]}),flush=True)
            result=construction.run_phase(recipe,admission,batch_loader=lambda digest:materialize(batch_cases[digest]),
                epochs=plan['epochs'][phase],destination=phase_dir/'final-checkpoint',epoch_checkpoint_directory=epochs)
            phase_results[phase]={'steps':result['step'],'checkpoint_manifest_sha256':result['manifest_sha256'],'model_state_sha256':result['model_state_sha256']}
            print(json.dumps({'stage':'governing_phase_checkpoint_retained','phase':phase,'optimizer_steps':result['step']}),flush=True)
        consumer=construction.consumer()
        response_path=output/'connected-responses.protected.jsonl'
        with response_path.open('xb') as f:
            for name,document in documents.items():
                produced=replay.produce(document)
                response=consumer.respond(document,produced,session)
                packet=model(produced.frame)
                if fingerprint(packet)!=response['packet_sha256']:refuse()
                f.write(canonical({'document_id':name,'split':document_splits[name],'response':response,'decision':decision_to_mapping(packet)})+b'\n')
                del packet,produced;gc.collect()
        for p in pins:p.verify()
        if inventory(overlay)!=overlay_sha or file_hash(helper)!=os.environ['PERSONALITY_CONSTRUCTION_HELPER_SHA256'] \
                or current_code_hashes()!=code_sha or compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts['source_package'],expected_receipt_sha256=COMPILE_SEAL)!=parent:refuse()
        namespace._clean_code(code,revision)
        summary={'schema':'alice-personality-connected-construction-summary-v1','state':'N1_N2_CONNECTED_CONSTRUCTION_RETAINED_UNQUALIFIED',
            'owner':'mxrayan','code_commit':revision,'helper_sha256':file_hash(helper),'namespace_sha256':GUARD_SHA,'isolation':isolation,
            'packet_index_sha256':plan['packet_index_sha256'],'owner_run_plan_sha256':os.environ['PERSONALITY_CONSTRUCTION_PLAN_SHA256'],
            'processor_overlay_sha256':overlay_sha,'source_geometry':[config.provider_state_count,config.provider_width],
            'phase_results':phase_results,'response_count':len(documents),'protected_response_file_sha256':file_hash(response_path),
            'feature_store':store.summary(),'closed_session_sha256':session['receipt_sha256'],
            'n3_gradient_steps':0,'n3_outcomes_pending_primary_review':True,'numeric_voice_gold_created':False,
            'private_gradient_authorized':True,'qualification':'UNQUALIFIED','acceptance_authority':False,
            'owner_fidelity_approved':False,'spoken_fidelity_verified':False,'n0_approved':False,'flora_work':False}
        summary['receipt_sha256']=sha256(canonical(summary)).hexdigest();write(output/'connected-construction-summary.json',summary)
        return summary


def main():
    # Only public code is accessed before run's actual P2 guard.
    code=Path(os.environ['PERSONALITY_REPO_ROOT'])
    path=code/'scripts/eipm/n1/enter_private_network_namespace.py'
    spec=importlib.util.spec_from_file_location('connected_existing_namespace',path)
    namespace=importlib.util.module_from_spec(spec);spec.loader.exec_module(namespace)
    try:
        result=run(namespace)
        print(json.dumps({'state':result['state'],'receipt_sha256':result['receipt_sha256'],'phase_results':result['phase_results']}));return 0
    except BaseException as exc:
        # Full diagnostic stays owner-only and is written only after a fresh
        # successful guard, never leaked through a raw private exception log.
        locations=[];frame=exc.__traceback__
        while frame:
            p=Path(frame.tb_frame.f_code.co_filename)
            if p.is_relative_to(code):locations.append({'module':p.relative_to(code).as_posix(),'line':frame.tb_lineno})
            frame=frame.tb_next
        try:
            namespace._p2_guard()
            root=Path(BASE);out=namespace._bounded_path(os.environ['PERSONALITY_RUN_ROOT'],root,file=False)
            os.umask(0o077)
            write(out/'failure-detail.protected.json',{'exception_type':type(exc).__name__,'message':str(exc),'public_code_locations':locations})
        except BaseException:pass
        print(json.dumps({'state':'FAILED_UNQUALIFIED','failure_type':type(exc).__name__,'public_code_locations':locations,'acceptance_authority':False}),file=sys.stderr)
        return 3


if __name__=='__main__':raise SystemExit(main())
