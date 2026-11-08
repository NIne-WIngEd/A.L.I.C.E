"""Owner-guarded actual frozen-base features for reviewed voice_intent teaching.

No optimizer, external teacher, source promotion or qualification. Immutable
scientific core and original verified publisher preparation are reused.
"""
import importlib.util,json,os,sys,gc
from pathlib import Path
from hashlib import sha256
from contextlib import ExitStack
from dataclasses import asdict
ROOT=Path('/homes/01/mxrayan/rayan-compute')
REV='333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME='rayan-voice-intent-features-r2-20261011'
INPUT_NAME='rayan-voice-intent-features-20261011'
VOICE_REFERENCE_SHA='7d928048ad3a59caa35ce2292eb2b9fd400ec11797c9e6309e5e4af36ac8db70'
INDEX='7c6c75d69f50d9c39abde4274d88e5543649a7ee69a62938242880fc0d7e1cd1'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def write(path,v):
 with path.open('xb') as f:f.write(canonical(v)+b'\n')
 return sha256(path.read_bytes()).hexdigest()

def run(namespace):
 isolation=namespace._p2_guard() # FIRST: before protected paths/stat/hash/read.
 os.umask(0o077);sys.dont_write_bytecode=True
 code=namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'],ROOT,file=False);namespace._clean_code(code,REV)
 assert sha256(Path(namespace.__file__).read_bytes()).hexdigest()==GUARD
 worker_sha=sha256(Path(__file__).read_bytes()).hexdigest()
 overlay=namespace._bounded_path(str(ROOT/'rayan-personality/runs/public-forward-576598/processor-deps'),ROOT,file=False)
 sys.path[:0]=[str(overlay),str(code)]
 import torch,PIL,torchvision,transformers
 assert (torch.__version__,PIL.__version__,torchvision.__version__,transformers.__version__)==('2.7.1+cu118','11.3.0','0.22.1+cu118','5.17.0')
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
 from src.alice_personality.n1 import compiler,raw_inference_lineage as raw
 from src.alice_personality.gemma_n0.preparation import verify_prepared
 from src.alice_personality.identity.codec import CognitiveFrameDocument,fingerprint
 from src.alice_personality.identity.contracts import IdentityModelConfig
 from src.alice_personality.identity.feature_producer import FrozenFrameProducer
 from src.alice_personality.identity.feature_store import FrozenFeatureStore
 from src.alice_personality.teaching.cases import ReviewedTeachingCase
 from src.alice_personality.teaching.materialize import materialize_case
 from src.alice_personality.teaching.lineage import verify_allocation
 from transformers import AutoProcessor
 helper=module('voice_intent_unchanged_construction',code/'scripts/eipm/gemma_n0/construct_personality.py')
 overlay_sha=helper.inventory(overlay)
 input_root=namespace._bounded_path(str(ROOT/'rayan-personality/private-inputs'/INPUT_NAME),ROOT,file=False)
 output=ROOT/'rayan-personality/private-compile'/('rayan-voice-intent-features-r2-'+os.environ['SLURM_JOB_ID'])
 output.mkdir(mode=0o700);output=namespace._bounded_path(str(output),ROOT,file=False)
 with ExitStack() as stack:
  held=[]
  def pin(path,digest,limit=64*1024**2):
   path=namespace._bounded_path(str(path),ROOT,file=True)
   value=stack.enter_context(_PinnedFile(path,digest,limit));held.append(value);value.stream.seek(0);return value
  def jpin(path,digest):return json.loads(pin(path,digest).stream.read())
  index=jpin(input_root/'packet-index.protected.json',INDEX)
  assert index['policy_count']==8 and index['documents']==16 and index['cases']==32
  assert index['absolute_acoustic_gold_created'] is False and index['empirical_failure_owner_or_spoken_gold_created'] is False
  artifacts=index['artifact_sha256']
  assert artifacts['source_package']=='3867ff04d1e326086b9086b2f106b9156b3a3ec8d637d3161e7bf01616183ee9'
  assert artifacts['compiled_receipt']=='fd7f3b09d68ffe20e054358aeb6d64aca80616182ff59870038dd6f5407e4bd9'
  assert artifacts['prepared_receipt']=='06fc6cc55fd6255d649ad0678704e25f78d29ea6313dc20399cc039f4addac8d'
  files={name:pin(input_root/name,digest) for name,digest in index['files'].items()}
  protocol=jpin(input_root/'annotation-protocol.protected.json',artifacts['annotation_protocol'])
  review=jpin(input_root/'primary-review.protected.json',artifacts['review_record'])
  assert protocol['other_teacher_used'] is False and protocol['acceptance_authority'] is False and review['private_gradient_authorized'] is False
  assert review['annotation_protocol_sha256']==artifacts['annotation_protocol']
  allocation=json.loads(files['construction-allocation.protected.json'].stream.read())
  docrows=[json.loads(line) for line in files['target-free-documents.protected.jsonl'].stream]
  assert all(set(row)=={'document_id','split','document'} and row['split']=='TRAIN' for row in docrows)
  documents={row['document_id']:CognitiveFrameDocument.from_mapping(row['document']) for row in docrows}
  cases=[ReviewedTeachingCase.from_mapping(json.loads(line)) for line in files['reviewed-teaching-cases.protected.jsonl'].stream]
  assert len(documents)==16 and len(cases)==32
  assert {phase:sum(case.phase==phase for case in cases) for phase in ('n1','n2','calibration')}=={'n1':16,'n2':16,'calibration':0}
  assert {name:doc.sha256 for name,doc in documents.items()}==index['document_sha256']
  assert {case.case_id:case.sha256 for case in cases}==index['case_sha256']
  assert all(case.artifact_sha256==artifacts and case.split=='TRAIN' and case.phase in {'n1','n2'} for case in cases)
  compiled=namespace._bounded_path(helper.COMPILED,ROOT,file=False)
  compiled_pin=pin(compiled/'compile_receipt.json',artifacts['compiled_receipt'])
  parent=compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts['source_package'],expected_receipt_sha256=helper.COMPILE_SEAL)
  originals_list=[json.loads(line) for line in pin(helper.SOURCE_CONTEXT,helper.SOURCE_CONTEXT_SHA).stream]
  originals={row['draft_workitem']['record_id']:row for row in originals_list}
  assert len(originals)==2556==len(originals_list)
  family_rows=[raw._json(line) for line in (compiled/'split_families.jsonl').read_bytes().splitlines()]
  reference_path=namespace._bounded_path(str(ROOT/'jobs'/'rayan-connected-voice-references-20261011.py'),ROOT,file=True)
  assert sha256(reference_path.read_bytes()).hexdigest()==VOICE_REFERENCE_SHA
  voice_reference=module('separately_pinned_connected_voice_reference',reference_path)
  connected_families=voice_reference.connected_voice_reference_families(family_rows,originals,expected_voice_count=96)
  assert protocol['source_complete_authoring_sha256']==helper.SOURCE_CONTEXT_SHA
  for document in documents.values():helper.verify_reference_families(document,allocation,connected_families,originals,protocol)
  for case in cases:verify_allocation(case.lineage,tuple(e.record for e in case.document.sources),case.split,allocation)
  original_plan=jpin(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json',
   '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
  reference=original_plan['completed_features']
  config=IdentityModelConfig(**jpin(Path(reference['directory'])/'model-config.protected.json',reference['file_sha256']['model-config.protected.json']))
  config.validate();assert [config.provider_state_count,config.provider_width]==[49,3840]
  prepared=pin(helper.PREPARED,artifacts['prepared_receipt']);preparation=verify_prepared(prepared.path)
  processor=AutoProcessor.from_pretrained(preparation['snapshot_path'],local_files_only=True,trust_remote_code=False)
  profile=helper.token_cost_profile(list(documents.values()),processor.tokenizer,layers=49,width=3840,max_tokens=preparation['model_geometry']['max_position_embeddings'])
  del processor;gc.collect()
  estimate=48*1024**3+profile['largest_padded_frame_value_bytes']*12
  helper.verify_memory_allocation(120*1024**3,estimate,os.environ['SLURM_MEM_PER_NODE'])
  disk=os.statvfs(output);assert disk.f_bavail*disk.f_frsize>=profile['state_value_bytes']+profile['serialization_overhead_budget_bytes']+2*1024**3
  profile.update(estimated_live_memory_budget_bytes=estimate,declared_memory_budget_bytes=120*1024**3,
   actual_slurm_memory_bytes=int(os.environ['SLURM_MEM_PER_NODE'])*1024**2)
  write(output/'token-cost-profile.json',profile);write(output/'model-config.protected.json',asdict(config))
  print(json.dumps({'stage':'voice_intent_source_target_custody_verified','documents':16,'cases':32,'unique_entries':profile['unique_entries'],
   'full_bf16_feature_value_bytes':profile['state_value_bytes'],'estimated_live_memory_bytes':estimate,'measured_peak':False}),flush=True)
  store=FrozenFeatureStore.create(output/'feature-store',source_package_sha256=artifacts['source_package'])
  producer=FrozenFrameProducer.load(prepared.path,expected_file_sha256=artifacts['prepared_receipt'],device='cpu',
   learned_width=config.learned_width,attention_heads=config.attention_heads,graph_layers=config.graph_layers,readout_layers=config.readout_layers,feature_store=store)
  assert producer.config==config
  bindings={}
  for count,(name,document) in enumerate(documents.items(),1):
   value=producer.produce(document);bindings[name]=value.binding;del value;gc.collect()
   print(json.dumps({'stage':'voice_intent_frozen_base_complete_frame','completed':count,'total':16}),flush=True)
  session=producer.close_and_verify()
  assert session['status']=='CLOSED_SOURCE_VERIFIED' and session['mechanical_fixture_only'] is False
  session_file=write(output/'feature-session.protected.json',session)
  binding_file=write(output/'frame-bindings.protected.json',bindings)
  del producer,store;gc.collect()
  store=FrozenFeatureStore.open(output/'feature-store',expected_manifest_sha256=session['feature_store']['manifest_sha256'])
  replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=session['receipt_sha256'])
  groups={doc.sha256:[name,doc,[]] for name,doc in documents.items()}
  for case in cases:groups[case.document.sha256][2].append(case)
  target_coverage={};materialized=0
  with (output/'materialization-receipts.protected.jsonl').open('xb') as receipts:
   for count,(_,document,group) in enumerate(groups.values(),1):
    produced=replay.produce(document)
    for case in group:
     result=materialize_case(case,produced,session,config,expected_case_sha256=case.sha256,expected_artifact_sha256=artifacts,family_splits=allocation)
     result.batch.validate(config);receipts.write(canonical(result.receipt)+b'\n');materialized+=1
     for field,coverage in result.receipt['coverage'].items():target_coverage[field]=target_coverage.get(field,0)+coverage['available_facts']
     del result
    del produced;gc.collect()
    print(json.dumps({'stage':'voice_intent_actual_case_materialization','completed_documents':count,'materialized_cases':materialized}),flush=True)
  assert materialized==32
  for value in held:value.verify()
  assert helper.inventory(overlay)==overlay_sha
  assert compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts['source_package'],expected_receipt_sha256=helper.COMPILE_SEAL)==parent
  namespace._clean_code(code,REV)
  assert sha256(reference_path.read_bytes()).hexdigest()==VOICE_REFERENCE_SHA
  assert sha256(Path(__file__).read_bytes()).hexdigest()==worker_sha
  summary={'schema':'alice-personality-voice_intent-actual-feature-materialization-v1','state':'REAL_FEATURES_MATERIALIZED_UNADMITTED',
   'job_id':int(os.environ['SLURM_JOB_ID']),'owner':'mxrayan','code_commit':REV,'namespace_sha256':GUARD,'isolation':isolation,
   'worker_file_sha256':worker_sha,'packet_index_file_sha256':INDEX,
   'connected_reference_adapter_sha256':VOICE_REFERENCE_SHA,'original_failed_job':577251,
   'original_compiler_membership_unchanged':True,'separate_connected_voice_membership_verified':True,
   'artifact_sha256':artifacts,'policy_count':8,'source_document_count':16,'materialized_case_count':32,
   'phase_counts':{'n1':16,'n2':16,'calibration':0},'source_geometry':[49,3840],'closed_session_sha256':session['receipt_sha256'],
   'closed_session_file_sha256':session_file,'frame_bindings_file_sha256':binding_file,
   'feature_store_manifest_sha256':session['feature_store']['manifest_sha256'],'target_available_fact_counts':target_coverage,
   'source_truncation':False,'base_generation_or_chat_template':False,'upstream_gradient':False,'private_gradient_steps':0,
   'training_authorized_by_this_job':False,'owner_fidelity_approved':False,'spoken_fidelity_verified':False,
   'n0_approved':False,'full_personality_complete':False,'complete_voice_intent_curriculum':False,'private_payload_exported':False,'flora_work':False}
  summary['receipt_sha256']=sha256(canonical(summary)).hexdigest()
  write(ROOT/'jobs'/(NAME+'.json'),summary)
  print(json.dumps(summary,sort_keys=True),flush=True)

code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
namespace=module('voice_intent_owner_namespace',guard)
try:run(namespace)
except Exception as exc:
 import traceback
 frames=[{'module':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
 print(json.dumps({'state':'VOICE_INTENT_FEATURES_REFUSED','exception_type':type(exc).__name__,'public_code_frames':frames}),flush=True)
 raise SystemExit(2)
