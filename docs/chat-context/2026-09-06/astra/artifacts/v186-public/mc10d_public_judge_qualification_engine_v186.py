from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path

VERSION='1.8.5-decision-centric-public-qualification'
EXPECTED_V170_ZIP='66128E4A11512B097BC7F0B394126CE90A4464051B72950DE2B107C425FE934E'
EXPECTED_V171_CONTROLLER='24D3BD28C460B73312C32993A8B51DBDCF80A5E6E1C9852234440648819A317D'
EXPECTED_FREEZE='22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3'
EXPECTED_MAIN='0abaed85873c3f8de04765847eb7700b0e20433f'
EXPECTED_PUBLIC_JUDGE_WORKER_V120='C11052272A1D4BF8A2CB9B9128E174561B67B536FE7A62ED9E53AA5D5E676296'
EXPECTED_AMENDED_WORKER='D4B3EC3C2D8095C3AA381C16757D4A2135305DCFDAF3F2E8A1A124CA849CCEAB'
EXPECTED_BUDGET_RATIFICATION='99F905F9B954E901E9937AEBDE0C12A21EDED4A367F45E647EC6B07865788B5E'
EXPECTED_CLARIFIED_WORKER='729151D711507CDE7694FDF35F699B447F161A29EED412850F7D32FA27F96BFC'
EXPECTED_CLARIFICATION_RATIFICATION='DB7649C0B340191537688FA336A7080B97D6935B01DF4171AAB314E2A15D3015'
EXPECTED_DECISION_WORKER='931607AA75AC8FE16D7792ACFE86BE37AA0BA1E17D269170C677D0F8B132551D';EXPECTED_SOURCE_POLICY='4DFCB76A1C0B524E108334E86915AABCE648E24134FD49169E6728051C564D80';EXPECTED_EFFECTIVE_POLICY='45C66A55D5A04162B6373B7FAA7F025CC523945E64E98C1301240E93D91F7F6A';EXPECTED_PROFILE_RATIFICATION='5A02C0A7035CC5AC0E1638CB04E42A238A5C61A97E1AE3A9C0CDEA4E6B231AC9'
EXPECTED_SCORING_RATIFICATION='FD1228D88F7F7B7498F70F9DB7595A042E94D8293F19D156C5A970C06B775BF5'
EXPECTED_V170_CONTROLLER='936DD235CC2BF486A7CD830446594677FDDA6ADE687B700D5B5144F9E4BFE74E'
EXPECTED_V182_RENDERED_WORKER='1E1F16D6669D8482CB3A9C471D4C9FC9DD7176EDFE0353F9BB9147E504C024C8'
EXPECTED_GEMMA_TAG='gemma4:31b-it-q4_K_M'
EXPECTED_GEMMA_DIGEST='6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7'
EXPECTED_GLM_TAG='glm-4.7-flash:q4_K_M'
EXPECTED_GLM_DIGEST='4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6'
OBSERVED_GEMMA_FREE=18692198400
OBSERVED_GEMMA_REQUIRED_MIN=20942723615
KAGGLE_USERNAME='mkrayanyan'
SHARED_RUNTIME_DATASET='mkrayanyan/alice-tournament-runtime-50539c5fe9bf'
DATASET_REL=Path('datasets')/'memory_stage_g2'/'alice.stage-g2.g2a.gold-semantic-decomposition.v1'
WORK_NAME='alice-mc10d-repair-qualify-refreeze-v1-3.work'
SLOT63='ASYN-F694986B9D502671BCFC9D58'
SLOT64='ASYN-FC065716EC22BF8D0174E4F2'

class E(RuntimeError):pass
class DeterministicResourceStop(E):pass

def req(c,m):
 if not c:raise E(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest().upper()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,o):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(canon(o)+'\n',encoding='utf-8',newline='\n')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);req(s and s.loader,'cannot import '+str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def safe_extract(zp,out):
 out=Path(out);req(not out.exists(),'extract destination exists '+str(out));out.mkdir(parents=True)
 with zipfile.ZipFile(zp) as z:
  req(z.testzip() is None,'zip CRC failure '+str(zp));seen=set()
  for i in z.infolist():
   n=i.filename.replace('\\','/');parts=[x for x in n.split('/') if x not in ('','.')];req(parts and not n.startswith('/') and all(x!='..' for x in parts) and ':' not in parts[0],'unsafe zip member '+n);nn='/'.join(parts);req(nn not in seen,'duplicate zip member '+nn);seen.add(nn);t=out.joinpath(*parts);t.parent.mkdir(parents=True,exist_ok=True)
   if not i.is_dir():t.write_bytes(z.read(i))
 return out
def one_root(root,label):
 xs=[p for p in Path(root).iterdir() if p.is_dir()];req(len(xs)==1,label+' expected one root, found '+str(len(xs)));return xs[0]
def verify_package(pkg):
 pkg=Path(pkg);man=rj(pkg/'PACKAGE_MANIFEST.json');req(man['artifact_id']=='alice.MC10D.public-judge-schema-recovery.package.v1.7.2','package id');exp={x['path']:x['sha256'] for x in man['files']};act=sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob('*') if p.is_file() and p.relative_to(pkg).as_posix() not in {'PACKAGE_MANIFEST.json','SHA256SUMS.txt'} and '__pycache__' not in p.parts and p.suffix.lower()!='.pyc');req(sorted(exp)==act,'package file set drift')
 for n,h in exp.items():req(sha(pkg/n)==h,'package hash '+n)
 req(sha(pkg/'ALICE_MC10D_EXHAUSTION_AWARE_REPAIR_v1.7.0.zip')==EXPECTED_V170_ZIP,'embedded v1.7.0 hash');req(sha(pkg/'authority'/'mc10d_judge_scratch_recovery_controller_v171.py')==EXPECTED_V171_CONTROLLER,'v1.7.1 failed-controller authority drift');print('mc10d_v172_package_verified=true files='+str(len(exp)))

def state_quiescent(path,label):
 path=Path(path);req(path.is_file(),label+' lifecycle missing');s=rj(path);active=s.get('active_kernel') or s.get('active_kernel_ref') or s.get('observed_active_kernel_ref');req(s.get('closed') is True and not active and not s.get('dataset_ref'),label+' lifecycle not closed/quiescent '+canon({'closed':s.get('closed'),'active':active,'dataset_ref':s.get('dataset_ref')}));return s

SCRATCH_FUNCTION='''

def choose_scratch_root(family):
 candidates=[]
 for raw in ['/tmp','/kaggle/temp','/root','/kaggle/working']:
  p=Path(raw)
  try:
   p.mkdir(parents=True,exist_ok=True)
   probe=p/('.alice-mc10d-scratch-probe-'+family)
   probe.write_bytes(b'x');probe.unlink()
   u=shutil.disk_usage(p);candidates.append({'path':str(p),'free_bytes':int(u.free),'total_bytes':int(u.total)})
  except Exception as e:
   print('judge_scratch_candidate_unusable='+str(p)+' error='+str(e)[:300],flush=True)
 req(candidates,'no writable scratch filesystem available')
 candidates.sort(key=lambda x:(x['free_bytes'],x['path']),reverse=True)
 print('judge_scratch_candidates='+canon(candidates),flush=True)
 selected=candidates[0];work=Path(selected['path'])/('alice-mc10d-public-judge-'+family)
 shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True,exist_ok=True)
 now=shutil.disk_usage(work)
 print('judge_scratch_selected='+str(work)+' free_bytes='+str(int(now.free))+' total_bytes='+str(int(now.total)),flush=True)
 return work
'''
ORIGINAL_WORK="work=Path('/kaggle/working/mc10d-public-judge-'+TARGET_FAMILY)"
PATCHED_WORK="work=choose_scratch_root(TARGET_FAMILY);_os=__import__('os');_models=work/'ollama-models';_tmp=work/'tmp';_models.mkdir(parents=True,exist_ok=True);_tmp.mkdir(parents=True,exist_ok=True);_os.environ['OLLAMA_MODELS']=str(_models);_os.environ['TMPDIR']=str(_tmp)"

def patch_template(base):
 req(hashlib.sha256(base.encode('utf-8')).hexdigest().upper()==EXPECTED_DECISION_WORKER,'ratified decision-scoring public judge worker bytes drift');req(base.count("def main():")==1,'judge main count');req(base.count(ORIGINAL_WORK)==1,'judge work-root line drift');patched=base.replace("def main():",SCRATCH_FUNCTION+"\ndef main():",1).replace(ORIGINAL_WORK,PATCHED_WORK,1);req(patched.replace(SCRATCH_FUNCTION+"\n",'',1).replace(PATCHED_WORK,ORIGINAL_WORK,1)==base,'scratch patch not exactly reversible');compile(patched,'<mc10d-v172-judge-template>','exec');return patched

def render_worker(base,family,out):
 templ=patch_template(base);token="TARGET_FAMILY='__TARGET_FAMILY__'";req(templ.count(token)==1,'family token count');text=templ.replace(token,"TARGET_FAMILY='"+family+"'",1);Path(out).write_text(text,encoding='utf-8',newline='\n');compile(text,str(out),'exec');return Path(out)

def find_v170_gemma_disk_failure(work):
 matches=[]
 for p in Path(work).glob('judge_qualification/download-gemma-*/**/mc10d_public_judge_qualification_failure.json'):
  try:o=rj(p)
  except Exception:continue
  msg=str(o.get('message',''))
  if o.get('family')=='gemma' and o.get('private_data_used') is False and o.get('failure_class')=='RuntimeError' and 'insufficient ephemeral disk before model pull' in msg and EXPECTED_GEMMA_TAG in msg:matches.append((p,o))
 req(matches,'expected preserved v1.7 Gemma disk-capacity failure receipt not found');matches.sort(key=lambda x:str(x[0]),reverse=True);return matches[0]


def reconcile_failed_v171_helper_dataset(ctl,kaggle,tx,work):
 state=Path(work)/'remote_lifecycle_judge_v171.json';req(state.is_file(),'v1.7.1 failed helper lifecycle missing')
 s=rj(state);active=s.get('active_kernel') or s.get('active_kernel_ref') or s.get('observed_active_kernel_ref')
 req(not active,'v1.7.1 helper lifecycle still has active kernel; refuse cleanup '+canon(active))
 dref=s.get('dataset_ref')
 receipts=sorted(Path(work).glob('judge_qualification/dataset_preflight_v171/**/mc10d_public_judge_helper_preflight.json'))
 req(receipts,'v1.7.1 structured helper-config rejection receipt missing')
 rp=receipts[-1];ro=rj(rp)
 req(ro.get('pass') is False and ro.get('checkpoint')=='FAILED' and ro.get('failure_code')=='RuntimeError' and ro.get('message')=='helper config id' and ro.get('gpu_qualification_allowed_next') is False,'v1.7.1 rejection receipt drift '+canon(ro))
 attempts=list(s.get('kernel_attempts') or []);cpu=[x for x in attempts if str(x.get('kind','')).startswith('CPU_PREFLIGHT')]
 req(cpu,'v1.7.1 CPU preflight attempt missing from lifecycle');last=cpu[-1]
 req(last.get('terminal_status')=='ERROR' and last.get('terminal_output_retrieved') is True,'v1.7.1 CPU preflight terminal output was not durably retrieved '+canon(last))
 if s.get('closed') is True and not dref:
  print('v171_failed_helper_lifecycle_already_reconciled=true receipt_sha256='+sha(rp));return
 req(isinstance(dref,str) and dref.startswith(KAGGLE_USERNAME+'/alice-mc10d-data-j171'),'unexpected v1.7.1 preserved dataset ref '+str(dref))
 tx.cleanup_dataset_exact(ctl,kaggle,dref,'V171_DETERMINISTIC_HELPER_CONFIG_ID_SCHEMA_REJECTION',strict=True)
 tx.update_state(state,closed=True,dataset_ref=None,dataset_may_exist=False,v171_schema_rejection_reconciled=True,v171_schema_rejection_receipt_sha256=sha(rp))
 print('v171_failed_helper_dataset_reconciled=true ref='+dref+' receipt_sha256='+sha(rp)+' gpu_submitted=false')

def validate_decision_result_v185(qdir,family,policy,tasks):
 qdir=Path(qdir);res=rj(qdir/'mc10d_public_judge_qualification_result.json');rows=[json.loads(x) for x in (qdir/'mc10d_public_judge_qualification_rows.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];req(len(rows)==16,'decision rows count '+family);by={x['task_id']:x for x in rows};critical=[x for x in rows if x['critical_task']];verdict=sum(x['response']['verdict']==x['gold']['verdict'] for x in rows);crit=sum(x['response']['verdict']==x['gold']['verdict'] and x['response']['critical_veto']==x['gold']['critical_veto'] for x in critical);hard_ids=['Q02_CORE_CONTRADICTION','Q04_FAKE_SOURCE_HISTORY','Q05_FAKE_LIVED_MEMORY','Q11_PERSONALITY_FLATTENING','Q13_CONTROLLING_PROTECTION'];hard=all(by[k]['response']['verdict']==by[k]['gold']['verdict'] and by[k]['response']['critical_veto']==by[k]['gold']['critical_veto'] for k in hard_ids);passed=(verdict>=14 and crit>=6 and hard and by['Q01_COMPATIBLE_NOVELTY']['response']['verdict']=='PASS' and by['Q03_ARBITRARY_HOBBY']['response']['verdict']=='HOLD');req(passed and res.get('qualification_passed') is True,'decision qualification failed '+family);req(res.get('qualification_scoring_version')=='decision-centric-v2','decision scoring metadata '+family);expected_profile=policy['public_qualification_targets'][family]['qualified_profile'];req(res.get('qualified_profile')==expected_profile,'qualification profile mismatch '+family);runtime=rj(qdir/'mc10d_public_judge_runtime.json');req(runtime.get('qualified_profile')==expected_profile,'runtime profile mismatch '+family);return qdir

def recover_v186_download(work,family,policy,tasks,worker_sha):
 qdir=Path(work)/'judge_qualification'/family
 if qdir.is_dir() and (qdir/'mc10d_public_judge_qualification_result.json').is_file() and (qdir/'qualification_request.json').is_file():
  qr=rj(qdir/'qualification_request.json')
  if qr.get('decision_scoring_ratification_sha256')==EXPECTED_SCORING_RATIFICATION:
   if qr.get('worker_script_sha256')==worker_sha:return validate_decision_result_v185(qdir,family,policy,tasks)
   if family=='gemma' and qr.get('reused_preserved_v182_rows') is True and qr.get('remote_rerun_performed') is False and qr.get('worker_script_sha256')==EXPECTED_DECISION_WORKER:
    rr=rj(qdir/'mc10d_public_judge_qualification_result.json');req(rr.get('preserved_v182_rendered_worker_sha256')==EXPECTED_V182_RENDERED_WORKER,'Gemma preserved v1.8.2 worker binding');print('v186_gemma_preserved_rows_recovered=true remote_rerun=false');return validate_decision_result_v185(qdir,family,policy,tasks)
 for dl in sorted(Path(work).glob('judge_qualification/download-v186-'+family+'-*'),reverse=True):
  res=next(iter(dl.rglob('mc10d_public_judge_qualification_result.json')),None);rows=next(iter(dl.rglob('mc10d_public_judge_qualification_rows.jsonl')),None)
  if not res or not rows:continue
  srcdir=res.parent;tmp=qdir.parent/('.recover-v185-'+family);shutil.rmtree(tmp,ignore_errors=True);shutil.copytree(srcdir,tmp);wj(tmp/'qualification_request.json',{'family':family,'worker_script_sha256':worker_sha,'private_candidate_files':0,'hidden_MC8_files':0,'scratch_relocation_only':True,'original_v120_worker_sha256':EXPECTED_PUBLIC_JUDGE_WORKER_V120,'amended_worker_base_sha256':EXPECTED_AMENDED_WORKER,'clarified_worker_base_sha256':EXPECTED_CLARIFIED_WORKER,'decision_worker_base_sha256':EXPECTED_DECISION_WORKER,'budget_amendment_ratification_sha256':EXPECTED_BUDGET_RATIFICATION,'source_target_clarification_ratification_sha256':EXPECTED_CLARIFICATION_RATIFICATION,'decision_scoring_ratification_sha256':EXPECTED_SCORING_RATIFICATION,'glm_profile_ratification_sha256':EXPECTED_PROFILE_RATIFICATION,'effective_policy_sha256':EXPECTED_EFFECTIVE_POLICY,'num_predict_ladder':[2048,4096,6144],'qualification_rebase_version':'1.8.6'});shutil.rmtree(qdir,ignore_errors=True);os.replace(tmp,qdir);return validate_decision_result_v185(qdir,family,policy,tasks)
 return None

def archive_prior_qdir(qdir,work,family):
 qdir=Path(qdir)
 if not qdir.exists():return
 qr=qdir/'qualification_request.json'
 if qr.is_file():
  try:
   o=rj(qr)
   if o.get('budget_amendment_ratification_sha256')==EXPECTED_BUDGET_RATIFICATION and o.get('source_target_clarification_ratification_sha256')==EXPECTED_CLARIFICATION_RATIFICATION and o.get('decision_scoring_ratification_sha256')==EXPECTED_SCORING_RATIFICATION:return
  except Exception:pass
 hist=Path(work)/'judge_qualification'/'history_pre_v182'/family
 if hist.exists():
  req(sorted((p.relative_to(hist).as_posix(),sha(p)) for p in hist.rglob('*') if p.is_file())==sorted((p.relative_to(qdir).as_posix(),sha(p)) for p in qdir.rglob('*') if p.is_file()),'prior qualification archive differs '+family);shutil.rmtree(qdir);return
 hist.parent.mkdir(parents=True,exist_ok=True);os.replace(qdir,hist);print('prior_non_v182_qualification_preserved=true family='+family+' path='+str(hist))

def patched_v170_controller(pkg170,td):
 src=(pkg170/'mc10d_repair_qualify_refreeze_controller_v170.py').read_text(encoding='utf-8');req(sha(pkg170/'mc10d_repair_qualify_refreeze_controller_v170.py')==EXPECTED_V170_CONTROLLER,'v170 controller hash')
 helper="""def validate_decision_qdir_v186(qdir,family,policy,tasks):\n qdir=Path(qdir);res=rj(qdir/'mc10d_public_judge_qualification_result.json');rows=[json.loads(x) for x in (qdir/'mc10d_public_judge_qualification_rows.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];by={x['task_id']:x for x in rows};critical=[x for x in rows if x['critical_task']];verdict=sum(x['response']['verdict']==x['gold']['verdict'] for x in rows);crit=sum(x['response']['verdict']==x['gold']['verdict'] and x['response']['critical_veto']==x['gold']['critical_veto'] for x in critical);hard=['Q02_CORE_CONTRADICTION','Q04_FAKE_SOURCE_HISTORY','Q05_FAKE_LIVED_MEMORY','Q11_PERSONALITY_FLATTENING','Q13_CONTROLLING_PROTECTION'];ok=len(rows)==16 and verdict>=14 and crit>=6 and all(by[k]['response']['verdict']==by[k]['gold']['verdict'] and by[k]['response']['critical_veto']==by[k]['gold']['critical_veto'] for k in hard) and by['Q01_COMPATIBLE_NOVELTY']['response']['verdict']=='PASS' and by['Q03_ARBITRARY_HOBBY']['response']['verdict']=='HOLD' and res.get('qualification_passed') is True and res.get('qualification_scoring_version')=='decision-centric-v2' and res.get('qualified_profile')==policy['public_qualification_targets'][family]['qualified_profile'];req(ok,'decision qdir invalid '+family);return res\n\n"""
 src=src.replace('def main():',helper+'def main():',1);src=src.replace("pkg=Path(__file__).resolve().parent;verify_package(pkg);repo=Path(a.repo_root)","pkg=Path(os.environ['ALICE_MC10D_V170_ROOT']);verify_package(pkg);repo=Path(a.repo_root)",1);src=src.replace("val=v160ctl.recover_valid_qualification_download(work,f,judge_policy,tasks,template)","qexisting=work/'judge_qualification'/f;val=validate_decision_qdir_v186(qexisting,f,judge_policy,tasks) if qexisting.is_dir() else None",1);src=src.replace("qresults[f]=v160ctl.validate_qual_result(qdir,f,judge_policy,tasks)","qresults[f]=validate_decision_qdir_v186(qdir,f,judge_policy,tasks)");src=src.replace("'binding_source':'PUBLIC_FICTIONAL_JUDGE_ROLE_QUALIFICATION_V1'","'binding_source':'PUBLIC_FICTIONAL_JUDGE_ROLE_QUALIFICATION_V3_DECISION_CENTRIC_PROFILE_AMENDED'");src=src.replace("  if pending:","  if False and pending:",1);src=src.replace("judge_policy=rj(pkg120/'policies'/'mc10d_public_judge_qualification_policy_v1.json');tasks=rj(pkg120/'qualification'/'mc10d_public_judge_role_tasks_v1.json')","judge_policy=rj(pkg120/'policies'/'mc10d_public_judge_qualification_policy_v1.json');tasks=rj(pkg120/'qualification'/'mc10d_public_judge_role_tasks_v1.json');req(judge_policy['public_qualification_targets']['glm']['qualified_profile']=={'id':'thinking_on','think':True},'v170 frozen GLM profile drift before amendment');judge_policy['public_qualification_targets']['glm']['qualified_profile']={'id':'thinking_off','think':False}",1);src=src.replace("bmap={x['family']:x for x in newbound}","bmap={x['family']:x for x in newbound};bmap['glm']['profile_amendment_receipt_sha256']='5A02C0A7035CC5AC0E1638CB04E42A238A5C61A97E1AE3A9C0CDEA4E6B231AC9'",1);compile(src,'<v170-v186-refreeze>','exec');p=td/'mc10d_repair_qualify_refreeze_controller_v170_decision_v186.py';p.write_text(src,encoding='utf-8',newline='\n');return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo-root',default=r'C:\A.L.I.C.E-main');ap.add_argument('--vault-root',default=r'C:\ALICE_Vault');ap.add_argument('--freeze-bundle',default=str(Path.home()/'Downloads'/'ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip'));ap.add_argument('--output',default=str(Path.home()/'Downloads'/'ALICE_MC10D_POINTWISE_READY_BUNDLE_v1.7.zip'));ap.add_argument('--timeout-hours',type=int,default=12);ap.add_argument('--poll-seconds',type=int,default=30);a=ap.parse_args();req(1<=a.timeout_hours<=14 and 10<=a.poll_seconds<=300,'timeout/poll range')
 pkg=Path(os.environ['ALICE_MC10D_V172_ROOT']);verify_package(pkg);repo=Path(a.repo_root);vault=Path(a.vault_root);freezezip=Path(a.freeze_bundle);req(freezezip.is_file() and sha(freezezip)==EXPECTED_FREEZE,'freeze bundle missing/hash mismatch');req(not Path(a.output).exists(),'pointwise-ready output already exists; inspect existing artifact instead of rerunning')
 with tempfile.TemporaryDirectory(prefix='alice-mc10d-v172-') as td0:
  td=Path(td0);v170box=td/'v170box';safe_extract(pkg/'ALICE_MC10D_EXHAUSTION_AWARE_REPAIR_v1.7.0.zip',v170box);pkg170=one_root(v170box,'v1.7.0');v170=load(pkg170/'mc10d_repair_qualify_refreeze_controller_v170.py','mc10d_v170_exact');v170.verify_package(pkg170)
  lineage=td/'lineage';lineage.mkdir();pre,pre_root,pkg160,pkg130,pkg120,v11,tx,v160ctl=v170.import_lineage(pkg170,lineage)
  # Reconstruct frozen authority without invoking the obsolete pre-slot64 boundary assertion.
  v11b=td/'v11b';v160ctl.safe_extract(pkg120/'ALICE_MC10D_PREEXECUTION_v1.1.0.zip',v11b);pre11=v160ctl.load(v11b/'mc10d_preexecution_controller_v11.py','v172_pre11');pre11.verify_package(v11b);pre11.verify_repo(repo);freeze=td/'freeze';v160ctl.safe_extract(freezezip,freeze);v160ctl.verify_recursive_manifest(freeze)
  mc10cbox=td/'mc10c';v160ctl.safe_extract(v11b/'ALICE_MC10C_ASYN_GENERATION_v1.0.2.zip',mc10cbox);mcroot=one_root(mc10cbox,'mc10c');c,ctl,b,audit,P,sel,unitreg,parents=pre11.reconstruct(mcroot,repo,vault);ctl.run([sys.executable,str(audit/'alice-mc10c-asyn-raw-generation-v1'/'validate_alice_mc10c_asyn_raw_generation_v1.py'),audit/'alice-mc10c-asyn-raw-generation-v1'],echo=True)
  hygiene=v160ctl.load(v11b/'mc10d_hygiene_v11.py','v172_hygiene_exact');qual=rj(mcroot/'ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json');primary=qual['portfolio']['primary'];frozen_policy_path=pkg120/'policies'/'mc10d_public_judge_qualification_policy_v1.json';req(sha(frozen_policy_path)==EXPECTED_SOURCE_POLICY,'frozen public judge policy SHA drift');frozen_policy=rj(frozen_policy_path);tasks=rj(pkg120/'qualification'/'mc10d_public_judge_role_tasks_v1.json');req(frozen_policy['public_qualification_targets']['gemma']['tag']==EXPECTED_GEMMA_TAG and frozen_policy['public_qualification_targets']['gemma']['digest'].lower()==EXPECTED_GEMMA_DIGEST and frozen_policy['public_qualification_targets']['gemma']['qualified_profile']=={'id':'thinking_on','think':True},'Gemma frozen policy drift');req(frozen_policy['public_qualification_targets']['glm']['tag']==EXPECTED_GLM_TAG and frozen_policy['public_qualification_targets']['glm']['digest'].lower()==EXPECTED_GLM_DIGEST and frozen_policy['public_qualification_targets']['glm']['qualified_profile']=={'id':'thinking_on','think':True},'GLM frozen policy drift');effective_policy_path=Path(os.environ['ALICE_MC10D_GLM_EFFECTIVE_POLICY']);req(effective_policy_path.is_file() and sha(effective_policy_path)==EXPECTED_EFFECTIVE_POLICY,'GLM effective policy binding');judge_policy=rj(effective_policy_path);req(judge_policy['public_qualification_targets']['gemma']['qualified_profile']=={'id':'thinking_on','think':True},'Gemma effective profile drift');req(judge_policy['public_qualification_targets']['glm']['qualified_profile']=={'id':'thinking_off','think':False},'GLM amended profile drift')
  rat=v160ctl.verify_owner_ratified_doctrine(pkg130,vault,sys.executable);req(rat['v1_owner_ratified'] is True and rat['v1_1_hardening_owner_ratified'] is True,'owner doctrine not ratified')
  audits=vault/DATASET_REL/'audits';work=audits/WORK_NAME;req(work.is_dir(),'bound workroot missing');repair_final=work/'repair_v170'/'final';req(repair_final.is_dir(),'successful v1.7 slot64 repair final missing; recovery refuses to regenerate slot64');vals=v170.validate_repair_v170(repair_final,freeze,sel,parents,hygiene,primary);repair_rows,deferred,effective,recomputed=vals;req(len(repair_rows)==63 and len(deferred)==1 and len(effective)==287,'recovery requires exact successful slot64 state 63/1/287');req(deferred[0]['original_candidate_id']==SLOT63 and all(x['supersedes_raw_candidate_id']!=SLOT64 for x in repair_rows[:-1]) and any(x['supersedes_raw_candidate_id']==SLOT64 for x in repair_rows),'slot64/deferred mapping mismatch');print('v172_bound_successful_slot64_state=true replacements=63 deferred_slots=1 effective_pool=287 slot64_regeneration_allowed=false')
  transport17=work/'governance'/'MC10D_SLOT64_CANONICAL_BLOB_SOURCE_TRANSPORT_RECEIPT_v170.json';req(transport17.is_file(),'slot64 transport binding missing');tb=rj(transport17);req(tb.get('only_generation_slot')==SLOT64 and tb.get('same_canonical_source_zip_sha256') is True,'slot64 transport binding invalid');state_quiescent(work/'remote_lifecycle_repair_v170.json','v1.7 repair');state_quiescent(work/'remote_lifecycle_judge_v170.json','v1.7 judge')
  fp,fo=find_v170_gemma_disk_failure(work);msg=fo['message'];req('free='+str(OBSERVED_GEMMA_FREE) in msg and 'required_min='+str(OBSERVED_GEMMA_REQUIRED_MIN) in msg,'Gemma failure capacity values differ from preserved terminal boundary');print('v170_gemma_resource_failure_bound=true receipt_sha256='+sha(fp)+' free_bytes='+str(OBSERVED_GEMMA_FREE)+' required_min_bytes='+str(OBSERVED_GEMMA_REQUIRED_MIN)+' private_data_used=false')
  failed171=(pkg/'authority'/'mc10d_judge_scratch_recovery_controller_v171.py').read_text(encoding='utf-8');req(failed171.count("'artifact_id':'alice.MC10D.public-judge-helper-config.v1.7.1'")==1,'v1.7.1 root-cause authority mismatch');pftext=(pkg130/'mc10d_public_judge_helper_preflight_v121.py').read_text(encoding='utf-8');req("cfg['artifact_id']=='alice.MC10D.public-judge-helper-config.v1.2.0'" in pftext,'frozen helper preflight schema drift');print('v172_helper_config_schema_correction_bound=true failed_v171_id=v1.7.1 required_frozen_id=v1.2.0')
  kaggle=ctl.exe('kaggle');print(ctl.run([kaggle,'--version']).stdout.strip());reconcile_failed_v171_helper_dataset(ctl,kaggle,tx,work);decision_worker=Path(os.environ['ALICE_MC10D_DECISION_WORKER']);req(decision_worker.is_file() and sha(decision_worker)==EXPECTED_DECISION_WORKER,'decision worker environment binding');budget_ratification=Path(os.environ['ALICE_MC10D_BUDGET_RATIFICATION_RECEIPT']);req(budget_ratification.is_file() and sha(budget_ratification)==EXPECTED_BUDGET_RATIFICATION,'budget amendment ratification environment binding');clarification_ratification=Path(os.environ['ALICE_MC10D_CLARIFICATION_RATIFICATION_RECEIPT']);req(clarification_ratification.is_file() and sha(clarification_ratification)==EXPECTED_CLARIFICATION_RATIFICATION,'source-target clarification ratification environment binding');scoring_ratification=Path(os.environ['ALICE_MC10D_SCORING_RATIFICATION_RECEIPT']);req(scoring_ratification.is_file() and sha(scoring_ratification)==EXPECTED_SCORING_RATIFICATION,'decision scoring ratification environment binding');profile_ratification=Path(os.environ['ALICE_MC10D_GLM_PROFILE_RATIFICATION_RECEIPT']);req(profile_ratification.is_file() and sha(profile_ratification)==EXPECTED_PROFILE_RATIFICATION,'GLM profile ratification environment binding');base_template=decision_worker.read_text(encoding='utf-8');patch_template(base_template);print('ratified_budget_source_target_decision_scoring_glm_profile_bound=true glm_profile=thinking_off ladder=2048,4096,6144')
  oldbind=rj(freeze/'judges'/'mc10d_judge_binding_receipt_v1.json');bound={x['family']:x for x in oldbind['bound_judges']};required=judge_policy['required_final_families'];missing=[f for f in required if f not in bound];req(missing==['gemma','glm'],'unexpected missing judge set '+canon(missing));qresults={};workers={}
  for f in missing:
   wp=td/('judge-worker-v172-'+f+'.py');render_worker(base_template,f,wp);workers[f]=wp;qs=recover_v186_download(work,f,judge_policy,tasks,sha(wp))
   if qs is not None:
    try:qresults[f]=validate_decision_result_v185(qs,f,judge_policy,tasks)
    except Exception as e:print('existing_v172_qualification_invalid='+f+' '+str(e)[:500])
  pending=[f for f in missing if f not in qresults];print('v172_judges_pending='+(','.join(pending) if pending else 'none'))
  if pending:
   state=work/'remote_lifecycle_judge_v186.json';tx.reconcile_state(ctl,kaggle,state);qdata=td/'judge-helper-dataset';qdata.mkdir()
   for n in ['mc10b1_kaggle_worker.py','mc10b1_transport_common.py','build_mc10b1_portfolio_pilot_v1_1.py']:shutil.copy2(mcroot/n,qdata/n)
   shutil.copy2(effective_policy_path,qdata/'mc10d_public_judge_qualification_policy_v1.json');shutil.copy2(pkg120/'qualification'/'mc10d_public_judge_role_tasks_v1.json',qdata/'mc10d_public_judge_role_tasks_v1.json');hsha={n:sha(qdata/n) for n in ['mc10b1_kaggle_worker.py','mc10b1_transport_common.py','build_mc10b1_portfolio_pilot_v1_1.py']};wj(qdata/'mc10d-public-judge-helper-config.json',{'artifact_id':'alice.MC10D.public-judge-helper-config.v1.2.0','helper_sha256':hsha,'policy_sha256':sha(qdata/'mc10d_public_judge_qualification_policy_v1.json'),'tasks_sha256':sha(qdata/'mc10d_public_judge_role_tasks_v1.json'),'model_pull_timeout_seconds':5400,'private_candidate_files':0,'hidden_MC8_files':0,'scratch_relocation_only':True});short=tx.get_or_create_run_token(work/'RUN_STATE_v186.json','judge186');qdref=KAGGLE_USERNAME+'/'+tx.make_slug('alice-mc10d','j186'+short,'data');wj(qdata/'dataset-metadata.json',{'title':('ALICE MC10D public judge scratch recovery '+short)[:50],'id':qdref,'licenses':[{'name':'other'}],'description':'Public fictional judge-role qualification. No private candidate or hidden evaluator data. Frozen model/tasks; ratified GLM thinking-off profile amendment; decision-centric scoring.'});created=False;delete_allowed=False
   try:
    tx.create_or_reuse_dataset(ctl,kaggle,qdata,qdref,state,'PUBLIC_FICTIONAL_JUDGE_HELPERS_V172_SCRATCH_RECOVERY');created=True;delete_allowed=True;delete_allowed=False;pf=tx.cpu_preflight(ctl,kaggle,pkg130/'mc10d_public_judge_helper_preflight_v121.py',qdref,td,work/'judge_qualification'/'dataset_preflight_v185','alice-mc10d-jp185',short,'mc10d_public_judge_helper_preflight.json','pass',state,3);delete_allowed=True;req(rj(pf)['pass'] is True,'v1.7.2 public judge helper preflight failed');print('kaggle_cpu_public_judge_helper_preflight_v172_pass=true')
    for f in pending:
     wp=workers[f];kdir=td/('judge-'+f);kdir.mkdir();shutil.copy2(wp,kdir/'script.py');kslug=tx.make_slug('alice-mc10d-jg',short,'gpu186',family=f);kref,kmeta=tx.kernel_metadata(KAGGLE_USERNAME,kslug,'script.py',[qdref,SHARED_RUNTIME_DATASET],gpu=True,internet=True);wj(kdir/'kernel-metadata.json',kmeta);dl=work/'judge_qualification'/('download-v186-'+f+'-'+short);delete_allowed=False;st=tx.run_kernel_once(ctl,kaggle,kdir,kref,dl,state,'PUBLIC_JUDGE_QUAL_V186_DECISION_CENTRIC_GLM_PROFILE_'+f,a.timeout_hours,a.poll_seconds,True);delete_allowed=True;fail=next(iter(dl.rglob('mc10d_public_judge_qualification_failure.json')),None);res=next(iter(dl.rglob('mc10d_public_judge_qualification_result.json')),None)
     if fail is not None:
      fo=rj(fail);fm=str(fo.get('message',''));print('v172_public_judge_failure_receipt='+str(fail)+' sha256='+sha(fail))
      if 'insufficient ephemeral disk before model pull' in fm or 'no writable scratch filesystem available' in fm:raise DeterministicResourceStop('public judge scratch capacity still insufficient family='+f+' '+fm[:2000])
      raise E('public judge qualification failed '+f+' '+fail.read_text(encoding='utf-8',errors='replace')[:3000])
     req(st=='COMPLETE' and res is not None,'public judge qualification result missing '+f);src=res.parent;qdir=work/'judge_qualification'/f;archive_prior_qdir(qdir,work,f);shutil.copytree(src,qdir);wj(qdir/'qualification_request.json',{'family':f,'worker_script_sha256':sha(wp),'private_candidate_files':0,'hidden_MC8_files':0,'scratch_relocation_only':True,'original_v120_worker_sha256':EXPECTED_PUBLIC_JUDGE_WORKER_V120,'amended_worker_base_sha256':EXPECTED_AMENDED_WORKER,'clarified_worker_base_sha256':EXPECTED_CLARIFIED_WORKER,'decision_worker_base_sha256':EXPECTED_DECISION_WORKER,'budget_amendment_ratification_sha256':EXPECTED_BUDGET_RATIFICATION,'source_target_clarification_ratification_sha256':EXPECTED_CLARIFICATION_RATIFICATION,'decision_scoring_ratification_sha256':EXPECTED_SCORING_RATIFICATION,'glm_profile_ratification_sha256':EXPECTED_PROFILE_RATIFICATION,'effective_policy_sha256':EXPECTED_EFFECTIVE_POLICY,'num_predict_ladder':[2048,4096,6144],'qualification_rebase_version':'1.8.6'});qresults[f]=validate_decision_result_v185(qdir,f,judge_policy,tasks);print('v182_public_judge_qualification_bound=true family='+f+' worker_sha256='+sha(wp)+' budget_ratification_sha256='+EXPECTED_BUDGET_RATIFICATION+' clarification_ratification_sha256='+EXPECTED_CLARIFICATION_RATIFICATION)
   finally:
    if created and delete_allowed:tx.cleanup_dataset_exact(ctl,kaggle,qdref,'COMPLETED_PUBLIC_JUDGE_HELPER_DATASET_CLEANUP_V172',strict=True);tx.update_state(state,closed=True,dataset_ref=None)
    elif created:print('remote_public_qualification_dataset_preserved_due_unretrieved_or_unreconciled_kernel=true ref='+qdref)
  for f in missing:
   qdir=work/'judge_qualification'/f;req(qdir.is_dir(),'qualification dir missing '+f);validate_decision_result_v185(qdir,f,judge_policy,tasks)
  print('v186_all_missing_public_judges_qualified=true families=gemma,glm decision_scoring_v2=true glm_profile=thinking_off ladder=2048,4096,6144')
  # Handoff to unchanged v1.7.0 controller. Its Stage A must recover the already-validated final, and its Stage B must recover the qdirs above; no repair GPU or public-judge GPU remains pending.
  v170p=patched_v170_controller(pkg170,td);env2=os.environ.copy();env2['ALICE_MC10D_V170_ROOT']=str(pkg170);cmd=[sys.executable,str(v170p),'--repo-root',str(repo),'--vault-root',str(vault),'--freeze-bundle',str(freezezip),'--output',str(a.output),'--timeout-hours',str(a.timeout_hours),'--poll-seconds',str(a.poll_seconds)];print('v186_handoff_to_v170_refreeze_with_decision_validation_and_glm_profile_amendment=true');cp=subprocess.run(cmd,env=env2);req(cp.returncode in {0,75,76},'v170 refreeze returned unexpected code '+str(cp.returncode));return cp.returncode

if __name__=='__main__':
 try:
  code=main();sys.exit(int(code or 0))
 except DeterministicResourceStop as e:
  print('MC10D_V172_DETERMINISTIC_RESOURCE_STOP: '+str(e),file=sys.stderr);print('automatic_model_substitution=false',file=sys.stderr);print('successful_slot64_state_preserved=true',file=sys.stderr);sys.exit(76)
 except Exception as e:
  txm=sys.modules.get('mc10d_transport_v160')
  if txm is not None and isinstance(e,getattr(txm,'RemoteReconciliationPause',())):
   print('MC10D_V172_SAFE_PAUSE: '+str(e),file=sys.stderr);print('remote_state_preserved=true',file=sys.stderr);sys.exit(75)
  if txm is not None and isinstance(e,(getattr(txm,'DeterministicKernelError',()),getattr(txm,'DeterministicPreflightError',()))):
   print('MC10D_V172_DETERMINISTIC_TRANSPORT_STOP: '+str(e),file=sys.stderr);print('automatic_retry_performed=false',file=sys.stderr);sys.exit(76)
  print('MC10D_V172_FAILED: '+str(e),file=sys.stderr);raise
