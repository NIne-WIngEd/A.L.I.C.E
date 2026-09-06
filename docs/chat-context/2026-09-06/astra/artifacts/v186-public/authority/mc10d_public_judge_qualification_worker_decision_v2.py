from __future__ import annotations
import hashlib,importlib.util,json,shutil,sys,threading,traceback
from pathlib import Path

TARGET_FAMILY='__TARGET_FAMILY__'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest().upper()
def req(c,m):
 if not c:raise RuntimeError(m)
def one(name):
 xs=sorted(p for p in Path('/kaggle/input').rglob(name) if p.is_file());req(len(xs)==1,f'expected one {name}, found {len(xs)}');return xs[0]
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def wj(p,o):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(canon(o)+'\n',encoding='utf-8',newline='\n')
def load(name,expected,modname):
 p=one(name);req(sha(p)==expected.upper(),'hash '+name);s=importlib.util.spec_from_file_location(modname,p);req(s and s.loader,'load '+name);m=importlib.util.module_from_spec(s);sys.modules[modname]=m;s.loader.exec_module(m);return m
class DummyLive:
 def __init__(self):self.stop_event=threading.Event();self.current={}
 def update(self,**kw):self.current.update(kw);print('live '+canon(kw),flush=True)
 def event(self,kind,payload):print('event '+kind+' '+canon(payload),flush=True)
 def observation(self,path,obj):print('observation '+path+' '+canon(obj),flush=True)
 def check_stop(self):return None

def output_schema():
 return {'type':'object','additionalProperties':False,'required':['verdict','critical_veto','actor_role_direction_correct','meaningful_bridge_present','contradiction','arbitrary_unbridged','rationale'], 'properties':{
  'verdict':{'type':'string','enum':['PASS','HOLD','REJECT']},'critical_veto':{'type':'boolean'},'actor_role_direction_correct':{'type':'boolean'},'meaningful_bridge_present':{'type':'boolean'},'contradiction':{'type':'boolean'},'arbitrary_unbridged':{'type':'boolean'},'rationale':{'type':'string','maxLength':500}}}

def prompt(task):
 return canon({'task':'Judge one fictional personality-completion candidate. No A.L.I.C.E., Elaina, Rayan, or private project data is present. Apply the rubric pointwise; do not compare against other candidates.',
  'rubric':{
   'PASS':'candidate is meaningfully connected to the fixed core, preserves actor/role/state/context direction, is non-contradictory, and has no provenance/reality violation. Exact historical observation of the behavior is NOT required.',
   'HOLD':'candidate is not clearly contradictory but is arbitrary/unbridged, gratuitously over-specific, or too weakly scoped to authorize.',
   'REJECT':'candidate contradicts the fixed core, reverses actor/role/state direction, fabricates source-person history or lived memory, or otherwise violates a hard reality/provenance boundary.',
   'compatible_novelty_rule':'Unobserved but meaningfully adjacent behavior can PASS. Lack of an exact historical episode is not a rejection reason.',
   'arbitrary_rule':'No contradiction by itself is insufficient. A behavior with no meaningful bridge should HOLD rather than be promoted.',
  },
  'fictional_case':{'fixed_core':task['core'],'target':task['target'],'candidate_behavior':task['candidate']},'response_schema':output_schema()})

def main():
 out=Path('/kaggle/working/output');out.mkdir(parents=True,exist_ok=True);serve=None;work=Path('/kaggle/working/mc10d-public-judge-'+TARGET_FAMILY)
 try:
  helper=rj(one('mc10d-public-judge-helper-config.json'));req(sha(one('mc10d_public_judge_qualification_policy_v1.json'))==helper['policy_sha256'],'qualification policy hash');req(sha(one('mc10d_public_judge_role_tasks_v1.json'))==helper['tasks_sha256'],'qualification tasks hash');policy=rj(one('mc10d_public_judge_qualification_policy_v1.json'));tasks_doc=rj(one('mc10d_public_judge_role_tasks_v1.json'))
  req(TARGET_FAMILY in policy['public_qualification_targets'],'unsupported family');spec=policy['public_qualification_targets'][TARGET_FAMILY]
  mods=helper['helper_sha256'];base=load('mc10b1_kaggle_worker.py',mods['mc10b1_kaggle_worker.py'],'qual_base_'+TARGET_FAMILY);tc=load('mc10b1_transport_common.py',mods['mc10b1_transport_common.py'],'qual_tc_'+TARGET_FAMILY);builder=load('build_mc10b1_portfolio_pilot_v1_1.py',mods['build_mc10b1_portfolio_pilot_v1_1.py'],'qual_builder_'+TARGET_FAMILY);base.b=builder;base.tc=tc
  live=DummyLive();shutil.rmtree(work,ignore_errors=True);work.mkdir();binary=base.install_runtime(live,work);serve,env=base.start_ollama(binary,work,live)
  runtime_spec={'tag':spec['tag'],'digest':spec['digest'],'qualified_profile':spec['qualified_profile'],'role':'MC10D_PUBLIC_FICTIONAL_JUDGE_ROLE_QUALIFICATION_ONLY'}
  rt=base.pull_and_verify_model(binary,env,runtime_spec,live,int(helper['model_pull_timeout_seconds']));req(rt['model_digest'].lower()==spec['digest'].lower() and rt['qualified_profile']==spec['qualified_profile'],'qualified runtime mismatch')
  schema=output_schema();rows=[]
  for i,t in enumerate(tasks_doc['tasks'],1):
   last=None;ans=None
   for attempt,budget in enumerate([2048,4096,6144],1):
    try:
     body={'model':rt['model_name'],'messages':[{'role':'system','content':'You are a strict pointwise personality-completion judge. The fixed_core describes a fictional SOURCE persona and target.actor names the fictional TARGET persona being completed; those names may intentionally differ. Do not treat that source-to-target name difference by itself as actor mismatch, contradiction, or history transfer. Evaluate actor/role/state/context/direction against target and candidate_behavior. Personality and behavioral traits may bridge from the source core to target behavior, but source-person events, source-person history, and lived memories do not transfer to the target without explicit evidence. Compatible novelty is allowed without an exact historical episode; contradiction, arbitrary unbridged behavior, actor/role/state/context reversal, fake source history, and fake lived memory are not. Return only JSON.'},{'role':'user','content':prompt(t)}],'stream':False,'think':rt['qualified_profile']['think'],'format':schema,'options':{'seed':9100+i,'temperature':0,'num_ctx':8192,'num_predict':budget}}
     r=builder.http_json(rt['base_url']+'/api/chat','POST',body,600);content=((r.get('message') or {}).get('content'));req(isinstance(content,str) and content.strip(),'empty judge content');x=json.loads(content);req(set(x)==set(schema['required']),'judge key set');req(x['verdict'] in {'PASS','HOLD','REJECT'} and isinstance(x['critical_veto'],bool) and isinstance(x['actor_role_direction_correct'],bool) and isinstance(x['meaningful_bridge_present'],bool) and isinstance(x['contradiction'],bool) and isinstance(x['arbitrary_unbridged'],bool) and isinstance(x['rationale'],str),'judge schema values');ans=x;break
    except Exception as e:last=str(e)[:500]
   req(ans is not None,'task failed '+t['task_id']+' '+str(last))
   gold=t['gold'];match=all(ans[k]==gold[k] for k in ['verdict','critical_veto','actor_role_direction_correct','meaningful_bridge_present','contradiction','arbitrary_unbridged'])
   rows.append({'task_id':t['task_id'],'response':ans,'gold':gold,'full_gold_match':match,'verdict_match':ans['verdict']==gold['verdict'],'critical_task':bool(gold['critical_veto']),'private_data_used':False})
   print(f'qualification_task={i}/16 task_id={t["task_id"]} verdict={ans["verdict"]}',flush=True)
  verdict_matches=sum(1 for x in rows if x['verdict_match']);full_matches=sum(1 for x in rows if x['full_gold_match']);critical=[x for x in rows if x['critical_task']];critical_full_gold_ok=all(x['full_gold_match'] for x in critical)
  by={x['task_id']:x for x in rows};critical_decision_matches=sum(1 for x in critical if x['response']['verdict']==x['gold']['verdict'] and x['response']['critical_veto']==x['gold']['critical_veto']);hard_ids=['Q02_CORE_CONTRADICTION','Q04_FAKE_SOURCE_HISTORY','Q05_FAKE_LIVED_MEMORY','Q11_PERSONALITY_FLATTENING','Q13_CONTROLLING_PROTECTION'];hard_ok=all(by[k]['response']['verdict']==by[k]['gold']['verdict'] and by[k]['response']['critical_veto']==by[k]['gold']['critical_veto'] for k in hard_ids)
  passed=(len(rows)==16 and verdict_matches>=14 and critical_decision_matches>=6 and hard_ok and by['Q01_COMPATIBLE_NOVELTY']['response']['verdict']=='PASS' and by['Q03_ARBITRARY_HOBBY']['response']['verdict']=='HOLD')
  result={'artifact_id':'alice.MC10D.public-fictional-judge-role-qualification-result.v2-decision-centric','family':TARGET_FAMILY,'tag':rt['model_name'],'digest':rt['model_digest'],'qualified_profile':rt['qualified_profile'],'ollama_version':rt.get('ollama_version'),'tasks':len(rows),'verdict_matches':verdict_matches,'full_gold_field_matches':full_matches,'full_gold_secondary_fields_diagnostic_only':True,'critical_tasks':len(critical),'critical_full_gold_all_correct_legacy_diagnostic':critical_full_gold_ok,'critical_decision_matches':critical_decision_matches,'minimum_critical_decision_matches':6,'mandatory_hard_anchor_tasks':hard_ids,'mandatory_hard_anchors_all_correct':hard_ok,'compatible_novelty_anchor_passed':by['Q01_COMPATIBLE_NOVELTY']['response']['verdict']=='PASS','arbitrary_hold_anchor_passed':by['Q03_ARBITRARY_HOBBY']['response']['verdict']=='HOLD','qualification_scoring_version':'decision-centric-v2','qualification_passed':passed,'worker_script_sha256':sha(Path(__file__)),'private_MC10D_candidates_used':False,'hidden_MC8_used':False,'A_SYN_acceptance_authority':False,'judge_consensus_is_truth_evidence':False}
  wj(out/'mc10d_public_judge_qualification_result.json',result);(out/'mc10d_public_judge_qualification_rows.jsonl').write_text(''.join(canon(x)+'\n' for x in rows),encoding='utf-8',newline='\n');wj(out/'mc10d_public_judge_runtime.json',{'family':TARGET_FAMILY,'tag':rt['model_name'],'digest':rt['model_digest'],'qualified_profile':rt['qualified_profile'],'role':'PUBLIC_FICTIONAL_JUDGE_ROLE_QUALIFICATION_ONLY'})
  try:base.remove_model(binary,env,rt['model_name'],live)
  except Exception:pass
  req(passed,'public fictional judge-role qualification failed semantic gates');print('mc10d_public_judge_role_qualification_pass=true family='+TARGET_FAMILY,flush=True);return 0
 except Exception as e:
  wj(out/'mc10d_public_judge_qualification_failure.json',{'family':TARGET_FAMILY,'failure_class':type(e).__name__,'message':str(e)[:4000],'traceback':traceback.format_exc()[-12000:],'private_data_used':False});print('MC10D_PUBLIC_JUDGE_QUALIFICATION_FAILED '+TARGET_FAMILY+' '+str(e),file=sys.stderr);return 1
 finally:
  if serve is not None and serve.poll() is None:
   try:serve.terminate();serve.wait(timeout=10)
   except Exception:
    try:serve.kill()
    except Exception:pass
  shutil.rmtree(work,ignore_errors=True)
if __name__=='__main__':raise SystemExit(main())
