"""Exact v186 public fictional prompt and response schema; no scoring labels enter requests."""
import json

def canon(o):
    return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False)

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

SYSTEM = 'You are a strict pointwise personality-completion judge. The fixed_core describes a fictional SOURCE persona and target.actor names the fictional TARGET persona being completed; those names may intentionally differ. Do not treat that source-to-target name difference by itself as actor mismatch, contradiction, or history transfer. Evaluate actor/role/state/context/direction against target and candidate_behavior. Personality and behavioral traits may bridge from the source core to target behavior, but source-person events, source-person history, and lived memories do not transfer to the target without explicit evidence. Compatible novelty is allowed without an exact historical episode; contradiction, arbitrary unbridged behavior, actor/role/state/context reversal, fake source history, and fake lived memory are not. Return only JSON.'
