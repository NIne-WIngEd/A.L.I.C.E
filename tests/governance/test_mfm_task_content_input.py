import unittest
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from cognitive_kernel.canonical import CognitiveKernelContractError,canonical_json_bytes,canonical_sha256
from cognitive_kernel import mission as G
from scripts.mfm import formation_task_content_input as T
from tests.governance.mfm_task_content_fixtures import TaskOriginals,seal,LATER


class TaskContentTests(unittest.TestCase):
    def setUp(self):self.f=TaskOriginals()

    def test_real_originals_preserve_actor_observer_and_all_scalar_workspace_body(self):
        values={family:self.f.originals(family) for family in ('mission','attempt','result','outcome','workspace')}
        self.assertEqual(values['result']['typed']['capsule']['status'],'succeeded')
        self.assertEqual(values['outcome']['record']['postcondition_reports'][0]['reported_state'],'unmet')
        self.assertEqual(values['attempt']['record']['phase'],'planned')
        self.assertFalse(values['result']['authenticated_service_claimed'])
        self.assertNotIn('verified_success',values['result'])
        self.assertEqual(values['workspace']['typed']['visible'][0]['visible_fields']['float'],0.5)
        self.assertIn('ana ana',values['mission']['texts']['mission-title'])

    def test_raw_digest_projection_cannot_be_guessed_from_legacy_label_digest(self):
        row=deepcopy(self.f.records['mission']);row['title_projection']='canonical-json-sha256-v1';row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.f.originals('mission',row)
        raw,parts,kw=self.f.arguments('mission');parts['mission-title']=parts['mission-title']+b'!'
        with self.assertRaises(CognitiveKernelContractError):T.task_originals(raw,parts,**kw)

    def test_refs_are_complete_unique_and_zero_byte_is_not_missing(self):
        raw,parts,kw=self.f.arguments('result');kw['refs']['foreign']=deepcopy(kw['refs']['summary']) if 'summary' in kw['refs'] else deepcopy(next(iter(kw['refs'].values())))
        with self.assertRaises(CognitiveKernelContractError):T.task_originals(raw,parts,**kw)
        row=deepcopy(self.f.records['result']);row['output_artifacts'][0]['artifact']={'artifact_id':'empty','bytes':0,'file_sha256':sha256(b'').hexdigest(),'media_type':'application/octet-stream'}
        row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.f.originals('result',row)

    def test_full_roundtrip_unknown_field_bool_version_and_unauthorized_truth_refuse(self):
        for field,value in (('content_version',True),('verified_success',True),('namespace','foreign'),('created_at','2026-08-01T10:00:00Z')):
            row=deepcopy(self.f.records['result']);row[field]=value;row=seal({k:v for k,v in row.items() if k!='record_sha256'})
            with self.subTest(field=field),self.assertRaises(CognitiveKernelContractError):self.f.originals('result',row)

    def test_graph_duplicate_detached_and_future_revision_refuse(self):
        row=deepcopy(self.f.records['mission']);graph=deepcopy(self.f.graph.metadata_record());graph['nodes'].append(deepcopy(graph['nodes'][0]))
        row['graph_artifact']=self.f.json('bad-graph',graph);row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.f.originals('mission',row)

    def test_text_summary_and_traceback_rationale_original_corruption_refuse(self):
        for part in ('result-summary','rationale'):
            raw,parts,kw=self.f.arguments('result');parts[part]+=b'changed'
            with self.subTest(part=part),self.assertRaises(CognitiveKernelContractError):T.task_originals(raw,parts,**kw)

    def test_plan_dag_foreign_and_cycle_and_attempt_later_step_refuse(self):
        for dep in ('step-1','future-step'):
            row=deepcopy(self.f.records['mission']);row['node_contents'][0]['plan_steps'][0]['depends_on_step_ids']=[dep]
            row=seal({k:v for k,v in row.items() if k!='record_sha256'})
            with self.subTest(dep=dep),self.assertRaises(CognitiveKernelContractError):self.f.originals('mission',row)
        row=deepcopy(self.f.records['attempt']);row['related_plan_step_ids']=['future-step'];row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):T.validate_dependency_originals(self.f.originals('attempt',row),
            {('mission',self.f.mission.mission_id,1):self.f.originals('mission')})

    def test_changed_graph_node_uses_actual_successor_not_timestamp_winner(self):
        previous=self.f.originals('mission');row=deepcopy(self.f.records['mission'])
        for status,execution,accepted in (('completed','succeeded',True),('planned','idle',False)):
            node=G.MissionNode.create(**{k:v for k,v in self.f.root.__dict__.items() if k not in ('node_id','node_sha256')}
                |{'status':status,'execution_state':execution,'updated_at':LATER})
            graph=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=node.node_id,nodes=(node,),edges=(),
                revision=2,snapshot_at=LATER,previous_snapshot_sha256=self.f.graph.snapshot_sha256)
            current=deepcopy(row);current.update(content_version=2,previous_content_sha256=row['record_sha256'],updated_at=LATER,
                graph_artifact=self.f.json('graph-node-'+status,graph.metadata_record()));current['node_contents'][0]['node_sha256']=node.node_sha256
            current=self.f.originals('mission',seal({k:v for k,v in current.items() if k!='record_sha256'}))
            if accepted:T.task_successor(previous,current)
            else:
                with self.assertRaises(CognitiveKernelContractError):T.task_successor(previous,current)

    def test_attempt_phase_edges_preserve_original_action_and_cancelled_is_terminal(self):
        allowed={'planned':{'planned','attempted','cancelled'},'attempted':{'attempted','cancelled'},'cancelled':set()}
        for before in allowed:
            for after in allowed:
                a=deepcopy(self.f.records['attempt']);a['phase']=before;a=seal({k:v for k,v in a.items() if k!='record_sha256'})
                b=deepcopy(a);b.update(phase=after,content_version=2,previous_content_sha256=a['record_sha256'],updated_at=LATER)
                b=seal({k:v for k,v in b.items() if k!='record_sha256'})
                with self.subTest(before=before,after=after):
                    if after in allowed[before]:T.task_successor(self.f.originals('attempt',a),self.f.originals('attempt',b))
                    else:
                        with self.assertRaises(CognitiveKernelContractError):T.task_successor(self.f.originals('attempt',a),self.f.originals('attempt',b))

    def test_successor_unchanged_node_reuse_and_changed_node_actual_transition(self):
        previous=self.f.originals('mission');row=deepcopy(self.f.records['mission'])
        graph=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=self.f.root.node_id,nodes=(self.f.root,),edges=(),
            revision=2,snapshot_at=LATER,previous_snapshot_sha256=self.f.graph.snapshot_sha256)
        row.update(content_version=2,previous_content_sha256=row['record_sha256'],updated_at=LATER,graph_artifact=self.f.json('graph-2',graph.metadata_record()))
        current=self.f.originals('mission',seal({k:v for k,v in row.items() if k!='record_sha256'}));T.task_successor(previous,current)
        bad=deepcopy(current);bad['record']['previous_content_sha256']='9'*64
        with self.assertRaises(CognitiveKernelContractError):T.task_successor(previous,bad)

    def test_structural_refs_are_not_claimed_causal_knowledge(self):
        result=self.f.originals('outcome');self.assertFalse(result['causal_dependencies_verified'])
        self.assertEqual(result['dependencies'][0]['content']['family'],'attempt')

    def test_derived_attention_priority_is_verified_as_original_not_ignored(self):
        row=deepcopy(self.f.records['workspace']);original=deepcopy(self.f.originals('workspace')['typed']['attention'])
        original['entries'][0]['priority_order']+=1;row['attention_artifact']=self.f.json('bad-attention',original)
        row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.f.originals('workspace',row)

    def test_outcome_condition_membership_is_connected_to_exact_attempt_original(self):
        attempt=self.f.originals('attempt');outcome=self.f.originals('outcome')
        T.validate_dependency_originals(outcome,{('attempt','attempt-1',1):attempt})
        self.assertEqual(T.validate_dependency_originals(outcome,{('attempt','attempt-1',1):attempt})['condition_availability'],
            [{'condition_id':'light','availability':'present'},{'condition_id':'heat','availability':'not_supplied'}])
        row=deepcopy(self.f.records['outcome']);row['postcondition_reports'][0]['condition_id']='future-condition';row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):T.validate_dependency_originals(self.f.originals('outcome',row),{('attempt','attempt-1',1):attempt})

if __name__=='__main__':unittest.main()
