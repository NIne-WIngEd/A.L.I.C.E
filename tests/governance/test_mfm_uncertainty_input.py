import unittest
from copy import deepcopy
from cognitive_kernel.canonical import CognitiveKernelContractError,canonical_json_bytes
from scripts.mfm import formation_uncertainty_input as U
from tests.governance.mfm_task_content_fixtures import TaskOriginals,seal,LATER,SCOPE,NAMESPACE


class UncertaintyTests(unittest.TestCase):
    def setUp(self):self.f=TaskOriginals();self.row=self.f.uncertainty()

    def original(self,row):
        raw,parts,kw=self.f.arguments('uncertainty',row);binding=U.uncertainty_input_binding()
        kw.update(contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        return U.uncertainty_original(raw,parts,**kw)

    def revised(self,state,version=2):
        row=deepcopy(self.row);row.update(state=state,content_version=version,previous_content_sha256=self.row['record_sha256'],updated_at=LATER)
        return row

    def observation(self,suffix=''):
        return {'observation_ref':{'ref_id':'resolution-source'+suffix,'registration_sha256':('6' if not suffix else '8')*64,'known_at_entry_sha256':('7' if not suffix else '9')*64,
            'artifact':self.f.text('resolution-source'+suffix,'Observer saw disconnected wire.'),'availability':'present'},
            'selected_alternative_ids':['wire'],'report':self.f.text('resolution-report'+suffix,'Observer attributes darkness to wire.'),'source_event_refs':[]}

    def test_open_original_has_distinct_alternatives_without_inferred_winner(self):
        value=self.original(self.row);self.assertEqual(value['record']['state'],'open');self.assertFalse(value['causal_dependencies_verified'])
        self.assertNotIn('winner',value);self.assertEqual(value['texts']['alt-switch'],'Switch not pressed')

    def test_resolve_reopen_retains_historical_observation_and_referenced_alternative(self):
        row=self.revised('resolved');row['resolution_observations']=[self.observation()];row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        previous=self.original(self.row);resolved=self.original(row);U.uncertainty_successor(previous,resolved)
        reopened=deepcopy(row);reopened.update(state='reopened',content_version=3,previous_content_sha256=row['record_sha256'],updated_at='2026-08-01T12:00:00.000000Z')
        reopened['alternatives']=reopened['alternatives'][1:];reopened=seal({k:v for k,v in reopened.items() if k!='record_sha256'})
        U.uncertainty_successor(resolved,self.original(reopened))
        bad=deepcopy(reopened);bad['alternatives']=bad['alternatives'][:1];bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.original(bad)
        bad=deepcopy(reopened);bad['resolution_observations']=[];bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):U.uncertainty_successor(resolved,self.original(bad))

    def test_wrong_identity_missing_resolution_and_question_rewrite_refuse(self):
        for mutation in ('identity','state','alternatives'):
            row=deepcopy(self.row)
            if mutation=='identity':row['identity_registration']['kind']='claim'
            elif mutation=='state':row['state']='resolved'
            else:row['alternatives']=row['alternatives'][:1]
            row=seal({k:v for k,v in row.items() if k!='record_sha256'})
            with self.subTest(mutation=mutation),self.assertRaises(CognitiveKernelContractError):self.original(row)
        row=self.revised('open');row['question']=self.f.text('changed-question','What time?');row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):U.uncertainty_successor(self.original(self.row),self.original(row))

    def test_every_declared_state_edge_and_retired_terminal(self):
        allowed={'open':{'open','resolved','retired'},'resolved':{'resolved','reopened','retired'},'reopened':{'reopened','resolved','retired'},'retired':set()}
        for before in allowed:
            for after in allowed:
                a=deepcopy(self.row);a['state']=before
                if before in ('resolved','reopened'):a['resolution_observations']=[self.observation()]
                a=seal({k:v for k,v in a.items() if k!='record_sha256'})
                b=deepcopy(a);b.update(state=after,content_version=2,previous_content_sha256=a['record_sha256'],updated_at=LATER)
                if after=='resolved' and before in ('open','reopened'):
                    b['resolution_observations'].append(self.observation('-later'))
                b=seal({k:v for k,v in b.items() if k!='record_sha256'})
                with self.subTest(before=before,after=after):
                    if after in allowed[before]:U.uncertainty_successor(self.original(a),self.original(b))
                    else:
                        with self.assertRaises(CognitiveKernelContractError):U.uncertainty_successor(self.original(a),self.original(b))

    def test_repackaged_same_original_observation_is_not_new_resolution(self):
        first=self.observation();row=self.revised('resolved');row['resolution_observations']=[first]
        row=seal({k:v for k,v in row.items() if k!='record_sha256'});reopened=deepcopy(row)
        reopened.update(state='reopened',content_version=3,previous_content_sha256=row['record_sha256'],updated_at='2026-08-01T12:00:00.000000Z')
        reopened=seal({k:v for k,v in reopened.items() if k!='record_sha256'});later=deepcopy(reopened)
        duplicate=deepcopy(first);duplicate['report']=self.f.text('same-observation-new-report','Observer attributes darkness to wire.')
        later.update(state='resolved',content_version=4,previous_content_sha256=reopened['record_sha256'],updated_at='2026-08-01T13:00:00.000000Z')
        later['resolution_observations'].append(duplicate);later=seal({k:v for k,v in later.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.original(later)

if __name__=='__main__':unittest.main()
