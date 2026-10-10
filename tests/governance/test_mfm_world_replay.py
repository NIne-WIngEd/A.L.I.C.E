from copy import deepcopy
import importlib
import unittest
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.experience import ExperienceEvent
from cognitive_kernel.ledger import ExperienceLedgerEntryReceipt
from scripts.mfm import formation_world_contracts as C
from tests.governance.mfm_world_fixtures import World, seal


class ReplayTests(unittest.TestCase):
    def changed_payload(self,world,index,change):
        record=C.read_json(world.rows[index]['artifact'],world.read,C.ConstructionLimits(30,100000,100000))
        change(record['payload'])
        e=deepcopy(record['event']);e.pop('event_sha256');e.pop('event_id')
        e['scope']=C.scope(e['scope']);e['provenance']=C.contracts.ProvenanceReference(**{
            **e['provenance'],'source_reference_ids':tuple(e['provenance']['source_reference_ids']),
            'supersedes_record_ids':tuple(e['provenance']['supersedes_record_ids'])})
        e['content_digest']=record['payload']['source_artifact']['file_sha256'] if record['payload_kind']=='observation' else C.canonical_sha256(record['payload'])
        for k in ('deletion_lineage','parent_event_ids','outcome_reference_ids','policy_bindings'):e[k]=tuple(e[k])
        event=ExperienceEvent.create(**e);record['event']=event.metadata_record()
        old=record['entry_receipt'];record['entry_receipt']=ExperienceLedgerEntryReceipt.create(sequence=old['sequence'],
            event_id=event.event_id,event_sha256=event.event_sha256,previous_entry_sha256=old['previous_entry_sha256'],committed_at=old['committed_at']).record()
        record=seal({k:v for k,v in record.items() if k!='record_sha256'})
        manifest=deepcopy(world.world);manifest['branches'][0]['records'][index]={'sequence':index+1,'record_sha256':record['record_sha256'],
            'artifact':world.add('changed-record-'+str(index),record)}
        manifest=seal({k:v for k,v in manifest.items() if k!='world_sha256'},'world_sha256')
        cut=seal({'schema':'mfm-world-cut-v1','world_id':'world-a','branch_id':'main','sequence':index+1,'head_sha256':record['record_sha256']})
        return manifest,cut

    def test_original_registration_speaker_and_clock_cannot_be_reauthored(self):
        for field in ('speaker_ref','observed_at'):
            world=World();manifest,cut=self.changed_payload(world,0,lambda p:p.update({field:'different-person' if field=='speaker_ref' else '2020-01-01T00:00:00.000000Z'}))
            with self.assertRaises(CognitiveKernelContractError):self.replay(world,manifest=manifest,cut=cut)

    def test_namespace_profile_is_actual_scope_derived_even_empty(self):
        world=World();manifest=deepcopy(world.world);manifest['authority_profile']['authority_id']='fabricated-authority'
        manifest=seal({k:v for k,v in manifest.items() if k!='world_sha256'},'world_sha256')
        genesis=C.canonical_sha256({'schema':'mfm-world-genesis-v1','world_id':'world-a','scope':manifest['scope'],'namespace':manifest['namespace'],
            'ledger_profile':manifest['ledger_profile'],'authority_profile':manifest['authority_profile']})
        cut=seal({'schema':'mfm-world-cut-v1','world_id':'world-a','branch_id':'main','sequence':0,'head_sha256':genesis})
        with self.assertRaises(CognitiveKernelContractError):self.replay(world,manifest=manifest,cut=cut)

    def test_current_version_and_governance_source_are_not_inferred(self):
        for defect in ('version','future-source'):
            world=World()
            def change(payload):
                gov=C.read_json(payload['governance_ref'],world.read,C.ConstructionLimits(30,100000,100000))
                if defect=='future-source':gov['source_refs']=['future-source']
                else:
                    req=C.read_json(payload['append_request'],world.read,C.ConstructionLimits(30,100000,100000));req['expected_current_claim_version_id']='unseen-version'
                    payload['append_request']=world.add('wrong-current-request',req);gov['expected_current_claim_version_id']='unseen-version'
                gov=seal({k:v for k,v in gov.items() if k!='record_sha256'});payload['governance_ref']=world.add('changed-governance',gov)
            manifest,cut=self.changed_payload(world,4,change)
            with self.assertRaises(CognitiveKernelContractError):self.replay(world,manifest=manifest,cut=cut)

    def replay(self, world, n=5, *, manifest=None, cut=None, read=None, limits=None):
        api=importlib.import_module('scripts.mfm.formation_world_replay')
        record=world.world if manifest is None else manifest; boundary=world.cut(n) if cut is None else cut
        contract=C.world_contract_binding()
        return api.replay_world_prefix(record,world_sha256=record['world_sha256'],cut=boundary,
            cut_sha256=boundary['record_sha256'],read_artifact=world.read if read is None else read,
            limits=limits or C.ConstructionLimits(30,100000,100000),contract_binding=contract,
            contract_binding_sha256=contract['record_sha256'])

    def test_backdated_future_never_opens_before_cut(self):
        world=World(); forbidden={r['artifact']['artifact_id'] for r in world.rows[2:]}
        def read(ref):
            self.assertNotIn(ref['artifact_id'],forbidden); return world.read(ref)
        earlier=self.replay(world,2,read=read)
        self.assertEqual(earlier['current_claims'][0]['records']['version']['value']['value']['text'],'dark mode')
        self.assertEqual(earlier['authority_source_position'],1)
        later=self.replay(world,5)
        self.assertEqual(later['current_claims'][0]['records']['version']['value']['value']['text'],'light mode')
        self.assertLess(C.read_json(world.rows[4]['artifact'],world.read,C.ConstructionLimits(30,100000,100000))['event']['occurred_at'],
                        C.read_json(world.rows[0]['artifact'],world.read,C.ConstructionLimits(30,100000,100000))['event']['occurred_at'])
        self.assertEqual(later['authority_generation'],1); self.assertEqual(later['authority_source_position'],2)
        self.assertEqual(later['cut']['sequence'],5)

    def test_fork_uses_exact_shared_prefix_and_never_sibling(self):
        main=World(); alternate=World(corrected='sepia mode',later_prefix='fork-')
        fork_rows=deepcopy(alternate.rows[4:])
        artifacts={**main.artifacts,**alternate.artifacts}
        def read(ref):
            return artifacts[ref['artifact_id']]
        parent=main.cut(4)
        manifest=deepcopy(main.world); manifest['branches'].append({'branch_id':'fork','parent_cut':parent,'records':fork_rows})
        manifest=seal({k:v for k,v in manifest.items() if k!='world_sha256'},'world_sha256')
        fork_cut=seal({'schema':'mfm-world-cut-v1','world_id':'world-a','branch_id':'fork','sequence':5,
                       'head_sha256':fork_rows[0]['record_sha256']})
        view=self.replay(main,manifest=manifest,cut=fork_cut,read=read)
        self.assertEqual(view['current_claims'][0]['records']['version']['value']['value']['text'],'sepia mode')
        self.assertEqual(self.replay(main,manifest=manifest)['current_claims'][0]['records']['version']['value']['value']['text'],'light mode')
        for fault in ('cycle','head','gap'):
            bad=deepcopy(manifest)
            if fault=='cycle': bad['branches'][1]['parent_cut']=fork_cut
            elif fault=='head': bad['branches'][1]['parent_cut']['head_sha256']='0'*64
            else: bad['branches'][1]['records'][0]['sequence']=6
            bad=seal({k:v for k,v in bad.items() if k!='world_sha256'},'world_sha256')
            with self.assertRaises(CognitiveKernelContractError): self.replay(main,manifest=bad,cut=fork_cut,read=read)

    def test_deny_and_visibility_preserve_original_state(self):
        world=World(); five=self.replay(world,5); denied=self.replay(world,6); hidden=self.replay(world,7)
        self.assertEqual(five['current_claims'],denied['current_claims'])
        self.assertEqual(hidden['current_claims'],five['current_claims'])
        self.assertEqual(hidden['visibility']['claim-1'],'withheld')
        deleted=World(deleted=True); view=self.replay(deleted,5)
        self.assertEqual(view['visibility']['claim-1'],'deleted')

    def test_real_idempotent_confirmation_never_regresses_current(self):
        world=World(confirmation=True); before=self.replay(world,7); after=self.replay(world,8)
        self.assertEqual(before['current_claims'],after['current_claims'])
        self.assertEqual(after['authority_source_position'],2)

    def test_zero_and_complete_limits_no_partial_result(self):
        world=World(); view=self.replay(world,0)
        self.assertEqual(view['current_claims'],[]); self.assertEqual(view['authority_source_position'],0)
        world.calls.clear()
        with self.assertRaises(CognitiveKernelContractError): self.replay(world,5,limits=C.ConstructionLimits(4,100000,100000))
        self.assertEqual(world.calls,[])

    def test_manifest_extra_teacher_and_contract_change_refuse(self):
        world=World(); bad=deepcopy(world.world); bad['private_truth']={}
        bad=seal({k:v for k,v in bad.items() if k!='world_sha256'},'world_sha256')
        with self.assertRaises(CognitiveKernelContractError): self.replay(world,manifest=bad)

    def test_future_tail_changes_only_manifest_custody(self):
        world=World(); first=self.replay(world,2); altered=deepcopy(world.world)
        altered['branches'][0]['records'][-1]['artifact']['file_sha256']='0'*64
        altered=seal({k:v for k,v in altered.items() if k!='world_sha256'},'world_sha256')
        self.assertEqual(first,self.replay(world,2,manifest=altered))
