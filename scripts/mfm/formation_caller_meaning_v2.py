"""Explicit Episode companion beside unchanged caller and Claim meaning v1."""
from base64 import b64decode,b64encode
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from . import formation_caller_input as F,formation_caller_meaning as M,formation_episode_input as E,formation_world_contracts as C,formation_world_replay_v2 as R

SCHEMA='mfm-formation-caller-meaning-v2'
PROFILE='text-causal-claim-episode-joint-v1'
EPISODE_SNAPSHOT_SCHEMA='mfm-episode-semantic-snapshot-v1'
EPISODE_FAMILY='episode-snapshot-v1'
PARTS=('episode','full_content','summary_content')
MEMBERS={'evidence':'member_evidence_ids','claim_versions':'member_claim_version_ids','candidates':'member_candidate_ids',
         'missions':'mission_node_ids','participants':'participant_ids'}


def normative_view_policy_v2():
    return C.seal({'schema':'mfm-caller-meaning-view-policy-v2','profile':PROFILE,
        'families':[*M.FAMILIES,EPISODE_FAMILY],'eligible_episode_states':['accepted','disputed','superseded'],
        'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'episode_binding':E.episode_input_binding(),
        'historical_replay_binding':R.world_contract_binding_v2(),
        'old_family_policy':M.normative_view_policy(),'teacher_input':False,'authority':False})


def _snapshot(row,expected,context,registry,claim_snapshot):
    C.pin(row,expected,EPISODE_SNAPSHOT_SCHEMA)
    C.exact(row,{'schema','scope','namespace','projection_profile_sha256','cut','episodes','read_service_id',
        'read_receipt_sha256','policy_sha256','historical_prefix_proof','record_sha256'},'Episode snapshot')
    C.require(C.scope(row['scope'])==context.base.scope and row['namespace']==context.base.authority_namespace_id,'foreign Episode snapshot')
    C.digest(row['projection_profile_sha256']);C.identifier(row['read_service_id'])
    for name in ('read_receipt_sha256','policy_sha256'):
        C.digest(row[name]);C.require(row[name]==claim_snapshot[name],'same current read boundary required')
    C.require(row['read_service_id']==claim_snapshot['read_service_id'],'Episode read service differs')
    cut=row['cut'];C.exact(cut,{'schema','world_id','branch_id','sequence','head_sha256','record_sha256'},'Episode causal cut')
    C.pin(cut,cut['record_sha256'],'mfm-world-cut-v1');C.identifier(cut['world_id']);C.identifier(cut['branch_id']);C.integer(cut['sequence']);C.digest(cut['head_sha256'])
    historical=R.prefix_proof_replay(row['historical_prefix_proof'],cut=cut)
    C.require(historical['scope']==row['scope'] and historical['namespace']==row['namespace'] and
        historical['episode_profile']['profile_sha256']==row['projection_profile_sha256'],'historical proof scope/profile differs')
    C.require(claim_snapshot['authority_generation']==historical['authority_generation'] and
        claim_snapshot['source_position']==historical['authority_source_position'],'Claim/world causal counters differ')
    actual_episodes={r['ref_id']:r for r in historical['episodes']}
    C.require(type(row['episodes']) is list,'Episode snapshot inventory differs')
    seen=set()
    for item in row['episodes']:
        C.exact(item,{'ref_id','episode_sha256','generation','known_at_sequence','known_at_entry_sha256','member_resolution','member_registrations'},'Episode known-at row')
        C.identifier(item['ref_id']);C.digest(item['episode_sha256']);C.digest(item['known_at_entry_sha256'])
        C.integer(item['generation'],1);C.integer(item['known_at_sequence'],1)
        C.require(item['known_at_sequence']<=cut['sequence'] and item['ref_id'] not in seen,'future/duplicate Episode snapshot')
        C.require(item['ref_id'] in actual_episodes,'Episode absent from original historical proof')
        original=actual_episodes[item['ref_id']]
        expected_row={k:deepcopy(original[k]) for k in ('ref_id','known_at_sequence','known_at_entry_sha256','member_resolution','member_registrations')}
        expected_row.update(episode_sha256=original['records']['episode']['episode_sha256'],generation=original['records']['episode']['generation'])
        C.require(item==expected_row,'Episode historical registration/known-at originals differ')
        seen.add(item['ref_id'])
    expected_ids={r['ref_id'] for r in registry['records'] if r['kind']=='episode'} if registry else set()
    C.require(seen==expected_ids,'complete Episode registry snapshot required')


def _resolution(value,episode,snapshot_row):
    C.exact(value,set(MEMBERS),'Episode member resolution')
    C.require(value==snapshot_row.get('member_resolution'),'Episode member resolution snapshot differs')
    for name,field in MEMBERS.items():
        rows=value[name];C.require(type(rows) is list and [r.get('ref_id') for r in rows]==episode[field],'Episode member IDs differ')
        for row in rows:
            C.exact(row,{'ref_id','status','provenance_sha256'},'member resolution')
            C.require(row['status'] in ('resolved','not_supplied'),'member status differs')
            if row['status']=='not_supplied':C.require(row['provenance_sha256'] is None,'missing member gains provenance')
            else:
                C.digest(row['provenance_sha256'])
                matches=[p for p in snapshot_row['member_registrations'][name] if p['ref_id']==row['ref_id']]
                C.require(len(matches)==1 and matches[0]['record_sha256']==row['provenance_sha256'] and
                    matches[0]['known_at_sequence']<snapshot_row['known_at_sequence'],'member historical registration differs')
            if name=='candidates':C.require(row['status']=='not_supplied','historical candidate family unsupported')


def _episode_meaning(row,candidate,snapshot,context,candidates,policy):
    C.exact(row,M._BODY_KEYS|{'original_parts','member_resolution'},'Episode meaning')
    registry=candidate['provenance'].get('record')
    C.require(candidate['origin']=='readonly_registry' and candidate['kind']=='episode' and registry['status']=='present' and
        row['availability']=='present','Episode prior role/lifecycle differs')
    body=M._body(row);C.exact(body,{'schema','episode','full_content','summary_content'},'Episode family body')
    C.require(body['schema']=='mfm-episode-snapshot-meaning-v1','Episode family schema differs')
    C.exact(row['original_parts'],set(PARTS),'Episode original parts')
    for name in PARTS:
        original=row['original_parts'][name];C.exact(original,{'artifact','base64'},'Episode original custody')
        C.require(type(original['base64']) is str,'original encoding differs')
        try:raw=b64decode(original['base64'],validate=True)
        except (ValueError,TypeError) as exc:raise C.CognitiveKernelContractError('malformed Episode original') from exc
        C.require(b64encode(raw).decode('ascii')==original['base64'] and E._original(raw,original['artifact'])==body[name],
            'Episode body original differs')
    episode=E._episode(body['episode'])
    C.require(episode.scope==context.base.scope and episode.envelope.authority_namespace_id==candidate['namespace'] and
        episode.episode_id==candidate['ref_id'] and registry['record_sha256']==episode.episode_sha256,
        'Episode registry/original identity differs')
    C.require(episode.envelope.deletion_state=='active' and episode.episode_state in policy['eligible_episode_states'],
        'inactive/ineligible Episode body')
    C.require(C.canonical_sha256(body['full_content'])==episode.full_content_digest and
        C.canonical_sha256(body['summary_content'])==episode.summary_content_digest,'Episode original part digest differs')
    known=next(r for r in snapshot['episodes'] if r['ref_id']==candidate['ref_id'])
    C.require(known['episode_sha256']==episode.episode_sha256 and known['generation']==episode.generation,'Episode known-at metadata differs')
    _resolution(row['member_resolution'],body['episode'],known)


def caller_meaning_v2_record(caller,*,caller_sha256,claim_snapshot,claim_snapshot_sha256,episode_snapshot,episode_snapshot_sha256,
    candidate_meanings,view_policy,view_policy_sha256):
    C.require(type(caller) is dict and caller.get('record_sha256')==caller_sha256,'external caller differs')
    context,opened,registry=F.caller_input_from_record(caller)
    M._snapshot(claim_snapshot,claim_snapshot_sha256,context,registry)
    _snapshot(episode_snapshot,episode_snapshot_sha256,context,registry,claim_snapshot)
    C.pin(view_policy,view_policy_sha256,'mfm-caller-meaning-view-policy-v2')
    C.require(view_policy==normative_view_policy_v2(),'normative Episode meaning policy differs')
    candidates=F.selection_candidates(context,opened,registry)
    C.require(type(candidate_meanings) is list and len(candidate_meanings)==len(candidates),'complete meaning inventory required')
    for slot,(row,candidate) in enumerate(zip(candidate_meanings,candidates)):
        C.require(type(row) is dict and type(row.get('slot')) is int and row['slot']==slot,'meaning slot differs')
        if row.get('family')==EPISODE_FAMILY:_episode_meaning(row,candidate,episode_snapshot,context,candidates,view_policy)
        else:M._meaning(row,candidate,claim_snapshot,context)
    return C.seal({'schema':SCHEMA,'caller_sha256':caller_sha256,'claim_snapshot':claim_snapshot,'episode_snapshot':episode_snapshot,
        'candidate_inventory':candidates,'candidate_meanings':candidate_meanings,'view_policy':view_policy})


def caller_meaning_v2_from_record(record,caller,*,meaning_sha256,caller_sha256):
    C.pin(record,meaning_sha256,SCHEMA)
    C.exact(record,{'schema','caller_sha256','claim_snapshot','episode_snapshot','candidate_inventory','candidate_meanings','view_policy','record_sha256'},'meaning v2')
    actual=caller_meaning_v2_record(caller,caller_sha256=caller_sha256,claim_snapshot=record['claim_snapshot'],
        claim_snapshot_sha256=record['claim_snapshot']['record_sha256'],episode_snapshot=record['episode_snapshot'],
        episode_snapshot_sha256=record['episode_snapshot']['record_sha256'],candidate_meanings=record['candidate_meanings'],
        view_policy=record['view_policy'],view_policy_sha256=record['view_policy']['record_sha256'])
    C.require(record==actual,'complete Episode companion differs');return actual


def semantic_view_v2(record,caller,*,meaning_sha256,caller_sha256):
    verified=caller_meaning_v2_from_record(record,caller,meaning_sha256=meaning_sha256,caller_sha256=caller_sha256)
    result=[]
    for candidate,row in zip(verified['candidate_inventory'],verified['candidate_meanings']):
        family=row['family']
        provenance='validated_episode_originals' if family==EPISODE_FAMILY else 'validated_claim_snapshot' if family=='claim-snapshot-v1' else 'registered_description' if family=='registered-description-v1' else 'opened_source_metadata' if family=='opened-source-v1' else 'source_subject_metadata' if family=='source-subject-v1' else 'metadata_only'
        item={'slot':row['slot'],'candidate':deepcopy(candidate),'availability':row['availability'],'family':family,'provenance_class':provenance,
            'content':M._body(row),'body_file_sha256':row['body_sha256'],'independent_observation':False}
        if family==EPISODE_FAMILY:item.update(original_parts=deepcopy(row['original_parts']),member_resolution=deepcopy(row['member_resolution']))
        result.append(item)
    return result
