"""Real reversible RawBuffer input custody; not a canonical or authenticated writer."""
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import stat
from cognitive_kernel import payload_store,raw_buffer
from cognitive_kernel.canonical import canonical_json_bytes,canonical_sha256
from . import formation_world_contracts as C,formation_task_content_input as T,formation_uncertainty_input as U

FAMILIES=('mission','result','attempt','outcome','workspace','uncertainty')


def _bindings():return {'task':T.task_content_binding(),'uncertainty':U.uncertainty_input_binding()}


def _metadata_ref(family,record,raw):
    return {'artifact_id':'input-content-metadata-'+canonical_sha256({'family':family,'entity_id':record['entity_id'],'content_version':record['content_version']}),
        'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),'media_type':'application/json'}


def _original(family,raw,parts,scope,namespace):
    C.require(family in FAMILIES,'unsupported content family');record=C.parse_json(raw)
    C.require(record['schema']==T.SCHEMAS[family],'family/record mismatch')
    rows=T.owned_artifacts(record);refs={'content-metadata':_metadata_ref(family,record,raw),**{r['part_name']:r['artifact'] for r in rows}}
    binding=U.uncertainty_input_binding() if family=='uncertainty' else T.task_content_binding()
    validator=U.uncertainty_original if family=='uncertainty' else T.task_originals
    return validator(raw,parts,refs=refs,scope=scope,namespace=namespace,contract_binding=binding,contract_binding_sha256=binding['record_sha256'])


def _ordinary(path):
    for item in (path,*path.parents):
        if not os.path.lexists(item):continue
        info=item.lstat()
        C.require(not stat.S_ISLNK(info.st_mode) and not (getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0)),
            'linked input custody path refused')
        C.require(stat.S_ISREG(info.st_mode) if item==path and item.is_file() else stat.S_ISDIR(info.st_mode),'nonordinary input custody path')


def _write_new(path,raw):
    _ordinary(path.parent)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def _commit_capture_originals(commit,profile,scope,namespace):
    """Validate the complete custody envelope before any original payload lookup."""
    keys={'schema','store_id','profile_sha256','scope','namespace','family','entity_id','content_version','content_store_sequence','previous_store_entry_sha256',
        'operation_id','request_sha256','expected_previous_content_sha256','record_sha256','original_artifacts','raw_buffer_receipts','committed_at','commit_sha256'}
    C.exact(commit,keys,'input commit');C.pin(commit,commit['commit_sha256'],'mfm-input-content-commit-receipt-v1','commit_sha256')
    C.require(commit['store_id']==profile['store_id'] and commit['profile_sha256']==profile['profile_sha256'] and
        commit['scope']==profile['scope']==scope.metadata_record() and commit['namespace']==profile['namespace']==namespace,'foreign input commit/profile')
    C.require(commit['family'] in FAMILIES,'unsupported input commit family');C.identifier(commit['entity_id']);C.identifier(commit['operation_id'])
    C.integer(commit['content_version'],1);C.integer(commit['content_store_sequence'],1);C.timestamp(commit['committed_at'])
    C.digest(commit['record_sha256']);C.digest(commit['request_sha256'])
    if commit['content_store_sequence']==1:C.require(commit['previous_store_entry_sha256'] is None,'initial store predecessor differs')
    else:C.digest(commit['previous_store_entry_sha256'])
    if commit['content_version']==1:C.require(commit['expected_previous_content_sha256'] is None,'initial content predecessor differs')
    else:C.digest(commit['expected_previous_content_sha256'])
    artifacts=commit['original_artifacts'];captures=commit['raw_buffer_receipts']
    C.require(type(artifacts) is list and type(captures) is list and bool(artifacts) and len(captures)==len(artifacts),'complete actual capture roster differs')
    names=set();references=[]
    for row,owned in zip(captures,artifacts):
        C.exact(owned,{'part_name','artifact'},'captured original artifact');C.identifier(owned['part_name']);T.artifact(owned['artifact'])
        C.integer(owned['artifact']['bytes'],1);C.require(owned['part_name'] not in names,'duplicate captured original part');names.add(owned['part_name'])
        C.exact(row,{'part_name','reference','capture_receipt'},'actual raw capture')
        C.require(type(row['reference']) is dict,'raw reference is not an original object');C.integer(row['reference'].get('byte_length'),1)
        reference=raw_buffer.RawBufferReference.from_record(row['reference'])
        C.require(canonical_json_bytes(reference.record())==canonical_json_bytes(row['reference']),'complete raw reference normalizes or has undeclared fields')
        C.exact(row['capture_receipt'],{'reference','physical_object_created','deduplicated','receipt_sha256'},'capture receipt')
        receipt=raw_buffer.RawBufferCaptureReceipt.create(reference=reference,physical_object_created=row['capture_receipt']['physical_object_created'],deduplicated=row['capture_receipt']['deduplicated'])
        C.require(canonical_json_bytes(receipt.record())==canonical_json_bytes(row['capture_receipt']),'complete capture receipt original differs')
        ref=owned['artifact'];logical='input-content-'+canonical_sha256({'family':commit['family'],'entity_id':commit['entity_id'],'content_version':commit['content_version'],'part_name':owned['part_name']})
        C.require(row['part_name']==owned['part_name'] and reference.scope==scope and reference.logical_record_id==logical and
            reference.byte_length==ref['bytes'] and reference.content_digest==ref['file_sha256'] and reference.media_type==ref['media_type'] and
            reference.captured_at==commit['committed_at'],'raw capture identity/custody differs')
        references.append(reference)
    C.require(artifacts[0]['part_name']=='content-metadata','original metadata capture must be first')
    request={k:commit[k] for k in ('family','entity_id','content_version','record_sha256','original_artifacts','expected_previous_content_sha256','committed_at')}
    C.require(commit['request_sha256']==canonical_sha256(request),'accepted request digest differs')
    return references


class InputContentPrototypeStore:
    def __init__(self,root,*,scope,namespace,store_key,created_at,family_bindings):
        scope.validate();C.identifier(namespace);C.identifier(store_key);C.timestamp(created_at)
        C.require(family_bindings==_bindings(),'normative family bindings differ')
        root=Path(root).absolute();_ordinary(root)
        self.root=payload_store.validate_raw_buffer_root(root);self.scope=scope;self.namespace=namespace
        self._profile=C.seal({'schema':'mfm-input-content-prototype-profile-v1','scope':scope.metadata_record(),'namespace':namespace,
            'store_id':'input-content-prototype-'+canonical_sha256({'scope':scope.metadata_record(),'namespace':namespace,'store_key':store_key,'created_at':created_at})[:32],
            'families':list(FAMILIES),'dependency_binding_sha256':canonical_sha256(family_bindings),'prototype_state':'reversible_nonproduction'},'profile_sha256')
        self.root.mkdir(parents=True,exist_ok=True);profile_path=self.root/'profile.json';_ordinary(profile_path)
        if profile_path.exists():C.require(C.parse_json(profile_path.read_bytes())==self._profile,'existing input profile differs')
        else:_write_new(profile_path,canonical_json_bytes(self._profile))
        self.commits_root=self.root/'commits';self.intents_root=self.root/'intents'
        for path in (self.commits_root,self.intents_root):_ordinary(path);path.mkdir(exist_ok=True)
        self.raw=payload_store.open_raw_buffer_store(self.root/'raw-buffer',scope=scope,created_at=created_at)
        try:self._load()
        except BaseException:self.raw.close();raise

    def profile_record(self):return deepcopy(self._profile)
    def close(self):self.raw.close()

    def _load(self):
        self._commits=[];self._operations={};self._versions={};self._current={};previous=None
        paths=sorted(self.commits_root.iterdir())
        for i,path in enumerate(paths,1):
            _ordinary(path);C.require(path.name==f'{i:08d}.json','content commit sequence differs')
            row=C.parse_json(path.read_bytes());C.pin(row,row['commit_sha256'],'mfm-input-content-commit-receipt-v1','commit_sha256')
            C.require(row['content_store_sequence']==i and row['previous_store_entry_sha256']==previous,'content commit chain differs')
            _commit_capture_originals(row,self._profile,self.scope,self.namespace)
            raw=self.raw.load_opaque_payload(row['raw_buffer_receipts'][0]['reference']['reference_id'])
            refs={'content':row['original_artifacts'][0]['artifact'],
                'commit':{'artifact_id':'stored-commit','bytes':path.stat().st_size,'file_sha256':sha256(path.read_bytes()).hexdigest(),'media_type':'application/json'},
                'profile':{'artifact_id':'stored-profile','bytes':len(canonical_json_bytes(self._profile)),'file_sha256':sha256(canonical_json_bytes(self._profile)).hexdigest(),'media_type':'application/json'}}
            binding=U.uncertainty_input_binding() if row['family']=='uncertainty' else T.task_content_binding()
            input_commit_original(raw,canonical_json_bytes(row),canonical_json_bytes(self._profile),refs=refs,family_binding=binding,
                family_binding_sha256=binding['record_sha256'],scope=self.scope,namespace=self.namespace)
            parts={r['part_name']:self.raw.load_opaque_payload(r['reference']['reference_id']) for r in row['raw_buffer_receipts'][1:]}
            original=_original(row['family'],raw,parts,self.scope,self.namespace);key=(row['family'],row['entity_id']);version=key+(row['content_version'],)
            C.require(version not in self._versions and row['operation_id'] not in self._operations,'duplicate input commit')
            if key in self._current:
                old=self._versions[key+(self._current[key]['content_version'],)]['original']
                (U.uncertainty_successor if row['family']=='uncertainty' else T.task_successor)(old,original)
                C.require(row['expected_previous_content_sha256']==old['record']['record_sha256'],'stored predecessor differs')
            else:C.require(row['content_version']==1 and row['expected_previous_content_sha256'] is None,'stored initial content differs')
            self._commits.append(row);self._operations[row['operation_id']]=row;self._versions[version]={'receipt':row,'original':original};self._current[key]=row;previous=row['commit_sha256']

    def inspect_commits(self):return tuple(deepcopy(self._commits))

    def commit(self,family,metadata_bytes,parts,*,operation_id,expected_previous_content_sha256,committed_at):
        C.identifier(operation_id);C.timestamp(committed_at)
        original=_original(family,metadata_bytes,parts,self.scope,self.namespace);record=original['record']
        artifacts=[{'part_name':'content-metadata','artifact':original['artifacts']['content-metadata']},*T.owned_artifacts(record)]
        request={'family':family,'entity_id':record['entity_id'],'content_version':record['content_version'],'record_sha256':record['record_sha256'],
            'original_artifacts':artifacts,'expected_previous_content_sha256':expected_previous_content_sha256,'committed_at':committed_at}
        request_sha=canonical_sha256(request)
        lock=self.root/'content.lock';_ordinary(lock);handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:
            self._load()
            if operation_id in self._operations:
                saved=self._operations[operation_id];C.require(saved['request_sha256']==request_sha,'changed operation replay');return deepcopy(saved)
            intent=self.intents_root/(canonical_sha256({'operation_id':operation_id})+'.json')
            C.require(not os.path.lexists(intent),'previous attempted operation remains uncommitted; no automatic retry')
            key=(family,record['entity_id']);old=self._current.get(key)
            if old is None:C.require(record['content_version']==1 and expected_previous_content_sha256 is None,'initial content predecessor differs')
            else:
                C.require(expected_previous_content_sha256==old['record_sha256'],'stale input predecessor')
                previous=self._versions[key+(old['content_version'],)]['original']
                (U.uncertainty_successor if family=='uncertainty' else T.task_successor)(previous,original)
            # Every original and zero-byte refusal has been checked before this first effect.
            _write_new(intent,canonical_json_bytes({'operation_id':operation_id,'request_sha256':request_sha,'profile_sha256':self._profile['profile_sha256']}))
            captures=[]
            for row in artifacts:
                name=row['part_name'];raw=metadata_bytes if name=='content-metadata' else parts[name]
                logical='input-content-'+canonical_sha256({'family':family,'entity_id':record['entity_id'],'content_version':record['content_version'],'part_name':name})
                receipt=self.raw.capture(raw,logical_record_id=logical,media_type=row['artifact']['media_type'],sensitivity_class='internal',
                    retention_class=record['retention_class'],captured_at=committed_at)
                reference=self.raw.get_reference(receipt.reference.reference_id)
                C.require(reference==receipt.reference,'actual raw capture lookup differs')
                captures.append({'part_name':name,'reference':reference.record(),'capture_receipt':receipt.record()})
            value=C.seal({'schema':'mfm-input-content-commit-receipt-v1','store_id':self._profile['store_id'],'profile_sha256':self._profile['profile_sha256'],
                'scope':self.scope.metadata_record(),'namespace':self.namespace,**request,'content_store_sequence':len(self._commits)+1,
                'previous_store_entry_sha256':None if not self._commits else self._commits[-1]['commit_sha256'],
                'operation_id':operation_id,'request_sha256':request_sha,'raw_buffer_receipts':captures},'commit_sha256')
            _write_new(self.commits_root/f'{value["content_store_sequence"]:08d}.json',canonical_json_bytes(value));self._load();return deepcopy(value)
        finally:os.close(handle);lock.unlink()

    def read_original_part(self,family,entity_id,content_version,part_name):
        C.require(family in FAMILIES,'unknown read family');C.identifier(entity_id);C.integer(content_version,1);C.identifier(part_name)
        key=(family,entity_id,content_version);C.require(key in self._versions,'exact family/version original absent')
        receipt=self._versions[key]['receipt'];row=next((r for r in receipt['raw_buffer_receipts'] if r['part_name']==part_name),None)
        C.require(row is not None,'exact original part absent');return self.raw.load_opaque_payload(row['reference']['reference_id'])


def input_commit_original(metadata_bytes,commit_bytes,profile_bytes,*,refs,family_binding,family_binding_sha256,scope,namespace):
    C.exact(refs,{'content','commit','profile'},'commit original refs')
    record=C.parse_json(T._bytes(metadata_bytes,refs['content']));profile=C.parse_json(T._bytes(profile_bytes,refs['profile']));commit=C.parse_json(T._bytes(commit_bytes,refs['commit']))
    family=T.common(record,scope,namespace)
    expected_binding=U.uncertainty_input_binding() if family=='uncertainty' else T.task_content_binding()
    C.pin(family_binding,family_binding_sha256,expected_binding['schema']);C.require(family_binding==expected_binding,'commit normative family binding differs')
    C.exact(profile,{'schema','scope','namespace','store_id','families','dependency_binding_sha256','prototype_state','profile_sha256'},'input profile')
    C.pin(profile,profile['profile_sha256'],'mfm-input-content-prototype-profile-v1','profile_sha256');C.identifier(profile['store_id'])
    C.require(profile['scope']==scope.metadata_record() and profile['namespace']==namespace and profile['families']==list(FAMILIES) and
        profile['dependency_binding_sha256']==canonical_sha256(_bindings()) and profile['prototype_state']=='reversible_nonproduction','input profile differs')
    references=_commit_capture_originals(commit,profile,scope,namespace)
    for key in ('entity_id','content_version','record_sha256'):C.require(canonical_json_bytes(commit[key])==canonical_json_bytes(record[key]),'input commit content identity differs')
    C.require(commit['family']==family and commit['expected_previous_content_sha256']==record['previous_content_sha256'],'input commit family/predecessor differs')
    artifacts=[{'part_name':'content-metadata','artifact':deepcopy(refs['content'])},*T.owned_artifacts(record)]
    C.require(canonical_json_bytes(commit['original_artifacts'])==canonical_json_bytes(artifacts),'complete actual capture roster differs')
    for reference in references:C.require(reference.retention_class==record['retention_class'],'raw capture retention differs')
    return {'record':record,'commit':commit,'profile':profile,'artifacts':deepcopy(refs),'body_bytes_verified':False,'authenticated_service_claimed':False}
