"""Frozen approval, task, request and evidence contract. Python standard library only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time

from frozen_prompt import SYSTEM, output_schema, prompt

BASE = Path(__file__).resolve().parent
VERSION = '1.0.3'
MAIN = '0abaed85873c3f8de04765847eb7700b0e20433f'
TAG = 'qwen3.8:27b-q4_K_M'
DRAFT_SHA = 'e16142639f13c9b2191afcfa4221dbf1610abf6af84273b07774fc13a0a1b40b'
APPROVAL_SHA = '8e7a384a745496f4c4f68996733ddf5220783bc871d852f99577ca0a7287373f'
APPROVED_SHA = '28102a66b0408d7a63abb36dbf86571172318215834f807076952715bc70ff55'
TASK_SHA = 'b3abc2e36e4e93d3f8f6c9398c7c39b6ca2f40f9b3dcbe875ce9005d096f9420'
CALIBRATION_ID = 'alice-qwen38-a1-' + APPROVAL_SHA[:16]
RUN_ID = 'alice-qwen38-a2-c926d9e355dd'
ROOT = Path('/homes/01/mxrayan/rayan-compute')
REMOTE_RUN = ROOT / 'runs' / RUN_ID
BOOL_FIELDS = ('critical_veto', 'actor_role_direction_correct', 'meaningful_bridge_present', 'contradiction', 'arbitrary_unbridged')
GOLD_FIELDS = ('verdict',) + BOOL_FIELDS
HARD = ('Q02_CORE_CONTRADICTION', 'Q04_FAKE_SOURCE_HISTORY', 'Q05_FAKE_LIVED_MEMORY', 'Q11_PERSONALITY_FLATTENING', 'Q13_CONTROLLING_PROTECTION')
NO_AUTHORITY = {k: False for k in ('private_pointwise_authorized', 'pointwise_ready', 'breadth_v104_eligible', 'A_SYN_acceptance', 'A_SYN_promotion', 'model_training', 'hidden_MC8_used', 'private_candidates_used', 'independent_certification', 'four_family_binding_created')}


class Stop(RuntimeError):
    """A deterministic stop; never an instruction to resample or substitute."""


def require(ok, message):
    if not ok:
        raise Stop(message)


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def strict(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def invalid(value):
        raise Stop('nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return strict(Path(path).read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write(path, obj):
    atomic_bytes(path, canonical(obj).encode('utf-8'))


def immutable(path, obj):
    path = Path(path)
    data = canonical(obj).encode('utf-8')
    if path.exists():
        require(path.read_bytes() == data, 'immutable evidence differs: ' + path.name)
    else:
        atomic_bytes(path, data)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def safe_relative(name):
    require(isinstance(name, str) and '\\' not in name, 'invalid relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.parts and all(x not in ('.', '..') for x in p.parts), 'unsafe path')
    require(':' not in name and str(p) == name, 'noncanonical path')
    return p


def verify_package(base=BASE):
    manifest = read(base / 'PACKAGE_MANIFEST.json')
    listed = set()
    for item in manifest['files']:
        name = item['path']
        safe_relative(name)
        require(name not in listed, 'duplicate manifest entry')
        listed.add(name)
        p = base / name
        require(p.is_file() and not p.is_symlink(), 'package file absent or linked')
        require(p.stat().st_size == item['bytes'] and file_sha(p) == item['sha256'], 'package hash drift: ' + name)
    observed = {str(p.relative_to(base)).replace('\\', '/') for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(observed == listed | {'PACKAGE_MANIFEST.json'}, 'unmanifested package files')
    return file_sha(base / 'PACKAGE_MANIFEST.json')


def authority(base=BASE):
    a = base / 'authority'
    for name, expected in [('draft.json', DRAFT_SHA), ('approval.json', APPROVAL_SHA), ('approved.json', APPROVED_SHA), ('tasks.json', TASK_SHA)]:
        require(file_sha(a / name) == expected, 'authority hash drift: ' + name)
    draft, approval, approved = (read(a / name) for name in ('draft.json', 'approval.json', 'approved.json'))
    require(approval['decision'] == 'APPROVED' and approval['owner_message_exact'] == 'approve', 'owner approval absent')
    require(approval['approved_draft_sha256'] == DRAFT_SHA and approved['owner_approval_sha256'] == APPROVAL_SHA, 'approval lineage')
    require(approved['owner_ratified'] is True and approved['approved_draft_sha256'] == DRAFT_SHA, 'amendment not ratified')
    changed = {'artifact_id', 'status', 'owner_ratified', 'authority_reason', 'approved_draft_sha256', 'owner_approval_sha256'}
    require(canonical({k:v for k,v in draft.items() if k not in changed}) == canonical({k:v for k,v in approved.items() if k not in changed}), 'approved draft scope changed')
    require(file_sha(a / 'source_binding_policy.json') == approved['source_binding_policy_sha256'], 'source binding drift')
    doc = read(a / 'tasks.json')
    require(doc['private_alice_or_elaina_data_used'] is False, 'nonpublic tasks')
    tasks = doc['tasks']
    require(len(tasks) == len({x['task_id'] for x in tasks}) == 16, 'frozen task identities')
    return approved, tasks, read(a / 'runtime_policy.json')


def request(task, index, approved):
    require(1 <= index <= 16, 'seed index')
    profile = approved['profile']
    options = {k: profile[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repeat_penalty', 'num_ctx', 'num_predict')}
    options.update(seed=9100 + index, num_gpu=0, num_thread=20)
    return {'model': TAG, 'messages': [{'role':'system', 'content':SYSTEM}, {'role':'user', 'content':prompt(task)}], 'stream':True, 'think':True, 'format':output_schema(), 'keep_alive':'30m', 'options':options}


def full_digest(value):
    require(isinstance(value, str), 'digest not a string')
    value = value.removeprefix('sha256:')
    require(re.fullmatch('[0-9a-f]{64}', value) is not None, 'full lowercase SHA-256 required')
    return value


def probe_request(approved, tasks):
    body = request(tasks[0], 1, approved)
    body.update(messages=[{'role':'user','content':'Produce a JSON object containing an array of all integers from 1 through 256 in ascending order.'}], format={'type':'object','required':['numbers'],'additionalProperties':False,'properties':{'numbers':{'type':'array','items':{'type':'integer'}}}})
    body['options']['num_predict'] = 128
    body['options']['seed'] = 9100
    return body


def response_value(content):
    require(isinstance(content, str) and content.strip(), 'empty structured response')
    value = strict(content)
    require(type(value) is dict and set(value) == set(output_schema()['required']), 'response key set')
    require(value['verdict'] in ('PASS', 'HOLD', 'REJECT'), 'invalid verdict')
    require(all(type(value[k]) is bool for k in BOOL_FIELDS), 'response boolean type')
    require(type(value['rationale']) is str and len(value['rationale']) <= 500, 'rationale type or limit')
    return value


def decode_stream(raw):
    """Preserve all returned chunks. Only one final, complete response is eligible."""
    chunks = [strict(line) for line in raw.splitlines() if line.strip()]
    require(chunks, 'empty HTTP stream')
    require(all(type(c) is dict and 'error' not in c for c in chunks), 'stream error')
    finals = [i for i,c in enumerate(chunks) if c.get('done') is True]
    require(finals == [len(chunks)-1], 'missing, duplicate, or nonterminal done event')
    require(all(c.get('model') == TAG for c in chunks), 'response model tag mismatch')
    content = ''.join((c.get('message') or {}).get('content', '') for c in chunks)
    thinking = ''.join((c.get('message') or {}).get('thinking', '') for c in chunks)
    final = dict(chunks[-1])
    final['message'] = {'role':'assistant', 'content':content, 'thinking':thinking}
    return final


def classify(raw):
    result = {'status':'INCOMPLETE', 'response':None, 'finish_reason':None, 'metrics':{}, 'error_code':None}
    try:
        final = decode_stream(raw)
        result['finish_reason'] = final.get('done_reason')
        keys = ('total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration')
        require(all(type(final.get(k)) is int and final[k] >= 0 for k in keys), 'missing/invalid timing or token counters')
        result['metrics'] = {k:final[k] for k in keys}
        require(final['eval_count'] <= 6144, 'approved output token ceiling exceeded')
        if final.get('done_reason') != 'stop':
            result['error_code'] = 'NON_STOP_FINISH'
            return result
        result['status'] = 'FORMAT_FAILED'
        result['response'] = response_value(final['message']['content'])
        result['status'] = 'COMPLETE'
    except (Stop, ValueError, TypeError, KeyError):
        result['error_code'] = 'INVALID_STREAM_OR_RESPONSE' if result['status'] == 'INCOMPLETE' else 'INVALID_STRUCTURED_RESPONSE'
    return result


def score(records, tasks):
    ids = [r['task_id'] for r in records]
    require(len(ids) == len(set(ids)) and set(ids) <= {t['task_id'] for t in tasks}, 'duplicate or unknown task evidence')
    by = {r['task_id']:r for r in records}
    complete = {k:r['response'] for k,r in by.items() if r['status'] == 'COMPLETE'}
    for r in complete.values():
        response_value(canonical(r))
    gold = {t['task_id']:t['gold'] for t in tasks}
    critical = [k for k,g in gold.items() if g['critical_veto']]
    decision = {k:k in complete and all(complete[k][f] == gold[k][f] for f in ('verdict','critical_veto')) for k in gold}
    verdict = sum(k in complete and complete[k]['verdict'] == gold[k]['verdict'] for k in gold)
    gates = {'sixteen_unique_complete_tasks': len(complete)==16 and set(complete)==set(gold), 'at_least_14_verdict_matches':verdict>=14, 'at_least_6_critical_decisions':sum(decision[k] for k in critical)>=6, 'five_hard_anchors':all(decision[k] for k in HARD), 'compatible_novelty_pass':complete.get('Q01_COMPATIBLE_NOVELTY',{}).get('verdict')=='PASS', 'arbitrary_hobby_hold':complete.get('Q03_ARBITRARY_HOBBY',{}).get('verdict')=='HOLD'}
    matrix = [{'task_id':t['task_id'], 'status':by.get(t['task_id'],{}).get('status','NOT_ATTEMPTED'), 'decision_match':decision[t['task_id']], 'expected':gold[t['task_id']], 'observed':{f:complete[t['task_id']][f] for f in GOLD_FIELDS} if t['task_id'] in complete else None} for t in tasks]
    return {'tasks_complete':len(complete), 'tasks_required':16, 'verdict_matches':verdict, 'critical_decision_matches':sum(decision[k] for k in critical), 'critical_tasks':len(critical), 'hard_anchors_correct':sum(decision[k] for k in HARD), 'full_gold_secondary_matches':sum(all(complete[k][f]==gold[k][f] for f in GOLD_FIELDS) for k in complete), 'gates':gates, 'qualification_passed':all(gates.values()), 'matrix':matrix}


def projection(metrics, tasks_left, remaining):
    count, duration = metrics.get('eval_count'), metrics.get('eval_duration')
    require(type(count) is int and count > 0 and type(duration) is int and duration > 0, 'throughput metrics unavailable')
    rate = count / (duration / 1e9)
    require(math.isfinite(rate) and rate > 0, 'invalid throughput')
    projected = 1.20 * tasks_left * (6144 / rate + metrics.get('prompt_eval_duration',0) / 1e9) + 300
    return {'tokens_per_second':rate, 'remaining_tasks':tasks_left, 'assumes_max_tokens_each':6144, 'margin':1.20, 'cleanup_reserve_seconds':300, 'projected_seconds':projected, 'remaining_job_seconds':remaining, 'fits':projected <= remaining}
