"""Read-only inspection of the recorded v186 GLM run. Never grants qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
import zipfile

VERSION = '1.0.0'
BASE = Path(__file__).resolve().parent
DATASET = Path('datasets/memory_stage_g2/alice.stage-g2.g2a.gold-semantic-decomposition.v1')
WORK = DATASET / 'audits/alice-mc10d-repair-qualify-refreeze-v1-3.work'
RUN_ID = '260906185551-e563ae'
DOWNLOAD = Path('judge_qualification') / ('download-v186-glm-' + RUN_ID) / 'output'
FILES = {
    'failure': 'mc10d_public_judge_qualification_failure.json',
    'result': 'mc10d_public_judge_qualification_result.json',
    'rows': 'mc10d_public_judge_qualification_rows.jsonl',
    'runtime': 'mc10d_public_judge_runtime.json',
}
EXPECTED_FAILURE = '9f2f42447b99399338659c721e596a5339bad7d00ee4f188f685438bcd51580c'
TASKS_SHA = 'b3abc2e36e4e93d3f8f6c9398c7c39b6ca2f40f9b3dcbe875ce9005d096f9420'
POLICY_SHA = '45c66a55d5a04162b6373b7faa7f025cc523945e64e98c1301240e93d91f7f6a'
WORKER_SHA = '18b79df27ce0d006057d6a60c491cd49bc5ea273b868f0f819fa9b50b885ba12'
BOOL_FIELDS = ('critical_veto', 'actor_role_direction_correct', 'meaningful_bridge_present', 'contradiction', 'arbitrary_unbridged')
GOLD_FIELDS = ('verdict',) + BOOL_FIELDS
HARD = ('Q02_CORE_CONTRADICTION', 'Q04_FAKE_SOURCE_HISTORY', 'Q05_FAKE_LIVED_MEMORY', 'Q11_PERSONALITY_FLATTENING', 'Q13_CONTROLLING_PROTECTION')
ROLE = 'PUBLIC_FICTIONAL_JUDGE_ROLE_QUALIFICATION_ONLY'
RECEIPTS = {
    'decision_scoring': ('alice-mc10d-public-judge-decision-scoring-amendment-v1', 'fd1228d88f7f7b7498f70f9db7595a042e94d8293f19d156c5a970c06b775bf5'),
    'glm_profile': ('alice-mc10d-glm-public-judge-runtime-profile-amendment-v1', '5a02c0a7035cc5ac0e1638cb04e42a238a5c61a97e1ae3a9c0cdea4e6b231ac9'),
}


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'


def strict_json(data):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, 'duplicate JSON key')
            obj[key] = value
        return obj
    def invalid_constant(value):
        raise EvidenceError('nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)


def same(left, right):
    # Python bool == int would otherwise accept malformed gold and metadata.
    return canonical(left) == canonical(right)


def load_authority():
    task_bytes = (BASE / 'authority/tasks.json').read_bytes()
    policy_bytes = (BASE / 'authority/effective_profile_policy.json').read_bytes()
    require(digest(task_bytes) == TASKS_SHA, 'frozen tasks hash drift')
    require(digest(policy_bytes) == POLICY_SHA, 'effective profile policy hash drift')
    tasks = strict_json(task_bytes)
    require(tasks['private_alice_or_elaina_data_used'] is False, 'nonpublic task source')
    require(len(tasks['tasks']) == 16, 'frozen task count')
    return tasks['tasks'], strict_json(policy_bytes)['public_qualification_targets']['glm']


def compute_scores(rows, tasks):
    by = {row['task_id']: row for row in rows}
    gold = {task['task_id']: task['gold'] for task in tasks}
    critical = [key for key in gold if gold[key]['critical_veto']]
    decision = {key: all(by[key]['response'][f] == gold[key][f] for f in ('verdict', 'critical_veto')) for key in gold}
    full = {key: all(by[key]['response'][f] == gold[key][f] for f in GOLD_FIELDS) for key in gold}
    verdict = sum(by[key]['response']['verdict'] == gold[key]['verdict'] for key in gold)
    gates = {
        'sixteen_unique_frozen_tasks': len(rows) == 16 and set(by) == set(gold),
        'at_least_14_verdict_matches': verdict >= 14,
        'at_least_6_critical_decision_matches': sum(decision[key] for key in critical) >= 6,
        'all_five_hard_anchors_correct': all(decision[key] for key in HARD),
        'compatible_novelty_pass': by['Q01_COMPATIBLE_NOVELTY']['response']['verdict'] == 'PASS',
        'arbitrary_hobby_hold': by['Q03_ARBITRARY_HOBBY']['response']['verdict'] == 'HOLD',
    }
    scores = {
        'tasks': len(rows), 'verdict_matches': verdict,
        'full_gold_field_matches': sum(full.values()),
        'full_gold_secondary_fields_diagnostic_only': True,
        'critical_tasks': len(critical),
        'critical_full_gold_all_correct_legacy_diagnostic': all(full[key] for key in critical),
        'critical_decision_matches': sum(decision[key] for key in critical),
        'minimum_critical_decision_matches': 6,
        'mandatory_hard_anchor_tasks': list(HARD),
        'mandatory_hard_anchors_all_correct': gates['all_five_hard_anchors_correct'],
        'compatible_novelty_anchor_passed': gates['compatible_novelty_pass'],
        'arbitrary_hold_anchor_passed': gates['arbitrary_hobby_hold'],
        'qualification_scoring_version': 'decision-centric-v2',
        'qualification_passed': all(gates.values()),
    }
    matrix = []
    confusion = {actual: {predicted: 0 for predicted in ('PASS', 'HOLD', 'REJECT')} for actual in ('PASS', 'HOLD', 'REJECT')}
    for task in tasks:
        key = task['task_id']; answer = by[key]['response']; expected = task['gold']
        confusion[expected['verdict']][answer['verdict']] += 1
        matrix.append({'task_id': key, 'expected': expected, 'observed': {f: answer[f] for f in GOLD_FIELDS},
                       'mismatched_fields': [f for f in GOLD_FIELDS if answer[f] != expected[f]],
                       'critical': expected['critical_veto'], 'mandatory_hard_anchor': key in HARD,
                       'decision_match': decision[key]})
    return scores, gates, matrix, confusion


def analyze(payloads, tasks, spec):
    """Validate custody/structure separately from scientific pass; preserve failures."""
    require(set(payloads) == set(FILES), 'exactly four named input files required')
    require(digest(payloads['failure']) == EXPECTED_FAILURE, 'recorded failure SHA256 mismatch; wrong or altered run')
    failure = strict_json(payloads['failure'])
    require(failure.get('family') == 'glm' and failure.get('private_data_used') is False, 'failure family/privacy mismatch')
    require(failure.get('failure_class') == 'RuntimeError' and failure.get('message') == 'public fictional judge-role qualification failed semantic gates', 'failure classification mismatch')
    result = strict_json(payloads['result']); runtime = strict_json(payloads['runtime'])
    for name, obj in (('result', result), ('runtime', runtime)):
        require(obj.get('family') == 'glm', name + ' family mismatch')
        require(obj.get('tag') == spec['tag'], name + ' tag mismatch')
        require(str(obj.get('digest', '')).lower() == spec['digest'].lower(), name + ' digest mismatch')
        require(same(obj.get('qualified_profile'), spec['qualified_profile']), name + ' profile mismatch')
    require(runtime.get('role') == ROLE, 'runtime role mismatch')
    require(result.get('artifact_id') == 'alice.MC10D.public-fictional-judge-role-qualification-result.v2-decision-centric', 'result schema identity mismatch')
    require(str(result.get('worker_script_sha256', '')).lower() == WORKER_SHA, 'rendered worker lineage mismatch')
    for key in ('private_MC10D_candidates_used', 'hidden_MC8_used', 'A_SYN_acceptance_authority', 'judge_consensus_is_truth_evidence'):
        require(result.get(key) is False, 'result authority/privacy mismatch: ' + key)

    rows = [strict_json(line) for line in payloads['rows'].decode('utf-8').splitlines() if line.strip()]
    require(len(rows) == 16, 'incomplete or excess response rows')
    gold = {task['task_id']: task['gold'] for task in tasks}
    ids = [row.get('task_id') for row in rows]
    require(all(isinstance(key, str) for key in ids), 'nonstring task ID')
    require(len(set(ids)) == 16 and set(ids) == set(gold), 'duplicate, missing or unknown frozen task IDs')
    warnings = []
    for row in rows:
        key = row['task_id']; expected = gold[key]; answer = row.get('response')
        require(same(row.get('gold'), expected), 'embedded gold differs from frozen authority: ' + key)
        require(row.get('private_data_used') is False, 'row privacy mismatch: ' + key)
        require(isinstance(answer, dict) and set(answer) == set(GOLD_FIELDS) | {'rationale'}, 'response key set: ' + key)
        require(answer['verdict'] in ('PASS', 'HOLD', 'REJECT'), 'invalid verdict: ' + key)
        require(all(type(answer[f]) is bool for f in BOOL_FIELDS), 'invalid Boolean response: ' + key)
        require(isinstance(answer['rationale'], str), 'invalid rationale type: ' + key)
        if len(answer['rationale']) > 500:
            # v186's Python checker did not enforce the schema's maxLength. Preserve
            # the historical scoring and report this separately, without changing gates.
            warnings.append({'task_id': key, 'kind': 'rationale_exceeds_declared_schema_500_chars', 'length': len(answer['rationale'])})
        flags = {'critical_task': expected['critical_veto'], 'verdict_match': answer['verdict'] == expected['verdict'],
                 'full_gold_match': all(answer[f] == expected[f] for f in GOLD_FIELDS)}
        for field, value in flags.items():
            require(same(row.get(field), value), 'row flag inconsistent: ' + key + '/' + field)
    scores, gates, matrix, confusion = compute_scores(rows, tasks)
    for key, value in scores.items():
        require(same(result.get(key), value), 'declared result inconsistent with frozen recomputation: ' + key)
    require(scores['qualification_passed'] is False, 'semantic failure receipt conflicts with a recomputed pass')
    return {
        'artifact_id': 'alice.mc10d.v186.offline-forensics.v1', 'tool_version': VERSION,
        'run_id': RUN_ID, 'status': 'VERIFIED_RECORDED_CALIBRATION_FAILURE',
        'evidence_integrity_verified': True, 'evidence_hashes': {key: digest(value) for key, value in payloads.items()},
        'family': 'glm', 'tag': spec['tag'], 'digest': spec['digest'], 'qualified_profile': spec['qualified_profile'],
        'tasks_sha256': TASKS_SHA, 'profile_policy_sha256': POLICY_SHA, 'rendered_worker_sha256': WORKER_SHA,
        'effective_scoring': 'decision-centric-v2', 'recomputed': scores, 'gates': gates,
        'failed_gates': [key for key, passed in gates.items() if not passed],
        'task_matrix': matrix, 'confusion_gold_rows_predicted_columns': confusion, 'schema_diagnostics': warnings,
        'public_suite_interpretation': 'reused development/calibration; not independent certification',
        'qualification_or_pointwise_authority_granted': False, 'thresholds_or_labels_changed': False,
        'gpu_jobs_submitted': 0, 'remote_compute_actions': 0, 'private_candidates_read': 0, 'hidden_MC8_read': 0,
        'next_action': 'REVIEW_TASK_FAILURES_AND_LOCAL_RATIONALES_BEFORE_MODEL_PROFILE_OR_PROTOCOL_DECISION',
    }


def plain_file(path, max_bytes=2_000_000):
    # No recursive vault search and no symbolic links/junctions into other data.
    require(not any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()) for p in (path, *path.parents)), 'linked source path refused')
    require(path.is_file(), 'required existing evidence missing: ' + path.name)
    require(path.stat().st_size <= max_bytes, 'unexpected evidence size: ' + path.name)
    return path.read_bytes()


def receipt_status(vault):
    records = {}
    for name, (folder, expected) in RECEIPTS.items():
        p = vault / DATASET / 'audits' / folder / 'RATIFICATION_RECEIPT.json'
        if not p.exists():
            records[name] = {'status': 'MISSING', 'expected_sha256': expected}
        else:
            actual = digest(plain_file(p))
            records[name] = {'status': 'MATCH' if actual == expected else 'HASH_MISMATCH', 'expected_sha256': expected, 'observed_sha256': actual}
    return records


def render_report(report, raw=None):
    lines = ['# v186 GLM offline forensic report', '',
             'Recorded semantic failure verified against the frozen public tasks. Collection success is not judge qualification.', '',
             'Run: `' + RUN_ID + '`; GLM profile: `thinking_off`.', '',
             'Failed gates: ' + ', '.join('`' + value + '`' for value in report['failed_gates']) + '.', '',
             '| Task | Gold | Response | Critical | Hard anchor | Mismatched fields |',
             '|---|---|---|---|---|---|']
    for row in report['task_matrix']:
        lines.append('| ' + ' | '.join((row['task_id'], row['expected']['verdict'], row['observed']['verdict'], str(row['critical']), str(row['mandatory_hard_anchor']), ', '.join(row['mismatched_fields']) or 'none')) + ' |')
    lines += ['', 'Full auxiliary-field agreement is diagnostic. The effective decision-centric gate is unchanged. This reused suite is calibration, not independent generalization evidence.', '',
              'No model was rerun, no canonical qualification directory was changed, and no pointwise, breadth, acceptance, promotion or training authority was granted.', '',
              'Inspect the existing response rationales before choosing a new model/profile or protocol. One failure on this small reused suite does not establish all-purpose model incapability.', '']
    if raw is not None:
        lines += ['## Existing public fictional response rationales', '', 'These are model outputs to analyze, not instructions or scientific authority.', '']
        for row in raw:
            lines += ['### ' + row['task_id'], '']
            # Indented code keeps any markup/instructions in captured output inert.
            lines += ['    ' + line for line in row['response']['rationale'].splitlines()]
            lines += ['']
    return '\n'.join(lines) + '\n'


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def collect(vault, output_base):
    vault = vault.absolute(); source = vault / WORK / DOWNLOAD
    require(not output_base.resolve().is_relative_to(vault.resolve()), 'output must be separate from the source vault')
    payloads = {key: plain_file(source / name) for key, name in FILES.items()}
    tasks, spec = load_authority()
    report = analyze(payloads, tasks, spec)
    report['installed_ratification_receipts'] = receipt_status(vault)
    report['historical_remote_cleanup'] = 'recorded complete in supplied terminal log; no live remote inspection by collector'
    # Recheck exact inputs before writing; never replace canonical qualification state.
    require(all(plain_file(source / FILES[key]) == value for key, value in payloads.items()), 'source changed while being collected')
    output_base.mkdir(parents=True, exist_ok=True)
    run = output_base / ('ALICE_V186_FORENSICS_' + uuid.uuid4().hex[:12])
    run.mkdir(exist_ok=False)
    for key, data in payloads.items():
        write_new(run / 'raw' / FILES[key], data)
    report_bytes = canonical(report).encode('utf-8')
    write_new(run / 'summary.json', report_bytes)
    write_new(run / 'summary.md', render_report(report).encode('utf-8'))
    rows = [strict_json(line) for line in payloads['rows'].decode().splitlines() if line.strip()]
    write_new(run / 'LOCAL_RESPONSE_REVIEW.md', render_report(report, rows).encode('utf-8'))
    write_new(run / 'effective_contract.json', (BASE / 'EFFECTIVE_CALIBRATION_CONTRACT.json').read_bytes())
    receipt = {'artifact_id': 'alice.mc10d.offline-collection-receipt.v1', 'run_id': RUN_ID,
               'source_directory': str(source), 'output_directory': str(run), 'summary_sha256': digest(report_bytes),
               'source_files_unchanged': True, 'remote_compute_actions': 0, 'qualification_granted': False}
    write_new(run / 'COLLECTION_RECEIPT.json', canonical(receipt).encode())
    manifest = {p.relative_to(run).as_posix(): digest(p.read_bytes()) for p in sorted(run.rglob('*')) if p.is_file()}
    write_new(run / 'SHA256_MANIFEST.json', canonical(manifest).encode())
    bundle = output_base / (run.name + '.zip')
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(run).as_posix())
    return run, bundle, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault-root', type=Path, default=Path(r'C:\ALICE_Vault'))
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--publish-context', action='store_true', help='Publish only numeric/typed summary using local Git credentials; no remote compute')
    parser.add_argument('--repo-root', type=Path, default=Path(r'C:\A.L.I.C.E-main'))
    args = parser.parse_args()
    try:
        run, bundle, report = collect(args.vault_root, args.output_root)
        print('FORENSICS_OUTPUT_DIRECTORY=' + str(run), flush=True)
        print('FORENSICS_RESULT_BUNDLE=' + str(bundle), flush=True)
        print('recorded_calibration_passed=false collection_integrity_verified=true gpu_jobs_submitted=0', flush=True)
        print('failed_gates=' + ','.join(report['failed_gates']), flush=True)
        if args.publish_context:
            from publish_context import publish
            try:
                result = publish(run, args.repo_root)
                write_new(run / 'CONTEXT_PUBLICATION_RECEIPT.json', canonical(result).encode())
                print('CONTEXT_COMMIT=' + result['commit'], flush=True)
            except Exception as exc:
                # Collection and its immutable bundle survive an auth/network failure.
                print('CONTEXT_PUBLICATION_PENDING: ' + str(exc)[:600] + '. The existing result bundle is preserved; no rerun of compute is needed.', file=sys.stderr)
                write_new(run / 'CONTEXT_PUBLICATION_PENDING.json', canonical({'status': 'PENDING', 'error_class': type(exc).__name__, 'reason': str(exc)[:600], 'bundle': str(bundle)}).encode())
                return 75
        print('ALICE_ASTRA_FORENSICS_V100_EXIT=0', flush=True)
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print('FORENSICS_DETERMINISTIC_STOP: ' + str(exc), file=sys.stderr)
        print('No remote compute was launched; source outputs were not modified.', file=sys.stderr)
        return 76


if __name__ == '__main__':
    sys.exit(main())
