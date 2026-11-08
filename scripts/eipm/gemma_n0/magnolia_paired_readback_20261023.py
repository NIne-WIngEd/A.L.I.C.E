"""Frozen, complete saved-output readback; never trains or forwards a model."""
import gc
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
NAME = 'rayan-paired-readback-20261023'
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
RESULT_SHA = '5607f0b201a5ab931f0d1bed851920015fbd14948d1e715151f840cb72bfeccc'
DEPENDENCIES = {
    'pair': ('rayan-paired-update-live-custody-20261021.py', '7675d14a2c04efc6a4244b5c6a22db6aa8be2b109e6e4b7cfaf8f3317b6c1537'),
    'saved': ('rayan-saved-path-readback-20261020.py', 'e2e9ae77e28cd4116aefaadfce56bb9f4f6ab2a0eebc0c7f561cad51408eca7f'),
    'analysis': ('rayan-cumulative-results-20261018-analysis.py', '8e0982e758ad81d735c474101fdd2e271d4dee3e124d33fc7f1e129e0c50b6c7'),
    'live': ('rayan-paired-update-live-custody-20261021-live.py', '7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533'),
}
CONDITIONAL = ('concept_applicability', 'concept_uncertainty', 'concept_source_support', 'concept_source_disconfirmation')


def require(condition):
    if not condition:
        raise ValueError('saved readback binding or geometry differs')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def distribution(values):
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values))
    values = sorted(values)
    if not values:
        return {'count': 0, 'mean': None, 'min': None, 'median': None, 'max': None}
    n = len(values)
    return {'count': n, 'mean': math.fsum(values)/n, 'min': values[0],
            'median': (values[(n-1)//2]+values[n//2])/2, 'max': values[-1]}


def source_aliases(case):
    """Different current episodes share comparison roles, never evidence identity."""
    result = {}
    for entry in case.document.sources:
        record = entry.record
        current = record.source_family_id in case.lineage.context_family_ids
        key = 'current-role:'+record.source_kind if current else 'fixed:'+record.record_id
        require(key not in result.values())
        result[record.record_id] = key
    return result


def coordinate_fields(decision, aliases):
    """Canonical semantic coordinates; padded or ambiguous geometry is refused."""
    require(len(decision['candidate_ids']) == len(decision['concept_ids']) == len(decision['sources']) == 1)
    candidates, concepts = decision['candidate_ids'][0], decision['concept_ids'][0]
    sources = [aliases[r['record_id']] for r in decision['sources'][0]]
    require(all(len(xs) == len(set(xs)) and len(xs) > 0 for xs in (candidates, concepts, sources)))
    ci, ki, si = [sorted(range(len(xs)), key=xs.__getitem__) for xs in (candidates, concepts, sources)]
    latent, conditional = decision['native_latent'], decision['conditional_concepts']
    require(decision['candidate_mask'] == [[True]*len(candidates)] and latent['concept_mask'] == [[True]*len(concepts)])
    require(set(aliases) == {r['record_id'] for r in decision['sources'][0]})
    def vector(value, order):
        require(len(value) == 1 and len(value[0]) == len(order))
        return [value[0][i] for i in order]
    def relation(value):
        rows = vector(value, ki)
        require(all(len(row) == len(si) for row in rows))
        return [[row[i] for i in si] for row in rows]
    fields = {'n1:concepts': vector(latent['concepts'], ki), 'n2:candidates': vector(latent['candidates'], ci)}
    for name in CONDITIONAL:
        fields['n1:'+name] = relation(conditional[name]) if 'source_' in name else vector(conditional[name], ki)
    view_order = si + [len(si)+i for i in ki]  # Native views concatenate sources, then concepts.
    for name, value in latent['views'].items():
        fields['n1:view:'+name] = vector(value, view_order)
    for name in ('preference_logits', 'preferences', 'co_valid_probabilities', 'contraindications'):
        fields['output:'+name] = vector(decision[name], ci)
    # Common shifts in preference logits cannot change the ranking.
    logits = fields['output:preference_logits']
    mean = math.fsum(logits)/len(logits)
    fields['output:centered_preference_logits'] = [v-mean for v in logits]
    geometry = {'candidates': sorted(candidates), 'concepts': sorted(concepts), 'sources': sorted(sources),
                'conditional_mask': relation(conditional['concept_source_mask']),
                'view_masks': {k: vector(v, view_order) for k, v in latent['view_masks'].items()}}
    return fields, geometry


def validate_saved_authority(case, decision):
    expected = {e.record.record_id: asdict(e.record) for e in case.document.sources}
    actual = {r['record_id']: r for r in decision['sources'][0]}
    require(len(actual) == len(decision['sources'][0]) and actual == expected)
    concepts = {e.record.concept_id: asdict(e.record) for e in case.document.concepts}
    require({r['concept_id']: r for r in decision['concepts'][0]} == concepts)
    require(set(decision['candidate_ids'][0]) == {c.candidate_id for c in case.document.candidates})


def preference_detail(case, decision):
    target, = [t for t in case.targets if t.field == 'preferences']
    require(target.mode == 'closed_distribution')
    facts = {f.pointers[0]: f.value for f in target.facts if f.available}
    ids = decision['candidate_ids'][0]
    require(set(facts) == set(ids) and len(ids) == len(set(ids)))
    logits = dict(zip(ids, decision['preference_logits'][0]))
    require(all(math.isfinite(v) for v in logits.values()))
    expected = {k for k, v in facts.items() if v == max(facts.values())}
    probabilities = dict(zip(ids, decision['preferences'][0]))
    require(len(decision['preferences'][0]) == len(ids) and all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values()))
    actual = {k for k, v in probabilities.items() if v == max(probabilities.values())}
    others = set(ids)-expected
    # Worst expected top versus best outside candidate; ties within expected are
    # measured separately and exact top-set reporting is retained unchanged.
    margin = min(logits[k] for k in expected)-max(logits[k] for k in others) if others else None
    return {'target_top_count': len(expected), 'exact_top_set_correct': expected == actual,
            'target_set_vs_outside_margin': margin,
            'within_expected_top_spread': max(logits[k] for k in expected)-min(logits[k] for k in expected),
            'predicted_logit_spread': max(logits.values())-min(logits.values())}


def conditional_target_detail(cases, decisions, aliases):
    result = {}
    for field in CONDITIONAL:
        truth, predicted = [], []
        for case, decision, alias in zip(cases, decisions, aliases):
            targets = [t for t in case.targets if t.field == field]
            require(len(targets) == 1)
            normalize = lambda pointers: tuple(alias.get(p, p) for p in pointers)
            truth.append({normalize(f.pointers): f.value for f in targets[0].facts if f.available})
            ids = decision['concept_ids'][0]
            values = decision['conditional_concepts'][field][0]
            if 'source_' in field:
                source_ids = [r['record_id'] for r in decision['sources'][0]]
                predicted.append({(cid, alias[sid]): values[i][j] for i, cid in enumerate(ids) for j, sid in enumerate(source_ids)})
            else:
                predicted.append({(cid,): values[i] for i, cid in enumerate(ids)})
        counts, signed, errors = Counter(), [], []
        for facts, outputs in zip(truth, predicted):
            require(set(facts) <= set(outputs))
            for key, target in facts.items():
                value = outputs[key]
                require(math.isfinite(value) and 0 <= value <= 1 and target is not None)
                errors.append(abs(value-target))
                if target in (0, 1):
                    counts['binary_facts'] += 1
                    counts['binary_errors'] += (value >= .5) != bool(target)
        for key in truth[0].keys() & truth[1].keys():
            delta = truth[1][key]-truth[0][key]
            if delta:
                signed.append((1 if delta > 0 else -1)*(predicted[1][key]-predicted[0][key]))
        result[field] = {**dict(counts), 'known_facts': len(errors), 'mean_absolute_error': math.fsum(errors)/len(errors) if errors else None,
                         'changed_target_comparisons': len(signed), 'correct_changed_target_direction': sum(v > 0 for v in signed),
                         'target_signed_condition_deltas': signed}
    return result


def summarize_pairs(pairs, n1_by_document, decisions, saved):
    details = []
    for left, right in pairs:
        aliases = [source_aliases(c) for c in (left, right)]
        # Role comparison requires identical authority flags and source kinds.
        authority = []
        for case, alias in zip((left, right), aliases):
            authority.append({alias[e.record.record_id]: {k: v for k, v in asdict(e.record).items()
                if k not in ('record_id', 'source_family_id')} for e in case.document.sources})
        require(authority[0] == authority[1])
        outputs = [decisions[c.case_id] for c in (left, right)]
        for case, decision in zip((left, right), outputs):
            validate_saved_authority(case, decision)
            interpretation = n1_by_document[case.document.sha256]
            require(interpretation.document.sha256 == case.document.sha256 and interpretation.lineage == case.lineage)
        coordinates = [coordinate_fields(d, a) for d, a in zip(outputs, aliases)]
        require(coordinates[0][1] == coordinates[1][1] and coordinates[0][0].keys() == coordinates[1][0].keys())
        details.append({'case_sha256': [left.sha256, right.sha256],
            'coordinate_changes': {k: saved.difference(v, coordinates[1][0][k]) for k, v in coordinates[0][0].items()},
            'preferences': [preference_detail(c, d) for c, d in zip((left, right), outputs)],
            'n1_targets': conditional_target_detail([n1_by_document[c.document.sha256] for c in (left, right)], outputs, aliases)})
    require(len(details) > 0)
    summary = {'source_pairs': len(details), 'coordinate_changes': {}, 'preferences': {}, 'n1_targets': {}}
    for field in details[0]['coordinate_changes']:
        summary['coordinate_changes'][field] = {metric: distribution([r['coordinate_changes'][field][metric] for r in details
            if r['coordinate_changes'][field][metric] is not None]) for metric in ('rms_delta', 'relative_rms_delta', 'max_absolute_delta')}
    preferences = [p for row in details for p in row['preferences']]
    summary['preferences'] = {'contexts': len(preferences), 'tied_target_top_sets': sum(p['target_top_count'] > 1 for p in preferences),
        'exact_top_sets_correct': sum(p['exact_top_set_correct'] for p in preferences),
        **{k: distribution([p[k] for p in preferences if p[k] is not None]) for k in
           ('target_set_vs_outside_margin', 'within_expected_top_spread', 'predicted_logit_spread')}}
    for field in CONDITIONAL:
        rows = [r['n1_targets'][field] for r in details]
        summary['n1_targets'][field] = {k: sum(r.get(k, 0) for r in rows) for k in
            ('binary_facts', 'binary_errors', 'known_facts', 'changed_target_comparisons', 'correct_changed_target_direction')}
        summary['n1_targets'][field].update({'family_mean_absolute_errors': distribution([r['mean_absolute_error'] for r in rows if r['mean_absolute_error'] is not None]),
            'target_signed_condition_deltas': distribution([v for r in rows for v in r['target_signed_condition_deltas']])})
    return details, summary


def run(namespace):
    isolation = namespace._p2_guard()  # FIRST, before protected path/stat/read.
    os.umask(0o077)
    sys.dont_write_bytecode = True
    code = namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'], ROOT, file=False)
    namespace._clean_code(code, REV)
    sys.path.insert(0, str(code))
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import raw_inference_lineage as raw
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    with ExitStack() as stack:
        held = []
        def pin(path, digest, limit=256*1024**2):
            path = namespace._bounded_path(str(path), ROOT, file=True)
            require(stat.S_IMODE(path.stat().st_mode) == 0o600)
            value = stack.enter_context(_PinnedFile(path, digest, limit))
            held.append(value)
            return value
        def read(path, digest):
            return raw._json(pin(path, digest).bounded_bytes())
        def lines(path, digest):
            value = pin(path, digest)
            value.stream.seek(0)
            return [raw._json(line) for line in value.stream]
        modules = {}
        for key, (name, digest) in DEPENDENCIES.items():
            path = ROOT/'jobs'/name
            pin(path, digest)
            modules[key] = load_module('paired_readback_'+key, path)
        pair, saved, analyzer, live = [modules[k] for k in ('pair', 'saved', 'analysis', 'live')]
        live.verify_active_namespace(namespace, isolation)
        result = pair.verify_seal(read(ROOT/'jobs/rayan-paired-update-live-custody-20261021.json', RESULT_SHA))
        require(result['job_id'] == 578772 and result['complete_source_and_checkpoint_custody_verified'])
        original = pair.verify_seal(read(ROOT/'jobs/rayan-cumulative-learning-r3-20261015.json', pair.SUMMARY_SHA))
        plan = read(ROOT/'rayan-personality/private-inputs/rayan-cumulative-learning-r3-20261015/owner-cumulative-run.protected.json', pair.PLAN_SHA)
        reference = plan['packets']['broader']
        directory = namespace._bounded_path(reference['directory'], ROOT, file=False)
        index = read(directory/'packet-index.protected.json', reference['index_sha256'])
        allocation = read(directory/'construction-allocation.protected.json', index['files']['construction-allocation.protected.json'])
        case_rows = lines(directory/'reviewed-teaching-cases.protected.jsonl', index['files']['reviewed-teaching-cases.protected.jsonl'])
        cases = [ReviewedTeachingCase.from_mapping(r) for r in case_rows]
        require(len(cases) == len({c.case_id for c in cases}) == 160)
        require({c.case_id: c.sha256 for c in cases} == index['case_sha256'])
        n1 = {c.document.sha256: c for c in cases if c.phase == 'n1'}
        n2 = [c for c in cases if c.phase == 'n2']
        require(len(n1) == len(n2) == 80)
        pairs = pair.source_pairs(n2, allocation)
        by_document = {c.document.sha256: c for c in n2}
        require(len(by_document) == 80)
        old_output = ROOT/'rayan-personality/private-compile/rayan-cumulative-learning-r3-577714'
        source_output = ROOT/'rayan-personality/private-compile/rayan-paired-update-live-custody-578772'
        metadata = original['successor_full_governed_responses']
        base_objectives = [r for r in lines(old_output/'successor-all-reviewed-objectives.protected.jsonl', metadata['all_reviewed_objective_file_sha256']) if r['phase'] == 'n2']
        require(len(base_objectives) == 216)
        output = ROOT/'rayan-personality/private-compile'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
        output.mkdir(mode=0o700)
        summaries, private_files, baseline, baseline_bindings, n1_digests = {}, {}, {}, {}, {}
        diagnostic_bindings = None
        for stage in ('baseline', 'matched', 'control'):
            live.verify_active_namespace(namespace, isolation)
            if stage == 'baseline':
                rows = lines(old_output/'successor-governed-responses.protected.jsonl', metadata['file_sha256'])
                expected_model = pair.CURRENT_MODEL_SHA
            else:
                arm = result['arms'][stage]
                rows = lines(source_output/(stage+'-candidate-native-packets.protected.jsonl'), arm['response_file_sha256'])
                expected_model = arm['model_state_sha256']
                objectives = lines(source_output/(stage+'-all-n2-objectives.protected.jsonl'), arm['objective_file_sha256'])
                require(len(objectives) == 216 and analyzer.compare_objectives(base_objectives, objectives) == arm['objective_comparison'])
            require(len(rows) == len({(r['packet'], r['document_id']) for r in rows}) == 236)
            decisions = {}
            for row in rows:
                key = (row['packet'], row['document_id'])
                response = row['response'] if stage == 'baseline' else row
                decision = response['decision']
                if stage == 'baseline':
                    pair.verify_seal(response)
                    selected = pair.verify_seal(response['preference_selection'])
                    require(selected['model_state_sha256'] == expected_model and response['packet_sha256'] == selected['packet_sha256'])
                    doc_sha = selected['document_sha256']
                    baseline_bindings[key] = doc_sha
                else:
                    require(response['arm'] == stage and response['model_state_sha256'] == expected_model
                            and response['native_checkpoint'] is False and response['native_consumer_admitted'] is False)
                    doc_sha = row['document_sha256']
                    require(baseline_bindings[key] == doc_sha)
                n1_digest = sha256(canonical({'conditional': decision['conditional_concepts'], **{k: decision['native_latent'][k]
                    for k in ('concepts', 'concept_mask', 'views', 'view_masks')}})).hexdigest()
                if stage == 'baseline':
                    n1_digests[key] = n1_digest
                else:
                    require(n1_digests[key] == n1_digest)
                if row['packet'] == 'broader':
                    require(doc_sha in by_document)
                    decisions[by_document[doc_sha].case_id] = decision
            require(len(decisions) == 80)
            details, summaries[stage] = summarize_pairs(pairs, n1, decisions, saved)
            if stage == 'baseline':
                baseline = decisions
            else:
                require(pair.paired_region_comparison(pairs, baseline, decisions) == result['arms'][stage]['broader_pair_comparison'])
            expected_top = result['arms']['matched' if stage == 'baseline' else stage]['broader_pair_comparison']['fields']['preferences'][
                ('before' if stage == 'baseline' else 'after')+'_correct_top_sets']
            require(summaries[stage]['preferences']['exact_top_sets_correct'] == expected_top)
            diagnostics = lines(source_output/(stage+'-raw-native-diagnostics.protected.jsonl'), result['diagnostic_private_file_sha256'][stage])
            bindings = {r['document_id']: (r['document_sha256'], r['parent_document_id'], r['comparison_kind']) for r in diagnostics}
            if diagnostic_bindings is None:
                diagnostic_bindings = bindings
            require(bindings == diagnostic_bindings and all(r['response']['model_state_sha256'] == expected_model for r in diagnostics))
            summaries[stage]['diagnostic_coordinate_changes'] = saved.summarize(diagnostics)
            target = output/(stage+'-family-readback.protected.json')
            with target.open('xb') as stream:
                stream.write(canonical(details)+b'\n')
            private_files[stage] = sha256(target.read_bytes()).hexdigest()
            del rows, decisions, diagnostics, details
            gc.collect()
            print(json.dumps({'stage': 'complete_saved_arm_readback', 'arm': stage, 'broader_pairs': 40, 'native_documents': 236, 'new_gradient_steps': 0}), flush=True)
        live.verify_active_namespace(namespace, isolation)
        for value in held:
            value.verify()
        namespace._clean_code(code, REV)
        receipt = {'schema': 'alice-personality-complete-paired-saved-readback-v1', 'owner': 'mxrayan',
            'job_id': int(os.environ['SLURM_JOB_ID']), 'source_job_id': 578772, 'source_receipt_file_sha256': RESULT_SHA,
            'worker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'namespace_sha256': GUARD, 'isolation': isolation,
            'dependencies': DEPENDENCIES, 'complete_saved_input_custody_verified': True,
            'native_document_records': 708, 'native_n2_objective_records': 648, 'diagnostic_records': 30,
            'full_objective_and_pair_comparisons_reproduced': True, 'n1_exact_for_all_documents': True,
            'source_role_alignment_is_not_evidence_identity': True, 'private_detail_file_sha256': private_files,
            'stages': summaries, 'new_forward_passes': 0, 'new_gradient_steps': 0, 'causal_intervention': False,
            'independent_qualification': False, 'n0_approved': False, 'personality_approved': False, 'private_payload_exported': False}
        receipt['receipt_sha256'] = sha256(canonical(receipt)).hexdigest()
        with (ROOT/'jobs'/(NAME+'.json')).open('xb') as stream:
            stream.write(canonical(receipt)+b'\n')
        print(json.dumps({'stage': 'complete_saved_readback', 'receipt_sha256': receipt['receipt_sha256']}), flush=True)


if __name__ == '__main__':
    try:
        path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
        require(sha256(path.read_bytes()).hexdigest() == GUARD)
        run(load_module('paired_readback_namespace', path))
    except BaseException as error:
        try:
            name, digest = DEPENDENCIES['live']
            path = ROOT/'jobs'/name
            require(sha256(path.read_bytes()).hexdigest() == digest)
            live = load_module('paired_readback_error', path)
            report = {'state': 'PAIRED_READBACK_REFUSED', 'exception_chain': live.public_exception_chain(error)}
        except BaseException as reporter_error:
            report = {'state': 'PAIRED_READBACK_REFUSED', 'exception_type': type(error).__name__, 'reporter_exception_type': type(reporter_error).__name__}
        try:
            print(json.dumps(report), flush=True)
        finally:
            raise SystemExit(2) from None
