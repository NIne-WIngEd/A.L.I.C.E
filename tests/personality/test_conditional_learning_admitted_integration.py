"""Actual public materialization, corrections, native grouped learning and resume.

The fixture is explicitly synthetic; it neither admits private gradients nor
claims the learned model is better or qualified.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from scripts.eipm.gemma_n0 import conditional_learning_experiment as runner
from scripts.eipm.gemma_n0 import conditional_teaching as teaching
from src.alice_personality.identity import IdentityModel
from src.alice_personality.identity.codec import SourceEntry, fingerprint
from src.alice_personality.identity.contracts import SourceRecord
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching import IdentityObjectives, TrainingRecipe
from src.alice_personality.teaching.cases import NamedTarget, TargetFact
from src.alice_personality.teaching.layout import target_layouts
from src.alice_personality.teaching.materialize import materialize_case, verify_materialized_batch
from tests.personality.test_conditional_teaching import public_fixture
from tests.personality.test_identity_codec import Backbone, Tokenizer
from tests.personality.test_paired_learning_recovery import exact
from tests.personality.test_reviewed_teaching_cases import PINS


def complete_targets(frame):
    """Explicit public synthetic values for every native N2 target family."""
    targets = []
    for field, layout in target_layouts(frame, 'n2').items():
        facts = []
        for coordinates in layout.allowed.nonzero().tolist():
            index = tuple(coordinates)
            pointers = tuple(axis[position] for axis, position in zip(layout.axes, coordinates[1:]))
            if layout.distribution:
                value = 1. / int(layout.allowed[index[:-1]].sum())
                if field == 'preferences':
                    value = (.2, .8)[coordinates[-1]]
            elif field in ('candidate_pair_probabilities', 'value_tradeoff_probabilities'):
                value = .7 if coordinates[-2] < coordinates[-1] else .3
            else:
                value = .2 + .2 * (sum(coordinates) % 3)
            facts.append(TargetFact(pointers, value, .1, True, (), ()))
        assert facts, 'Full native fixture must exercise every target family'
        targets.append(NamedTarget(field, 'closed_distribution' if layout.distribution else 'independent', tuple(facts)))
    return tuple(targets)


@pytest.fixture
def admitted_native():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(83)
    config, base_n1, base_n2, *_ = public_fixture()
    model = IdentityModel(config)
    objectives = IdentityObjectives(config.learned_width).requires_grad_(False).eval()
    recipe = TrainingRecipe('n2', .0003, 1., 89)
    batches, corrections, reviews, cases = {}, {}, {}, {}
    for number in range(4):
        doc = base_n1.document
        doc = replace(doc,
            sources=(*doc.sources, SourceEntry('Public fictional experience', SourceRecord(
                'experience', doc.sources[0].record.source_family_id, 'AEXP', 'A-EXP',
                False, True, False, True, True, False))),
            fields=tuple(replace(field, text=field.text + ' Public context ' + str(number))
                         if field.available else field for field in doc.fields),
            labels={family: tuple(replace(specs[0], label_id=family + '-' + str(i),
                         name=specs[0].name + ' ' + str(i)) for i in range(3))
                    for family, specs in doc.labels.items()})
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), config, _fixture=True)
        produced = producer.produce(doc)
        session = producer.close_and_verify()
        n1 = replace(base_n1, case_id='n1-context-' + str(number), document=doc)
        n2 = replace(base_n2, case_id='n2-context-' + str(number), document=doc,
                     targets=complete_targets(produced.frame))
        def materialize(case):
            return materialize_case(case, produced, session, config, expected_case_sha256=case.sha256,
                expected_artifact_sha256=PINS,
                family_splits={family: 'TRAIN' for family in case.source_family_ids}).batch
        b1, b2 = materialize(n1), materialize(n2)
        reviewed_n1 = {b1.fingerprint(): {'split': 'TRAIN', 'case_ids': [n1.case_id],
                                        'source_family_ids': [list(n1.source_family_ids)]}}
        record = teaching.build_correction_record(n1, n2, b1, b2, frame_binding=produced.binding,
                                                   reviewed_n1_batches=reviewed_n1)
        digest = b2.fingerprint()
        batches[digest], corrections[digest], cases[digest] = b2, record, n2
        reviews[n2.sha256] = dict(PINS)
    digests = list(batches)
    assert len(set(digests)) == 4
    artifacts = {name: SimpleNamespace(sha256=digest) for name, digest in PINS.items()}
    reviewed = {'reviewed_case_artifact_sha256': reviews}
    original = fingerprint(model.state_dict()), fingerprint(objectives.state_dict())
    def validate(batch):
        digest = batch.fingerprint()
        assert digest in batches and batch.source_class == 'public_mechanical_fixture'
        verify_materialized_batch(batch, artifacts, reviewed)
        assert batch.materialization_binding['case_sha256'] == cases[digest].sha256
        return digest
    def guard():
        assert original == (fingerprint(model.state_dict()), fingerprint(objectives.state_dict()))
    plan = runner.make_plan(model, objectives, recipe, batch_sha256=digests,
        matched_groups=[digests[:2], digests[2:]],
        correction_sha256={digest: teaching.record_sha256(record) for digest, record in corrections.items()},
        bindings={'source_class': 'public_mechanical_fixture', 'private_gradient_authorized': False})
    yield SimpleNamespace(model=model, objectives=objectives, recipe=recipe, batches=batches,
                          corrections=corrections, validate=validate, guard=guard, plan=plan)
    torch.set_num_threads(old_threads)


def run(native, directory, **options):
    return runner.run_experiment(native.model, native.objectives, native.recipe, plan=native.plan,
        expected_plan_sha256=runner.plan_sha256(native.plan), batch_loader=native.batches.__getitem__,
        validate_batch=native.validate, correction_loader=native.corrections.__getitem__,
        custody_guard=native.guard, progress_directory=directory, **options)


def test_admitted_corrections_reach_real_n2_with_full_native_objectives(admitted_native, tmp_path):
    native = admitted_native
    before = deepcopy(native.model.state_dict())
    result = run(native, tmp_path / 'whole')
    control, intervention = (result['arms'][name] for name in ('control', 'intervention'))
    exact(before, native.model.state_dict())
    assert result['total_committed_updates'] == 4
    differences = {name for name in before if not torch.equal(control['model_state'][name], intervention['model_state'][name])}
    assert differences and all(name.startswith('n2.') for name in differences)
    correction_changed_losses = []
    for arm in (control, intervention):
        assert arm['updates'] == 2 and arm['endpoint_forwards'] == 4
        for name in before:
            if name.startswith(('n1.', 'n3.')):
                assert torch.equal(before[name], arm['model_state'][name])
        exact(dict(native.objectives.state_dict()), arm['objective_state'])
        for event in arm['events']:
            for endpoint in event['endpoints']:
                batch = native.batches[endpoint['batch_sha256']]
                expected_fields = set(batch.targets)
                assert {'preferences', 'head:stance', 'head:values', 'voice_control_values',
                        'evidence_pointers', 'concept_pointers', 'value_tradeoff_probabilities'} <= expected_fields
                assert set(endpoint['objectives']) == expected_fields
                assert set(endpoint['views']) == {'natural', 'replay'}
                for view in endpoint['views'].values():
                    assert set(view['objectives']) == expected_fields
                    assert view['coverage'] == endpoint['coverage']
                if arm is control:
                    assert endpoint['views']['natural'] == endpoint['views']['replay']
                else:
                    correction_changed_losses.append(endpoint['views']['natural']['loss'] != endpoint['views']['replay']['loss'])
    assert any(correction_changed_losses)
    assert result['native_checkpoint'] is False and result['independent_qualification'] is False


def test_admitted_intervention_resume_is_exact_for_both_completed_arms(admitted_native, tmp_path):
    expected = run(admitted_native, tmp_path / 'whole')
    class StopAfterCommittedGroup(Exception):
        pass
    def interrupt(event):
        if event['arm'] == 'intervention' and event['update'] == 1:
            raise StopAfterCommittedGroup()
    with pytest.raises(StopAfterCommittedGroup):
        run(admitted_native, tmp_path / 'partial', on_event=interrupt)
    snapshot = tmp_path / 'partial' / 'group-0003'
    digest = sha256((snapshot / 'manifest.json').read_bytes()).hexdigest()
    resumed = run(admitted_native, tmp_path / 'resumed', resume_from=snapshot,
                  expected_resume_manifest_sha256=digest)
    exact(expected['arms'], resumed['arms'])
    assert resumed['new_updates_this_invocation'] == 1
    assert resumed['total_committed_updates'] == 4
