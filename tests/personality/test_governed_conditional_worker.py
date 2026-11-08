"""Execute the worker's actual interface assertions against native public data."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest

from src.alice_personality.identity.consumer import DecisionConsumer
from src.alice_personality.identity.contracts import IdentityModelConfig
from src.alice_personality.identity.feature_producer import ProducedFrame
from scripts.eipm.gemma_n0.governed_response import respond_with_governance
from tests.personality import test_identity_consumer as consumer_fixture
from tests.personality.test_identity_model import _frame

WORKER=Path(__file__).resolve().parents[2]/'scripts/eipm/gemma_n0/magnolia_governed_conditional_20261011.py'

def selected_assertions(predicate):
    tree=ast.parse(WORKER.read_bytes())
    statements=[node for node in ast.walk(tree) if isinstance(node,ast.Assert) and predicate(ast.unparse(node.test))]
    if len(statements)!=1:
        raise AssertionError('The worker must have exactly one actual assertion for this binding')
    return compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(WORKER),'exec')

class GovernedConditionalWorkerTests(unittest.TestCase):
    def test_current_episode_ids_do_not_reject_the_identity_matched_query_donor(self):
        config=IdentityModelConfig(provider_width=8,provider_state_count=3,learned_width=8,attention_heads=2)
        frame=_frame(config,batch=1)
        records=list(frame.source_records[0]);records[2]=replace(records[2],record_id='another-current-host')
        ids=list(frame.sources.entry_ids[0]);ids[2]='another-current-host'
        donor=replace(frame,source_records=(tuple(records),),sources=replace(frame.sources,entry_ids=(tuple(ids),)))
        donor.validate(config)
        condition=selected_assertions(lambda text:'donor_produced.frame' in text)
        exec(condition,{'produced':ProducedFrame(frame,{}),'donor_produced':ProducedFrame(donor,{})})

    def test_worker_verifies_the_actual_governed_response_model_binding(self):
        fixture=consumer_fixture.ConsumerTests();fixture.setUp()
        try:
            consumer=fixture.build(co_valid=True)
            self.assertIsInstance(consumer,DecisionConsumer)
            governed=respond_with_governance(consumer,fixture.doc,fixture.produced,fixture.session)
            summary={'model_state_sha256':governed['preference_selection']['model_state_sha256']}
            condition=selected_assertions(lambda text:'governed[' in text and 'model_state_sha256' in text)
            exec(condition,{'governed':governed,'summary':summary})
        finally:
            fixture.doCleanups()
