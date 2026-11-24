from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

from scripts.eipm.gemma_n0 import magnolia_conditional_channel_probe_20261023 as api
from scripts.eipm.gemma_n0 import magnolia_paired_readback_20261023 as saved
from tests.personality.test_paired_update_worker import PairedUpdateWorkerTests


class ChannelWorkerTests(unittest.TestCase):
    def test_changed_known_facts_use_donor_predictions_and_recipient_source_ids(self):
        cases, _ = PairedUpdateWorkerTests().pair()
        concept = cases[0].document.concepts[0].record.concept_id
        prepared=[]
        for i,case in enumerate(cases):
            host=case.document.sources[-1].record.record_id
            target=lambda field,pointer,value,available=True:NS(field=field,facts=[NS(pointers=pointer,value=value,available=available)])
            prepared.append(replace(case,targets=tuple([
                target('concept_applicability',(concept,),float(i)),
                target('concept_uncertainty',(concept,),.2),
                target('concept_source_support',(concept,host),float(i)),
                target('concept_source_disconfirmation',(concept,host),None if i==0 else 1.,i!=0)])))
        donor={'concept_ids':[[concept]],'sources':[[{'record_id':e.record.record_id} for e in cases[1].document.sources]],
               'conditional_concepts':{'concept_applicability':[[.7]],'concept_source_support':[[[.1,.8]]]}}
        values=api.changed_channel_values(*prepared,donor,saved)
        self.assertEqual(values['reviewed_current']['concept_applicability'],{(concept,):0.})
        self.assertEqual(values['reviewed_partner']['concept_applicability'],{(concept,):1.})
        self.assertEqual(values['natural_partner']['concept_applicability'],{(concept,):.7})
        pointer=(concept,cases[0].document.sources[-1].record.record_id)
        self.assertEqual(values['natural_partner']['concept_source_support'],{pointer:.8})
        for row in values.values():
            self.assertEqual(row['concept_uncertainty'],{})
            self.assertEqual(row['concept_source_disconfirmation'],{})

    def test_worker_guard_precedes_private_operations(self):
        class Refused(Exception):pass
        class Namespace:
            def _p2_guard(self):raise Refused()
        with self.assertRaises(Refused):api.run(Namespace())

    def test_public_progress_and_no_optimizer_step(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path(api.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in calls))
        for call in calls:
            if isinstance(call.func,ast.Name) and call.func.id=='print':
                for d in ast.walk(call):
                    if isinstance(d,ast.Dict):
                        keys={k.value for k in d.keys if isinstance(k,ast.Constant)}
                        self.assertFalse(keys & {'case_id','document_id','case_sha256','document_sha256'})


if __name__=='__main__':unittest.main()
