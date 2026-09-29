from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_context_planner import (
    FormationPlanningRequest, FormationRetrievalHit, assemble_formation_context,
)
from cognitive_kernel.formation_contracts import FormationEvidenceRef


class FakeStore:
    def __init__(self, refs, content, denied=()):
        self.refs = {r.ref_id: r for r in refs}
        self.content = content
        self.denied = set(denied)
        self.reads = []

    def resolve(self, ref_id):
        return self.refs.get(ref_id)

    def permits_formation(self, ref):
        return ref.ref_id not in self.denied

    def read(self, ref):
        self.reads.append(ref.ref_id)
        return self.content[ref.ref_id]


class FormationPlannerTests(unittest.TestCase):
    def setUp(self):
        self.scope = ProductHostScope.create(
            product_id="alice", host_instance_id="fictional-host",
            schema_version="1.0.0", encryption_domain="fictional-private",
        )
        self.content = {"new-event": b"owner said X", "prior-claim": b"old X",
                        "episode": b"observed outcome", "other": b"someone else's life"}

        def ref(ref_id, role="authenticated_owner_statement", scope=None):
            return FormationEvidenceRef(
                ref_id=ref_id, scope=scope or self.scope,
                authority_namespace_id="fictional-ns",
                content_digest=sha256(self.content[ref_id]).hexdigest(),
                role=role, modality="text", subject_ref="fictional-host",
            )

        self.refs = (ref("new-event"), ref("prior-claim"),
                     ref("episode", "tool_or_action_observation"))
        self.request = FormationPlanningRequest(
            scope=self.scope, authority_namespace_id="fictional-ns",
            experience_refs=("new-event",),
            candidates=(FormationRetrievalHit("prior-claim", "claim"),
                        FormationRetrievalHit("episode", "episode"),
                        FormationRetrievalHit("prior-claim", "graph")),
        )

    def test_scoped_multiplane_packet_and_digest_checked_content(self):
        store = FakeStore(self.refs, self.content)
        result = assemble_formation_context(self.request, store)
        self.assertEqual(tuple(r.ref_id for r in result.packet.evidence),
                         ("new-event", "prior-claim", "episode"))
        self.assertEqual(result.selected_planes[1], ("prior-claim", ("claim", "graph")))
        self.assertEqual(store.reads, ["new-event", "prior-claim", "episode"])
        self.assertNotIn("owner said X", str(result.packet.metadata_record()))

    def test_zero_historical_planes_and_selector_cannot_invent_source(self):
        store = FakeStore(self.refs, self.content)
        result = assemble_formation_context(
            self.request, store, lambda request, refs, hits: ())
        self.assertEqual(tuple(result.opened_content), (("new-event", b"owner said X"),))
        self.assertEqual(store.reads, ["new-event"])
        with self.assertRaisesRegex(CognitiveKernelContractError, "unregistered"):
            assemble_formation_context(
                self.request, store, lambda request, refs, hits: ("invented",))

    def test_denied_foreign_and_missing_candidates_fail_closed(self):
        denied = FakeStore(self.refs, self.content, denied=("episode",))
        with self.assertRaisesRegex(CognitiveKernelContractError, "not permitted"):
            assemble_formation_context(self.request, denied)
        self.assertEqual(denied.reads, [])
        other_scope = replace(self.scope, host_instance_id="other-host")
        foreign = FakeStore((*self.refs[:2], replace(self.refs[2], scope=other_scope)), self.content)
        with self.assertRaisesRegex(CognitiveKernelContractError, "foreign-scope"):
            assemble_formation_context(self.request, foreign)
        missing = FakeStore(self.refs[:2], self.content)
        with self.assertRaisesRegex(CognitiveKernelContractError, "unregistered"):
            assemble_formation_context(self.request, missing)

    def test_mutated_bytes_and_forged_role_fail(self):
        content = {**self.content, "episode": b"silently changed"}
        with self.assertRaisesRegex(CognitiveKernelContractError, "digest mismatch"):
            assemble_formation_context(self.request, FakeStore(self.refs, content))
        forged = replace(self.refs[1], role="made-up-owner")
        with self.assertRaisesRegex(CognitiveKernelContractError, "role"):
            assemble_formation_context(self.request, FakeStore((self.refs[0], forged, self.refs[2]), self.content))


if __name__ == "__main__":
    unittest.main()
