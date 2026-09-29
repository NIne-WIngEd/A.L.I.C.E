from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, normalize_timestamp
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_context_planner import (
    FormationPlanningRequest, FormationRetrievalHit, assemble_formation_context,
)
from cognitive_kernel.formation_contracts import FormationEvidenceRef
from cognitive_kernel.formation_sources import (
    RegisteredFormationSource, RegisteredFormationStore, record_formation_read,
)


class Registry:
    def __init__(self, sources):
        self.sources = {source.evidence.ref_id: source for source in sources}

    def lookup(self, ref_id):
        return self.sources.get(ref_id)


class Custody:
    def __init__(self, objects, on_read=None):
        self.objects, self.on_read = objects, on_read

    def read(self, scope, object_ref):
        if self.on_read:
            self.on_read()
        return self.objects[object_ref]


class RegisteredFormationTests(unittest.TestCase):
    def setUp(self):
        self.scope = ProductHostScope.create(
            product_id="alice", host_instance_id="fictional-host-a",
            schema_version="1.0.0", encryption_domain="fictional-private-a",
        )
        self.objects = {"object-one": b"I now prefer fewer meetings.",
                        "object-two": b"She wrote 'I prefer meetings' in a quotation."}
        def evidence(ref_id, object_id, role, speaker, time):
            return FormationEvidenceRef(
                ref_id=ref_id, scope=self.scope, authority_namespace_id="owner-ns",
                content_digest=sha256(self.objects[object_id]).hexdigest(),
                role=role, modality="text", subject_ref="fictional-host-a",
                speaker_ref=speaker, source_item_ref=object_id,
                observed_at=normalize_timestamp(time), recorded_at=normalize_timestamp(time),
                duplicate_group_ref=ref_id,
            )
        self.one = RegisteredFormationSource.create(evidence=evidence(
            "event-one", "object-one", "authenticated_owner_statement",
            "fictional-host-a", "2026-01-02T00:00:00Z"), object_ref="object-one")
        self.two = RegisteredFormationSource.create(evidence=evidence(
            "event-two", "object-two", "outside_source", "fictional-other",
            "2026-01-01T00:00:00Z"), object_ref="object-two")
        self.registry = Registry((self.one, self.two))
        self.allowed = {"event-one", "event-two"}
        self.custody = Custody(self.objects)
        self.request = FormationPlanningRequest(
            scope=self.scope, authority_namespace_id="owner-ns",
            experience_refs=("event-one",),
            candidates=(FormationRetrievalHit("event-two", "source_native"),),
            as_of=normalize_timestamp("2026-01-03T00:00:00Z"),
        )

    def store(self):
        return RegisteredFormationStore(
            scope=self.scope, authority_namespace_id="owner-ns",
            registry=self.registry, custody=self.custody,
            permits=lambda source, purpose: source.evidence.ref_id in self.allowed,
        )

    def test_source_roles_time_and_read_receipt_are_bound(self):
        store = self.store()
        result = assemble_formation_context(self.request, store)
        self.assertEqual(result.packet.evidence[1].speaker_ref, "fictional-other")
        self.assertNotEqual(result.packet.evidence[0].duplicate_group_ref,
                            result.packet.evidence[1].duplicate_group_ref)
        receipt = record_formation_read(result, store)
        self.assertEqual(receipt.context_digest, result.packet.content_digest())
        self.assertEqual(receipt.opened[1][3], ("source_native",))
        self.assertNotIn("prefer fewer meetings", str(receipt))

    def test_future_record_and_revocation_reject(self):
        store = self.store()
        with self.assertRaisesRegex(CognitiveKernelContractError, "future-recorded"):
            assemble_formation_context(replace(
                self.request, as_of=normalize_timestamp("2026-01-01T12:00:00Z")), store)
        self.allowed.remove("event-two")
        with self.assertRaisesRegex(CognitiveKernelContractError, "not permitted"):
            assemble_formation_context(self.request, store)

    def test_registration_change_and_read_race_reject(self):
        store = self.store()
        altered = replace(self.two, evidence=replace(self.two.evidence,
                          role="authenticated_owner_statement"))
        self.registry.sources["event-two"] = altered
        with self.assertRaisesRegex(CognitiveKernelContractError, "registration changed"):
            assemble_formation_context(self.request, store)
        self.registry.sources["event-two"] = self.two
        self.custody.on_read = lambda: self.allowed.remove("event-one")
        with self.assertRaisesRegex(CognitiveKernelContractError, "revoked"):
            assemble_formation_context(self.request, store)

    def test_changed_bytes_and_foreign_registration_reject(self):
        original = self.custody.objects["object-two"]
        self.custody.objects["object-two"] = b"tampered"
        with self.assertRaisesRegex(CognitiveKernelContractError, "content verification"):
            assemble_formation_context(self.request, self.store())
        self.custody.objects["object-two"] = original
        other_scope = replace(self.scope, host_instance_id="fictional-host-b")
        foreign = RegisteredFormationSource.create(
            evidence=replace(self.two.evidence, scope=other_scope), object_ref="object-two")
        self.registry.sources["event-two"] = foreign
        with self.assertRaisesRegex(CognitiveKernelContractError, "crosses scope"):
            assemble_formation_context(self.request, self.store())


if __name__ == "__main__":
    unittest.main()
