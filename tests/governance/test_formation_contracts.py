from __future__ import annotations

from dataclasses import replace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import FormationProposal, MemoryProposalBundle


def _bundle(product: str = "alice") -> MemoryProposalBundle:
    return MemoryProposalBundle(
        scope=ProductHostScope.create(
            product_id=product,
            host_instance_id="owner-instance",
            schema_version="1.0.0",
            encryption_domain="private-owner",
        ),
        authority_namespace_id="owner-namespace",
        bundle_id="bundle-1",
        experience_refs=("experience-1",),
        context_digest="a" * 64,
        model_artifact_digest="b" * 64,
        inference_run_id="run-1",
        proposals=(FormationProposal(
            proposal_id="proposal-1",
            kind="correction_request",
            domain="host",
            subject_ref="entity-1",
            value_ref="private-value-1",
            evidence_refs=("experience-1",),
            valid_from="2026-09-23T00:00:00Z",
        ),),
    )


class FormationContractsTests(unittest.TestCase):
    def test_bundle_is_host_scoped_and_backend_neutral(self) -> None:
        alice = _bundle()
        friday = _bundle("friday")
        alice_record = alice.metadata_record()
        friday_record = friday.metadata_record()
        self.assertNotEqual(alice.content_digest(), friday.content_digest())
        self.assertEqual(alice_record["scope"]["product_id"], "alice")
        self.assertEqual(friday_record["scope"]["product_id"], "friday")
        self.assertNotIn("backend", alice_record)
        self.assertNotIn("adjudication", alice_record)
        self.assertEqual(alice_record["proposals"][0]["kind"], "correction_request")

    def test_proposals_require_evidence_and_reject_authority(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "evidence"):
            replace(bundle, proposals=(replace(bundle.proposals[0], evidence_refs=()),)).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "kind"):
            replace(bundle, proposals=(replace(bundle.proposals[0], kind="authoritative_write"),)).validate()

    def test_duplicate_proposals_and_backward_time_fail(self) -> None:
        bundle = _bundle()
        with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate"):
            replace(bundle, proposals=bundle.proposals * 2).validate()
        with self.assertRaisesRegex(CognitiveKernelContractError, "valid_to"):
            replace(bundle, proposals=(replace(
                bundle.proposals[0], valid_to="2026-09-22T00:00:00Z"
            ),)).validate()


if __name__ == "__main__":
    unittest.main()
