from __future__ import annotations

from dataclasses import replace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_context_planner import FormationPlanningRequest
from cognitive_kernel.formation_retrieval import FormationRoute, gather_formation_candidates


class Plane:
    def __init__(self, scope, namespace, sources):
        self.scope = scope
        self.authority_namespace_id = namespace
        self.sources = sources

    def source_ids(self, query_ref):
        return self.sources[query_ref]


class FormationRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.scope = ProductHostScope.create(
            product_id="alice", host_instance_id="fictional-host",
            schema_version="1.0.0", encryption_domain="fictional-custody",
        )
        self.request = FormationPlanningRequest(
            scope=self.scope, authority_namespace_id="namespace-1",
            experience_refs=("new-event",),
        )

    def test_multiple_routes_can_return_same_underlying_source(self):
        providers = {
            "graph": Plane(self.scope, "namespace-1", {"entity-1": ("source-1",)}),
            "vector": Plane(self.scope, "namespace-1", {"query-1": ("source-1", "source-2")}),
        }
        result = gather_formation_candidates(
            request=self.request,
            routes=(FormationRoute("graph", "entity-1"),
                    FormationRoute("vector", "query-1")), providers=providers)
        self.assertEqual([(x.ref_id, x.plane) for x in result.candidates],
                         [("source-1", "graph"), ("source-1", "vector"),
                          ("source-2", "vector")])
        self.assertEqual(gather_formation_candidates(
            request=self.request, routes=(), providers=providers).candidates, ())

    def test_unknown_or_foreign_plane_fails(self):
        with self.assertRaisesRegex(CognitiveKernelContractError, "no registered plane"):
            gather_formation_candidates(request=self.request,
                                        routes=(FormationRoute("unknown", "query-1"),), providers={})
        foreign = replace(self.scope, host_instance_id="another-host")
        with self.assertRaisesRegex(CognitiveKernelContractError, "crosses host"):
            gather_formation_candidates(request=self.request,
                routes=(FormationRoute("graph", "query-1"),),
                providers={"graph": Plane(foreign, "namespace-1", {"query-1": ("source-1",)})})


if __name__ == "__main__":
    unittest.main()
