"""A.L.I.C.E. memory-plane candidate routing for formation context.

Derived indexes return registered source IDs, not authority. A planner chooses
routes. This module imposes no fixed ontology or count cap on memory planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .canonical import CognitiveKernelContractError, require_identifier
from .contracts import ProductHostScope
from .formation_context_planner import FormationPlanningRequest, FormationRetrievalHit


@dataclass(frozen=True)
class FormationRoute:
    plane: str
    query_ref: str

    def validate(self) -> None:
        if (require_identifier(self.plane, "plane") != self.plane
                or require_identifier(self.query_ref, "query_ref") != self.query_ref):
            raise CognitiveKernelContractError("formation route must be canonical")


class FormationCandidatePlane(Protocol):
    scope: ProductHostScope
    authority_namespace_id: str

    def source_ids(self, query_ref: str) -> tuple[str, ...]: ...


def gather_formation_candidates(
    *, request: FormationPlanningRequest, routes: tuple[FormationRoute, ...],
    providers: Mapping[str, FormationCandidatePlane],
) -> FormationPlanningRequest:
    """Plan an arbitrary combination of planes; preserve exact route provenance.

    Provider output is untrusted until assemble_formation_context resolves each
    ID through the separate registered source/custody/policy boundary.
    """
    if request.candidates:
        raise CognitiveKernelContractError("retrieval request already contains candidate hits")
    hits: list[FormationRetrievalHit] = []
    for route in routes:
        route.validate()
        provider = providers.get(route.plane)
        if provider is None:
            raise CognitiveKernelContractError("formation route has no registered plane")
        if (provider.scope != request.scope
                or provider.authority_namespace_id != request.authority_namespace_id):
            raise CognitiveKernelContractError("formation route crosses host scope")
        ids = provider.source_ids(route.query_ref)
        if not isinstance(ids, tuple):
            raise CognitiveKernelContractError("formation plane must return source IDs")
        for ref_id in ids:
            hit = FormationRetrievalHit(ref_id, route.plane)
            hit.validate()
            hits.append(hit)
    return FormationPlanningRequest(
        scope=request.scope, authority_namespace_id=request.authority_namespace_id,
        experience_refs=request.experience_refs,
        candidates=tuple(hits), as_of=request.as_of,
    )
