"""Source-bound context assembly for Memory Formation Model inference.

Retrieval proposes candidate identifiers; only the registered evidence store can
resolve and open their bytes. The selector may be learned, but cannot introduce a
new source, expand its permission, or change its evidence role. This module does
not decide whether a resulting memory proposal becomes authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Protocol

from .canonical import CognitiveKernelContractError, normalize_timestamp, require_identifier
from .contracts import ProductHostScope
from .formation_contracts import FormationContextPacket, FormationEvidenceRef


@dataclass(frozen=True)
class FormationRetrievalHit:
    """A source identifier returned by any scoped memory/retrieval plane."""

    ref_id: str
    plane: str

    def validate(self) -> None:
        if require_identifier(self.ref_id, "ref_id") != self.ref_id:
            raise CognitiveKernelContractError("retrieval ref_id must be canonical")
        if require_identifier(self.plane, "plane") != self.plane:
            raise CognitiveKernelContractError("retrieval plane must be canonical")


class FormationEvidenceStore(Protocol):
    """Registered metadata and authorized bytes; implementations own custody."""

    def resolve(self, ref_id: str) -> FormationEvidenceRef | None: ...

    def read(self, ref: FormationEvidenceRef) -> bytes: ...

    def permits_formation(self, ref: FormationEvidenceRef) -> bool: ...


@dataclass(frozen=True)
class FormationPlanningRequest:
    scope: ProductHostScope
    authority_namespace_id: str
    experience_refs: tuple[str, ...]
    candidates: tuple[FormationRetrievalHit, ...] = ()
    as_of: str | None = None


@dataclass(frozen=True)
class AssembledFormationContext:
    packet: FormationContextPacket
    # Kept out of packet metadata and serialization; caller handles custody.
    opened_content: tuple[tuple[str, bytes], ...]
    selected_planes: tuple[tuple[str, tuple[str, ...]], ...]
    closure_refs: tuple[str, ...] = ()


FormationSelector = Callable[
    [FormationPlanningRequest, tuple[FormationEvidenceRef, ...], tuple[FormationRetrievalHit, ...]],
    tuple[str, ...],
]


def select_all_registered(
    request: FormationPlanningRequest,
    evidence: tuple[FormationEvidenceRef, ...],
    hits: tuple[FormationRetrievalHit, ...],
) -> tuple[str, ...]:
    """Exact-read reference policy for a baseline; not a trained planner."""
    return tuple(ref.ref_id for ref in evidence)


def assemble_formation_context(
    request: FormationPlanningRequest,
    store: FormationEvidenceStore,
    selector: FormationSelector = select_all_registered,
) -> AssembledFormationContext:
    """Bind a selector's output to registered, permitted, digest-checked evidence.

    The mandatory experience is opened even if no historical plane is selected.
    Repeated retrieval hits are collapsed by source ID while preserving every
    plane that found that source. Rejection is atomic: no partial packet escapes.
    """
    request.scope.validate()
    if not request.experience_refs:
        raise CognitiveKernelContractError("context needs an experience reference")
    if len(set(request.experience_refs)) != len(request.experience_refs):
        raise CognitiveKernelContractError("duplicate experience reference")
    for ref_id in request.experience_refs:
        if require_identifier(ref_id, "experience_ref") != ref_id:
            raise CognitiveKernelContractError("experience reference must be canonical")
    if require_identifier(request.authority_namespace_id, "authority_namespace_id") != request.authority_namespace_id:
        raise CognitiveKernelContractError("authority_namespace_id must be canonical")
    if request.as_of is not None and normalize_timestamp(request.as_of, "as_of") != request.as_of:
        raise CognitiveKernelContractError("as_of must be canonical")

    hits_by_ref: dict[str, list[str]] = {}
    for hit in request.candidates:
        hit.validate()
        planes = hits_by_ref.setdefault(hit.ref_id, [])
        if hit.plane not in planes:
            planes.append(hit.plane)
    all_ids = dict.fromkeys((*request.experience_refs, *hits_by_ref))
    registered: dict[str, FormationEvidenceRef] = {}
    def resolve_registered(ref_id: str) -> FormationEvidenceRef:
        if ref_id in registered:
            return registered[ref_id]
        ref = store.resolve(ref_id)
        if ref is None or ref.ref_id != ref_id:
            raise CognitiveKernelContractError("unregistered formation evidence")
        ref.validate()
        if ref.scope != request.scope or ref.authority_namespace_id != request.authority_namespace_id:
            raise CognitiveKernelContractError("foreign-scope formation evidence")
        if request.as_of is not None:
            if ref.recorded_at is None:
                raise CognitiveKernelContractError("historical context requires source record time")
            if normalize_timestamp(ref.recorded_at) > normalize_timestamp(request.as_of):
                raise CognitiveKernelContractError("context contains a future-recorded source")
        if not store.permits_formation(ref):
            raise CognitiveKernelContractError("formation evidence is not permitted")
        registered[ref_id] = ref
        return ref

    for ref_id in all_ids:
        resolve_registered(ref_id)

    eligible = tuple(registered.values())
    selected = selector(request, eligible, request.candidates)
    if not isinstance(selected, tuple) or len(set(selected)) != len(selected):
        raise CognitiveKernelContractError("selector must return unique reference IDs")
    if not all(isinstance(ref_id, str) and ref_id in registered for ref_id in selected):
        raise CognitiveKernelContractError("selector named unregistered evidence")
    # Retrieval of a derived item must bring its original evidence along. A
    # similarity hit is never sufficient provenance, including on read-time
    # reconstruction of an old episode or a superseded state.
    roots = tuple(dict.fromkeys((*request.experience_refs, *selected)))
    ordered_ids: list[str] = []
    state: dict[str, int] = {}
    closure_ids: set[str] = set()
    for root in roots:
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            ref_id, completed = stack.pop()
            if completed:
                state[ref_id] = 2
                ordered_ids.append(ref_id)
                continue
            if state.get(ref_id) == 2:
                continue
            if state.get(ref_id) == 1:
                raise CognitiveKernelContractError("formation evidence lineage cycle")
            ref = resolve_registered(ref_id)
            state[ref_id] = 1
            stack.append((ref_id, True))
            for parent in reversed(ref.parent_refs):
                if state.get(parent) == 1:
                    raise CognitiveKernelContractError("formation evidence lineage cycle")
                if state.get(parent) != 2:
                    closure_ids.add(parent)
                    stack.append((parent, False))
    ordered = tuple(ordered_ids)
    opened: list[tuple[str, bytes]] = []
    for ref_id in ordered:
        ref = registered[ref_id]
        content = store.read(ref)
        if not isinstance(content, bytes) or sha256(content).hexdigest() != ref.content_digest:
            raise CognitiveKernelContractError("formation evidence digest mismatch")
        opened.append((ref_id, content))
    packet = FormationContextPacket(
        scope=request.scope,
        authority_namespace_id=request.authority_namespace_id,
        experience_refs=request.experience_refs,
        evidence=tuple(registered[ref_id] for ref_id in ordered),
    )
    packet.validate()
    planes = tuple((ref_id, tuple(hits_by_ref.get(ref_id, ()))) for ref_id in ordered)
    return AssembledFormationContext(
        packet, tuple(opened), planes,
        tuple(ref_id for ref_id in ordered if ref_id in closure_ids),
    )
