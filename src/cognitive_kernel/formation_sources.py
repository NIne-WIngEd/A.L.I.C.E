"""A.L.I.C.E.-owned source registration boundary for MFM context assembly.

The registry, custody reader and current permission policy are supplied by the
memory fabric. A search result is only a reference. The MFM cannot register its
own speaker, subject, permission, origin or content digest by writing text.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Protocol

from .canonical import CognitiveKernelContractError, canonical_sha256, require_identifier
from .contracts import ProductHostScope
from .formation_context_planner import AssembledFormationContext
from .formation_contracts import FormationEvidenceRef


@dataclass(frozen=True)
class RegisteredFormationSource:
    evidence: FormationEvidenceRef
    object_ref: str
    registration_sha256: str

    @classmethod
    def create(cls, *, evidence: FormationEvidenceRef, object_ref: str) -> "RegisteredFormationSource":
        evidence.validate()
        require_identifier(object_ref, "object_ref")
        material = {"evidence": evidence.metadata_record(), "object_ref": object_ref}
        return cls(evidence, object_ref, canonical_sha256(material))

    def validate(self) -> None:
        expected = self.create(evidence=self.evidence, object_ref=self.object_ref)
        if expected.registration_sha256 != self.registration_sha256:
            raise CognitiveKernelContractError("formation source registration changed")


class FormationRegistry(Protocol):
    def lookup(self, ref_id: str) -> RegisteredFormationSource | None: ...


class FormationCustody(Protocol):
    def read(self, scope: ProductHostScope, object_ref: str) -> bytes: ...


class RegisteredFormationStore:
    """Adapter to the selected fabric's registry, private objects and policy."""

    def __init__(
        self, *, scope: ProductHostScope, authority_namespace_id: str,
        registry: FormationRegistry, custody: FormationCustody,
        permits: Callable[[RegisteredFormationSource, str], bool],
        purpose: str = "memory_formation",
    ) -> None:
        scope.validate()
        require_identifier(authority_namespace_id, "authority_namespace_id")
        self.scope = scope
        self.authority_namespace_id = authority_namespace_id
        self.registry = registry
        self.custody = custody
        self.permits = permits
        self.purpose = require_identifier(purpose, "purpose")

    def _lookup(self, ref_id: str) -> RegisteredFormationSource | None:
        source = self.registry.lookup(ref_id)
        if source is None:
            return None
        source.validate()
        ref = source.evidence
        if (ref.ref_id != ref_id or ref.scope != self.scope
                or ref.authority_namespace_id != self.authority_namespace_id):
            raise CognitiveKernelContractError("formation source registration crosses scope")
        return source

    def resolve(self, ref_id: str) -> FormationEvidenceRef | None:
        source = self._lookup(ref_id)
        return source.evidence if source is not None else None

    def permits_formation(self, ref: FormationEvidenceRef) -> bool:
        source = self._lookup(ref.ref_id)
        return source is not None and source.evidence == ref and self.permits(source, self.purpose)

    def read(self, ref: FormationEvidenceRef) -> bytes:
        source = self._lookup(ref.ref_id)
        if source is None or source.evidence != ref or not self.permits(source, self.purpose):
            raise CognitiveKernelContractError("source changed or permission revoked before read")
        content = self.custody.read(self.scope, source.object_ref)
        # Recheck policy and registration after I/O. A request that races a
        # revocation must not return the opened content to a model.
        if (not isinstance(content, bytes) or sha256(content).hexdigest() != ref.content_digest
                or self._lookup(ref.ref_id) != source or not self.permits(source, self.purpose)):
            raise CognitiveKernelContractError("source changed, revoked or failed content verification")
        return content


@dataclass(frozen=True)
class FormationReadReceipt:
    """Evidence delivered to the model input interface; no attention claim."""

    context_digest: str
    opened: tuple[tuple[str, str, str, tuple[str, ...]], ...]
    receipt_sha256: str


def record_formation_read(
    assembled: AssembledFormationContext, store: RegisteredFormationStore,
) -> FormationReadReceipt:
    if assembled.packet.scope != store.scope:
        raise CognitiveKernelContractError("formation read receipt crosses host scope")
    if len(assembled.packet.evidence) != len(assembled.opened_content):
        raise CognitiveKernelContractError("read receipt is missing opened source content")
    planes = dict(assembled.selected_planes)
    opened: list[tuple[str, str, str, tuple[str, ...]]] = []
    for ref, (ref_id, content) in zip(assembled.packet.evidence, assembled.opened_content):
        if ref.ref_id != ref_id:
            raise CognitiveKernelContractError("read receipt source order changed")
        source = store._lookup(ref_id)
        if (source is None or source.evidence != ref
                or not store.permits_formation(ref)
                or sha256(content).hexdigest() != ref.content_digest):
            raise CognitiveKernelContractError("read receipt source changed")
        opened.append((ref_id, ref.content_digest, source.registration_sha256, planes[ref_id]))
    items = tuple(opened)
    digest = assembled.packet.content_digest()
    return FormationReadReceipt(
        context_digest=digest, opened=items,
        receipt_sha256=canonical_sha256({"context_digest": digest, "opened": items}),
    )
