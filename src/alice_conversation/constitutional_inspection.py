"""Metadata-safe inspection for P3.4 constitutional system contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .constitutional_prompt import ConstitutionalSystemContract


@dataclass(frozen=True)
class ConstitutionalSourceInspection:
    path: str
    version: str
    normalized_sha256: str


@dataclass(frozen=True)
class ConstitutionalContractInspection:
    version: str
    policy_version: str
    constitution_version: str
    content_sha256: str
    content_characters: int
    source_count: int
    sources: tuple[ConstitutionalSourceInspection, ...]
    contains_untrusted_grounding: bool
    contains_source_text: bool


def inspect_constitutional_system_contract(
    contract: ConstitutionalSystemContract,
) -> ConstitutionalContractInspection:
    """Return metadata without returning the contract or governance source text."""

    contract.validate()
    sources = tuple(
        ConstitutionalSourceInspection(
            path=source.path,
            version=source.version,
            normalized_sha256=source.normalized_sha256,
        )
        for source in contract.sources
    )
    return ConstitutionalContractInspection(
        version=contract.version,
        policy_version=contract.policy_version,
        constitution_version=contract.constitution_version,
        content_sha256=contract.content_sha256,
        content_characters=len(contract.content),
        source_count=len(sources),
        sources=sources,
        contains_untrusted_grounding=(
            "BEGIN UNTRUSTED GROUNDING DATA" in contract.content.splitlines()
            or "END UNTRUSTED GROUNDING DATA" in contract.content.splitlines()
        ),
        contains_source_text=False,
    )


def render_constitutional_contract_inspection(
    inspection: ConstitutionalContractInspection,
) -> str:
    """Render a deterministic metadata-only inspection report."""

    lines = [
        "A.L.I.C.E. constitutional contract inspection",
        f"version={inspection.version}",
        f"policy_version={inspection.policy_version}",
        f"constitution_version={inspection.constitution_version}",
        f"content_sha256={inspection.content_sha256}",
        f"content_characters={inspection.content_characters}",
        f"source_count={inspection.source_count}",
        f"contains_untrusted_grounding={str(inspection.contains_untrusted_grounding).lower()}",
        f"contains_source_text={str(inspection.contains_source_text).lower()}",
    ]
    for source in inspection.sources:
        lines.append(
            "source="
            f"{source.path}|version={source.version}|sha256={source.normalized_sha256}"
        )
    return "\n".join(lines) + "\n"
