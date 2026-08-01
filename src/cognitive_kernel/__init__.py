"""Host-neutral Personal Cognitive Kernel contract foundation."""

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_sha256,
)
from .contracts import (
    IDENTITY_LAYERS,
    PRIVATE_DIRECTIVE_CODES,
    PRODUCT_IDS,
    PROVENANCE_TYPES,
    RETENTION_CLASSES,
    STORAGE_TIERS,
    OpaquePrivateCompanionReference,
    ProductHostScope,
    ProvenanceReference,
)
from .experience import ExperienceEvent
from .policy import (
    CognitiveKernelFoundationPolicy,
    load_cognitive_kernel_foundation_policy,
)

__version__ = "0.1.0"

__all__ = [
    "CognitiveKernelContractError",
    "CognitiveKernelFoundationPolicy",
    "ExperienceEvent",
    "IDENTITY_LAYERS",
    "OpaquePrivateCompanionReference",
    "PRIVATE_DIRECTIVE_CODES",
    "PRODUCT_IDS",
    "PROVENANCE_TYPES",
    "ProductHostScope",
    "ProvenanceReference",
    "RETENTION_CLASSES",
    "STORAGE_TIERS",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_cognitive_kernel_foundation_policy",
    "normalize_timestamp",
    "require_sha256",
]
