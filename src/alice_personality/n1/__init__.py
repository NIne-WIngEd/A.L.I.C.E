"""Governed identity source compilation; this package grants no training authority."""

from .compiler import (
    IdentitySubstrateError, PackagePin, compile_package, load_package_pin,
    curated_frontier_v2_pin, verify_compiled,
)

__all__ = ["IdentitySubstrateError", "PackagePin", "compile_package", "load_package_pin",
           "curated_frontier_v2_pin", "verify_compiled"]
