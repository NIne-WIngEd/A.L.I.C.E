from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class ProductManifestError(ValueError):
    """Raised when the product-family architecture is internally inconsistent."""


@dataclass(frozen=True)
class ProductIdentity:
    product_id: str
    display_name: str
    kind: str
    kernel_consumer: bool
    personal_state_sharing: str
    raw_host_data_vendor_access: bool | None = None
    mandatory_vendor_account: bool | None = None
    offline_core_required: bool | None = None
    name_status: str | None = None
    product_codename: str | None = None
    public_product_brand_status: str | None = None
    host_selects_assistant_name: bool | None = None
    full_capability_parity_with_alice: bool | None = None
    permanent_consumer_capability_ceiling: bool | None = None

    @classmethod
    def from_mapping(cls, product_id: str, payload: Mapping[str, Any]) -> "ProductIdentity":
        required = ("display_name", "kind", "kernel_consumer", "personal_state_sharing")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ProductManifestError(
                f"product {product_id!r} is missing required fields: {', '.join(missing)}"
            )
        sharing = str(payload["personal_state_sharing"])
        if sharing != "prohibited":
            raise ProductManifestError(
                f"product {product_id!r} must prohibit personal-state sharing"
            )
        optional_bool = lambda key: bool(payload[key]) if key in payload else None
        return cls(
            product_id=product_id,
            display_name=str(payload["display_name"]),
            kind=str(payload["kind"]),
            kernel_consumer=bool(payload["kernel_consumer"]),
            personal_state_sharing=sharing,
            raw_host_data_vendor_access=optional_bool("raw_host_data_vendor_access"),
            mandatory_vendor_account=optional_bool("mandatory_vendor_account"),
            offline_core_required=optional_bool("offline_core_required"),
            name_status=(str(payload["name_status"]) if payload.get("name_status") else None),
            product_codename=(str(payload["product_codename"]) if payload.get("product_codename") else None),
            public_product_brand_status=(
                str(payload["public_product_brand_status"])
                if payload.get("public_product_brand_status") else None
            ),
            host_selects_assistant_name=optional_bool("host_selects_assistant_name"),
            full_capability_parity_with_alice=optional_bool("full_capability_parity_with_alice"),
            permanent_consumer_capability_ceiling=optional_bool("permanent_consumer_capability_ceiling"),
        )


@dataclass(frozen=True)
class HostInstance:
    product_id: str
    host_instance_id: str
    schema_version: str
    encryption_domain: str

    def storage_scope(self) -> str:
        return f"{self.product_id}/{self.host_instance_id}/{self.encryption_domain}"

    def assert_isolated_from(self, other: "HostInstance") -> None:
        if self.storage_scope() == other.storage_scope():
            raise ProductManifestError("two host instances resolve to the same storage scope")


@dataclass(frozen=True)
class HostSelectedIdentity:
    host_instance_id: str
    assistant_name: str
    product_codename: str = "friday"
    public_product_brand: str | None = None

    def validate(self) -> None:
        name = self.assistant_name.strip()
        if not name:
            raise ProductManifestError("host-selected assistant name may not be empty")
        if len(name) > 128:
            raise ProductManifestError("host-selected assistant name exceeds 128 characters")
        if self.product_codename.strip().lower() != "friday":
            raise ProductManifestError("consumer development codename must remain friday")


@dataclass(frozen=True)
class ProductFamilyManifest:
    version: str
    shared_kernel_id: str
    shared_kernel_starts_at_phase: str
    formal_repository_split_gate: str
    shared_kernel_may_contain_personal_data: bool
    products: Mapping[str, ProductIdentity]
    separation_rules: Mapping[str, bool]
    parity_policy: Mapping[str, Any]

    def product(self, product_id: str) -> ProductIdentity:
        try:
            return self.products[product_id]
        except KeyError as exc:
            raise ProductManifestError(f"unknown product id: {product_id}") from exc

    def validate(self) -> None:
        if self.shared_kernel_may_contain_personal_data:
            raise ProductManifestError("shared kernel distribution may not contain personal data")
        if self.shared_kernel_starts_at_phase != "5.0":
            raise ProductManifestError("shared-kernel extraction must start at Phase 5.0")
        if self.formal_repository_split_gate != "6.5":
            raise ProductManifestError("formal product split must occur at the Phase 6.5 gate")
        required_products = {"alice", "friday"}
        missing = required_products.difference(self.products)
        if missing:
            raise ProductManifestError(f"missing product identities: {sorted(missing)}")

        alice = self.product("alice")
        friday = self.product("friday")
        if alice.kind == friday.kind:
            raise ProductManifestError("A.L.I.C.E. and Friday must remain different product kinds")
        if friday.raw_host_data_vendor_access is not False:
            raise ProductManifestError("consumer product must prohibit vendor access to raw host data")
        if friday.mandatory_vendor_account is not False:
            raise ProductManifestError("consumer local core may not require a vendor account")
        if friday.offline_core_required is not True:
            raise ProductManifestError("consumer product must provide offline core functionality")
        if friday.product_codename != "friday":
            raise ProductManifestError("Friday must be represented as an internal product codename")
        if friday.public_product_brand_status != "to_be_selected":
            raise ProductManifestError("public product brand must remain separate and unselected")
        if friday.host_selects_assistant_name is not True:
            raise ProductManifestError("each host must select the assistant name")
        if friday.full_capability_parity_with_alice is not True:
            raise ProductManifestError("consumer product must share A.L.I.C.E.'s capability destination")
        if friday.permanent_consumer_capability_ceiling is not False:
            raise ProductManifestError("consumer product may not have a permanent capability ceiling")

        required_false = (
            "alice_personal_data_may_seed_friday",
            "friday_host_data_may_enter_shared_kernel_distribution",
            "cross_host_cache_without_scope_allowed",
            "product_brand_may_define_storage_schema",
            "friday_may_be_permanently_reduced_edition",
            "host_selected_name_may_define_storage_schema",
        )
        for key in required_false:
            if self.separation_rules.get(key) is not False:
                raise ProductManifestError(f"separation rule {key!r} must be false")
        required_true = (
            "phase_1_to_4_files_are_migratable",
            "all_generalizable_alice_capabilities_enter_parity_backlog",
            "product_and_assistant_names_must_be_separate",
            "difference_is_state_maturity_hardware_and_permissions_not_destination_capability",
        )
        for key in required_true:
            if self.separation_rules.get(key) is not True:
                raise ProductManifestError(f"separation rule {key!r} must be true")
        if self.parity_policy.get("consumer_destination_matches_alice") is not True:
            raise ProductManifestError("parity policy must match consumer destination to A.L.I.C.E.")
        if self.parity_policy.get("ledger_required") is not True:
            raise ProductManifestError("capability parity ledger is required")


def load_product_family_manifest(path: str | Path | None = None) -> ProductFamilyManifest:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "policies" / "product_lines.json"
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    kernel = payload.get("shared_kernel", {})
    products_payload = payload.get("products", {})
    products = {
        product_id: ProductIdentity.from_mapping(product_id, product_payload)
        for product_id, product_payload in products_payload.items()
    }
    manifest = ProductFamilyManifest(
        version=str(payload.get("version", "")),
        shared_kernel_id=str(kernel.get("id", "")),
        shared_kernel_starts_at_phase=str(kernel.get("starts_at_phase", "")),
        formal_repository_split_gate=str(kernel.get("formal_repository_split_gate", "")),
        shared_kernel_may_contain_personal_data=bool(kernel.get("may_contain_personal_data", True)),
        products=products,
        separation_rules={str(key): bool(value) for key, value in payload.get("separation_rules", {}).items()},
        parity_policy=payload.get("parity_policy", {}),
    )
    manifest.validate()
    return manifest
