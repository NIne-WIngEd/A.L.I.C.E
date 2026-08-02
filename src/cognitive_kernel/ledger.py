"""Deterministic receipts and integrity records for the experience ledger."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)

LEDGER_SCHEMA_VERSION = "1.0.0"


def ledger_scope_record(
    *,
    product_id: str,
    host_instance_id: str,
    encryption_domain: str,
) -> dict[str, str]:
    return {
        "product_id": require_identifier(product_id, "product_id"),
        "host_instance_id": require_identifier(
            host_instance_id, "host_instance_id"
        ),
        "encryption_domain": require_identifier(
            encryption_domain, "encryption_domain"
        ),
    }


def ledger_scope_digest(
    *,
    product_id: str,
    host_instance_id: str,
    encryption_domain: str,
) -> str:
    return canonical_sha256(
        ledger_scope_record(
            product_id=product_id,
            host_instance_id=host_instance_id,
            encryption_domain=encryption_domain,
        )
    )


@dataclass(frozen=True)
class ExperienceLedgerEntryReceipt:
    """Receipt for one immutable event row in a ledger transaction."""

    sequence: int
    event_id: str
    event_sha256: str
    previous_entry_sha256: str
    entry_sha256: str
    committed_at: str

    @classmethod
    def create(
        cls,
        *,
        sequence: object,
        event_id: object,
        event_sha256: object,
        previous_entry_sha256: object,
        committed_at: object,
    ) -> "ExperienceLedgerEntryReceipt":
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise CognitiveKernelContractError(
                "ledger sequence must be an integer"
            )
        if sequence < 1:
            raise CognitiveKernelContractError(
                "ledger sequence must be positive"
            )
        normalized_event_id = require_identifier(event_id, "event_id")
        normalized_event_sha256 = require_sha256(
            event_sha256, "event_sha256"
        )
        normalized_previous = require_sha256(
            previous_entry_sha256, "previous_entry_sha256"
        )
        normalized_committed_at = normalize_timestamp(
            committed_at, "committed_at"
        )
        material = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": sequence,
            "event_id": normalized_event_id,
            "event_sha256": normalized_event_sha256,
            "previous_entry_sha256": normalized_previous,
            "committed_at": normalized_committed_at,
        }
        return cls(
            sequence=sequence,
            event_id=normalized_event_id,
            event_sha256=normalized_event_sha256,
            previous_entry_sha256=normalized_previous,
            entry_sha256=canonical_sha256(material),
            committed_at=normalized_committed_at,
        )

    def validate(self) -> None:
        expected = self.create(
            sequence=self.sequence,
            event_id=self.event_id,
            event_sha256=self.event_sha256,
            previous_entry_sha256=self.previous_entry_sha256,
            committed_at=self.committed_at,
        )
        if self.entry_sha256 != expected.entry_sha256:
            raise CognitiveKernelContractError(
                "experience ledger entry digest mismatch"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_sha256": self.event_sha256,
            "previous_entry_sha256": self.previous_entry_sha256,
            "entry_sha256": self.entry_sha256,
            "committed_at": self.committed_at,
        }


@dataclass(frozen=True)
class ExperienceLedgerTransactionReceipt:
    """Atomic commit receipt for one or more ledger entries."""

    schema_version: str
    transaction_id: str
    ledger_id: str
    committed_at: str
    entries: tuple[ExperienceLedgerEntryReceipt, ...]
    transaction_sha256: str

    @classmethod
    def create(
        cls,
        *,
        ledger_id: object,
        committed_at: object,
        entries: tuple[ExperienceLedgerEntryReceipt, ...],
        schema_version: object = LEDGER_SCHEMA_VERSION,
    ) -> "ExperienceLedgerTransactionReceipt":
        normalized_schema = require_schema_version(schema_version)
        if normalized_schema != LEDGER_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "experience ledger receipt schema version changed"
            )
        normalized_ledger_id = require_identifier(ledger_id, "ledger_id")
        normalized_committed_at = normalize_timestamp(
            committed_at, "committed_at"
        )
        if not isinstance(entries, tuple) or not entries:
            raise CognitiveKernelContractError(
                "ledger transaction must contain at least one entry"
            )
        previous_sequence: int | None = None
        for entry in entries:
            if not isinstance(entry, ExperienceLedgerEntryReceipt):
                raise CognitiveKernelContractError(
                    "ledger transaction entries must be entry receipts"
                )
            entry.validate()
            if entry.committed_at != normalized_committed_at:
                raise CognitiveKernelContractError(
                    "ledger transaction timestamps must match"
                )
            if previous_sequence is not None and entry.sequence != previous_sequence + 1:
                raise CognitiveKernelContractError(
                    "ledger transaction sequences must be contiguous"
                )
            previous_sequence = entry.sequence
        material = {
            "schema_version": normalized_schema,
            "ledger_id": normalized_ledger_id,
            "committed_at": normalized_committed_at,
            "entries": [entry.record() for entry in entries],
        }
        digest = canonical_sha256(material)
        receipt = cls(
            schema_version=normalized_schema,
            transaction_id=f"ledger-transaction-{digest[:32]}",
            ledger_id=normalized_ledger_id,
            committed_at=normalized_committed_at,
            entries=entries,
            transaction_sha256=digest,
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        if require_schema_version(self.schema_version) != LEDGER_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "experience ledger receipt schema version changed"
            )
        require_identifier(self.ledger_id, "ledger_id")
        normalized_time = normalize_timestamp(self.committed_at, "committed_at")
        if normalized_time != self.committed_at:
            raise CognitiveKernelContractError(
                "ledger transaction timestamp is not canonical"
            )
        if not self.entries:
            raise CognitiveKernelContractError(
                "ledger transaction must contain entries"
            )
        previous_sequence: int | None = None
        for entry in self.entries:
            entry.validate()
            if entry.committed_at != self.committed_at:
                raise CognitiveKernelContractError(
                    "ledger transaction timestamps must match"
                )
            if previous_sequence is not None and entry.sequence != previous_sequence + 1:
                raise CognitiveKernelContractError(
                    "ledger transaction sequences must be contiguous"
                )
            previous_sequence = entry.sequence
        material = {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "committed_at": self.committed_at,
            "entries": [entry.record() for entry in self.entries],
        }
        digest = canonical_sha256(material)
        require_sha256(self.transaction_sha256, "transaction_sha256")
        if self.transaction_sha256 != digest:
            raise CognitiveKernelContractError(
                "experience ledger transaction digest mismatch"
            )
        if self.transaction_id != f"ledger-transaction-{digest[:32]}":
            raise CognitiveKernelContractError(
                "experience ledger transaction identity mismatch"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "ledger_id": self.ledger_id,
            "committed_at": self.committed_at,
            "entries": [entry.record() for entry in self.entries],
            "transaction_sha256": self.transaction_sha256,
        }


@dataclass(frozen=True)
class ExperienceLedgerRecord:
    """Sanitized metadata view of one stored experience event."""

    sequence: int
    event_id: str
    event_type: str
    occurred_at: str
    content_digest: str
    retention_class: str
    storage_tier: str
    event_sha256: str
    previous_entry_sha256: str
    entry_sha256: str
    committed_at: str

    def record(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "content_digest": self.content_digest,
            "retention_class": self.retention_class,
            "storage_tier": self.storage_tier,
            "event_sha256": self.event_sha256,
            "previous_entry_sha256": self.previous_entry_sha256,
            "entry_sha256": self.entry_sha256,
            "committed_at": self.committed_at,
        }


@dataclass(frozen=True)
class ExperienceLedgerIntegrityReport:
    """Deterministic result of full ledger verification."""

    schema_version: str
    ledger_id: str
    entry_count: int
    first_sequence: int | None
    last_sequence: int | None
    head_entry_sha256: str
    valid: bool
    report_sha256: str

    @classmethod
    def create(
        cls,
        *,
        ledger_id: object,
        entry_count: object,
        first_sequence: object | None,
        last_sequence: object | None,
        head_entry_sha256: object,
        valid: object,
    ) -> "ExperienceLedgerIntegrityReport":
        normalized_ledger_id = require_identifier(ledger_id, "ledger_id")
        if isinstance(entry_count, bool) or not isinstance(entry_count, int):
            raise CognitiveKernelContractError(
                "entry_count must be an integer"
            )
        if entry_count < 0:
            raise CognitiveKernelContractError(
                "entry_count may not be negative"
            )
        if not isinstance(valid, bool):
            raise CognitiveKernelContractError("valid must be boolean")
        if entry_count == 0:
            if first_sequence is not None or last_sequence is not None:
                raise CognitiveKernelContractError(
                    "empty ledger report may not contain sequences"
                )
        else:
            if not isinstance(first_sequence, int) or not isinstance(
                last_sequence, int
            ):
                raise CognitiveKernelContractError(
                    "non-empty ledger report requires integer sequences"
                )
            if first_sequence != 1 or last_sequence != entry_count:
                raise CognitiveKernelContractError(
                    "ledger report sequences are not contiguous"
                )
        normalized_head = require_sha256(
            head_entry_sha256, "head_entry_sha256"
        )
        material = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "ledger_id": normalized_ledger_id,
            "entry_count": entry_count,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "head_entry_sha256": normalized_head,
            "valid": valid,
        }
        digest = canonical_sha256(material)
        report = cls(
            schema_version=LEDGER_SCHEMA_VERSION,
            ledger_id=normalized_ledger_id,
            entry_count=entry_count,
            first_sequence=first_sequence,
            last_sequence=last_sequence,
            head_entry_sha256=normalized_head,
            valid=valid,
            report_sha256=digest,
        )
        report.validate()
        return report

    def validate(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "experience ledger report schema version changed"
            )
        require_identifier(self.ledger_id, "ledger_id")
        if isinstance(self.entry_count, bool) or not isinstance(
            self.entry_count, int
        ):
            raise CognitiveKernelContractError(
                "entry_count must be an integer"
            )
        if self.entry_count < 0:
            raise CognitiveKernelContractError(
                "entry_count may not be negative"
            )
        if not isinstance(self.valid, bool):
            raise CognitiveKernelContractError("valid must be boolean")
        if self.entry_count == 0:
            if self.first_sequence is not None or self.last_sequence is not None:
                raise CognitiveKernelContractError(
                    "empty ledger report may not contain sequences"
                )
        else:
            if not isinstance(self.first_sequence, int) or not isinstance(
                self.last_sequence, int
            ):
                raise CognitiveKernelContractError(
                    "non-empty ledger report requires integer sequences"
                )
            if self.first_sequence != 1 or self.last_sequence != self.entry_count:
                raise CognitiveKernelContractError(
                    "ledger report sequences are not contiguous"
                )
        require_sha256(self.head_entry_sha256, "head_entry_sha256")
        material = {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "entry_count": self.entry_count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "head_entry_sha256": self.head_entry_sha256,
            "valid": self.valid,
        }
        digest = canonical_sha256(material)
        require_sha256(self.report_sha256, "report_sha256")
        if self.report_sha256 != digest:
            raise CognitiveKernelContractError(
                "experience ledger integrity report digest mismatch"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "entry_count": self.entry_count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "head_entry_sha256": self.head_entry_sha256,
            "valid": self.valid,
            "report_sha256": self.report_sha256,
        }
