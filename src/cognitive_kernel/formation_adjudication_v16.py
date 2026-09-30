"""Authenticate MFM 1.6 corpus review claims against an externally pinned roster.

This verifies signatures and exact byte bindings, not a person's identity,
independence, source rights, or the semantic correctness of their judgment.
The roster digest must come from a trusted data steward outside the corpus.
FINAL payloads and rights files remain sealed; only signed metadata is read.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .canonical import CognitiveKernelContractError, canonical_json_bytes, require_sha256
from .formation_dataset_admission import CorpusAdmission, admit_formation_corpus


ROSTER_SCHEMA = "mfm-v16-external-trust-roster-v1"
RECEIPT_SCHEMA = "mfm-v16-cryptographic-review-v1"
COMPLETE_SCOPE = (
    "all_opened_sources", "all_relevant_proposals", "all_dispositions",
    "no_other_relevant_memory_at_this_scope",
)
DOMAINS = {"rights": b"ALICE-MFM-V16-RIGHTS\n",
           "review": b"ALICE-MFM-V16-REVIEW\n",
           "final_rights": b"ALICE-MFM-V16-FINAL-RIGHTS\n"}


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"{name} must be an object")
    return value


def _read_bound(path: Path, expected: str, name: str) -> dict:
    if _digest(raw := path.read_bytes()) != require_sha256(expected, name):
        raise CognitiveKernelContractError(f"{name} bytes differ from pin")
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise CognitiveKernelContractError(f"{name} has duplicate JSON key")
            result[key] = value
        return result
    return _object(json.loads(raw, object_pairs_hook=unique), name)


def _decode(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise CognitiveKernelContractError(f"{name} must be base64")
    try:
        raw = b64decode(value, validate=True)
    except (ValueError, Base64Error) as exc:
        raise CognitiveKernelContractError(f"{name} is invalid base64") from exc
    if b64encode(raw).decode("ascii") != value:
        raise CognitiveKernelContractError(f"{name} is noncanonical base64")
    return raw


def _keys(roster: dict, name: str) -> dict[str, bytes]:
    entries = _object(roster.get(name), name)
    result = {identifier: _decode(encoded, f"{name} key")
              for identifier, encoded in entries.items()}
    if any(not isinstance(identifier, str) or not identifier or len(key) != 32
           for identifier, key in result.items()):
        raise CognitiveKernelContractError(f"{name} needs canonical Ed25519 keys")
    if len(set(result.values())) != len(result):
        raise CognitiveKernelContractError(f"{name} repeats a public key")
    return result


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CognitiveKernelContractError(f"{name} must be UTC")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CognitiveKernelContractError(f"{name} is invalid") from exc
    if result.tzinfo != timezone.utc:
        raise CognitiveKernelContractError(f"{name} must be UTC")
    return result


def _signed(row: object, *, public_key: bytes, domain: str, as_of: datetime) -> dict:
    value = _object(row, domain)
    signature = _decode(value.get("signature_v16"), f"{domain} signature")
    if len(signature) != 64 or domain not in DOMAINS:
        raise CognitiveKernelContractError(f"{domain} signature or domain is invalid")
    unsigned = {key: field for key, field in value.items() if key != "signature_v16"}
    if _timestamp(unsigned.get("valid_until"), f"{domain} valid_until") <= as_of:
        raise CognitiveKernelContractError(f"{domain} attestation expired")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise CognitiveKernelContractError("Ed25519 review verification dependency missing") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, DOMAINS[domain] + canonical_json_bytes(unsigned))
    except (InvalidSignature, ValueError) as exc:
        raise CognitiveKernelContractError(f"{domain} signature does not authenticate") from exc
    return unsigned


def verify_adjudicated_corpus_v16(
    admission: CorpusAdmission, manifest_path: str | Path,
    trust_roster_path: str | Path, *, expected_roster_sha256: str,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Recheck signed rights/review metadata before a shared MFM fit.

    Caller must independently authenticate the roster digest and reviewers'
    real identities, and a FINAL custodian must later check sealed bytes. A
    valid return is not a model qualification or an authorization to publish.
    """
    instant = as_of or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise CognitiveKernelContractError("admission as_of needs a timezone")
    instant = instant.astimezone(timezone.utc)
    manifest = _read_bound(Path(manifest_path), admission.manifest_sha256, "corpus manifest")
    # A caller-constructed dataclass is not an admission receipt. Re-run the
    # complete base rights, lineage, split and exact-byte verifier here.
    if admit_formation_corpus(manifest_path,
                              expected_sha256=admission.manifest_sha256) != admission:
        raise CognitiveKernelContractError("signed corpus differs from base admission")
    roster = _read_bound(Path(trust_roster_path), expected_roster_sha256, "trust roster")
    if manifest.get("schema") != "mfm-formation-corpus-v1" or \
            manifest.get("corpus_id") != admission.corpus_id:
        raise CognitiveKernelContractError("signed corpus manifest differs from admission")
    if roster.get("schema") != ROSTER_SCHEMA:
        raise CognitiveKernelContractError("unsupported external trust roster")
    if _timestamp(roster.get("valid_until"), "roster valid_until") <= instant:
        raise CognitiveKernelContractError("external trust roster expired")
    revoked = roster.get("revoked_ids")
    if not isinstance(revoked, list) or any(not isinstance(x, str) for x in revoked):
        raise CognitiveKernelContractError("invalid revoked identity list")
    authors, reviewers, issuers = (_keys(roster, field) for field in
                                   ("authors", "reviewers", "rights_issuers"))
    if not authors or len(reviewers) < 2 or not issuers:
        raise CognitiveKernelContractError("trust roster lacks independent roles")
    if len(set(revoked)) != len(revoked):
        raise CognitiveKernelContractError("duplicate revoked identity")
    expected_cases = {case.case_id: case for case in
                      (*admission.train, *admission.development, *admission.final_metadata)}
    rows = manifest.get("cases")
    if (not isinstance(rows, list) or len(rows) != len(expected_cases) or
            {row.get("case_id") for row in rows if isinstance(row, dict)} != set(expected_cases)):
        raise CognitiveKernelContractError("corpus cases differ from admitted roster")
    counts = {"train": 0, "development": 0, "final": 0}
    for row in rows:
        case = expected_cases[row["case_id"]]
        if row.get("split") != case.split:
            raise CognitiveKernelContractError("signed case changed split")
        author_id = row.get("author_id")
        if author_id not in authors or author_id in revoked:
            raise CognitiveKernelContractError("case author is not externally registered")
        source_rows = row["sources"]
        source_sha256s = [source["sha256"] for source in source_rows]
        rights_sha256s = [source["rights_sha256"] for source in source_rows]
        if source_sha256s != [source.sha256 for source in case.source_payloads]:
            raise CognitiveKernelContractError("signed case sources differ")
        rights_keys: set[bytes] = set()
        for source, ref in zip(source_rows, case.source_payloads, strict=True):
            if case.split != "final":
                rights = _read_bound(admission.root / source["rights_path"],
                                     source["rights_sha256"], "rights receipt")
                issuer_id = rights.get("issuer_id")
                if issuer_id not in issuers or issuer_id in revoked:
                    raise CognitiveKernelContractError("rights issuer is not trusted")
                _signed(rights, public_key=issuers[issuer_id], domain="rights", as_of=instant)
                rights_keys.add(issuers[issuer_id])
            else:
                attestation = _object(source.get("final_rights_attestation_v16"),
                                      "sealed FINAL rights attestation")
                issuer_id = attestation.get("issuer_id")
                if issuer_id not in issuers or issuer_id in revoked:
                    raise CognitiveKernelContractError("FINAL rights issuer is not trusted")
                signed = _signed(attestation, public_key=issuers[issuer_id],
                                 domain="final_rights", as_of=instant)
                if any(signed.get(k) != v for k, v in {
                    "case_id": case.case_id, "source_id": source["source_id"],
                    "source_sha256": ref.sha256,
                    "rights_sha256": source["rights_sha256"],
                    "host_family": row["host_family"],
                    "formation_evaluation": True,
                }.items()):
                    raise CognitiveKernelContractError("FINAL rights attestation changed source")
                rights_keys.add(issuers[issuer_id])
        seen_keys: set[bytes] = set()
        for review in row["reviews"]:
            reviewer_id = review.get("reviewer_id")
            if reviewer_id not in reviewers or reviewer_id in revoked:
                raise CognitiveKernelContractError("reviewer is not trusted")
            key = reviewers[reviewer_id]
            if key in seen_keys or key == authors[author_id] or key in rights_keys:
                raise CognitiveKernelContractError("review signatures are not independent keys")
            signed = _signed(review, public_key=key, domain="review", as_of=instant)
            expected = {"case_id": case.case_id, "split": case.split,
                        "author_id": author_id, "target_sha256": case.target_payload.sha256,
                        "source_sha256s": source_sha256s,
                        "source_rights_sha256s": rights_sha256s, "blind": True,
                        "decision": "accept", "complete_target_review_v16": True,
                        "coverage_scope": list(COMPLETE_SCOPE)}
            if any(signed.get(field) != value for field, value in expected.items()):
                raise CognitiveKernelContractError("signed review lacks complete exact target coverage")
            # A signed review must also bind scenario and generator lineage,
            # so a valid signature cannot be transplanted to a changed family.
            if signed.get("lineage_sha256") != _digest(canonical_json_bytes({
                    field: row[field] for field in (
                        "host_family", "source_family", "generator_family",
                        "scenario_family", "duplicate_group", "parent_case_ids")})):
                raise CognitiveKernelContractError("signed review changed case lineage")
            seen_keys.add(key)
        if len(seen_keys) < 2:
            raise CognitiveKernelContractError("case needs two authenticated reviews")
        counts[case.split] += 1
    return {"schema": RECEIPT_SCHEMA,
            "corpus_manifest_sha256": admission.manifest_sha256,
            "externally_pinned_roster_sha256": expected_roster_sha256,
            "case_counts": counts, "rights_and_review_signatures_verified": True,
            "final_payloads_opened": False,
            "external_identity_and_semantic_truth_verified": False,
            "qualified_model": False}
