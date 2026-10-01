"""Admission boundary for a host-neutral, independently labelled MFM corpus.

The public repository carries this verifier, never private histories. FINAL is
metadata-only here: its payloads are neither opened nor handed to an optimizer.
This validates receipts and declared reviewer identities; an external data
steward remains responsible for authenticating consent and reviewer identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .canonical import CognitiveKernelContractError, require_identifier, require_sha256


SCHEMA = "mfm-formation-corpus-v1"
TEACHER_SCHEMA = "mfm-v16-owner-teacher-corpus-v1"
RIGHTS_SCHEMA = "mfm-source-rights-v1"
SPLITS = frozenset({"train", "development", "final"})


def _identifier(value: object, name: str) -> str:
    result = require_identifier(value, name)
    if result != value:
        raise CognitiveKernelContractError(f"{name} must be canonical")
    return result


def _digest(value: object, name: str) -> str:
    result = require_sha256(value, name)
    if result != value:
        raise CognitiveKernelContractError(f"{name} must be canonical")
    return result


def _object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"{name} must be an object")
    return value


def _unique_json(raw: bytes, name: str) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise CognitiveKernelContractError(f"{name} has duplicate JSON key")
            result[key] = value
        return result
    try:
        return _object(json.loads(raw, object_pairs_hook=unique), name)
    except json.JSONDecodeError as exc:
        raise CognitiveKernelContractError(f"{name} is not one JSON object") from exc


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise CognitiveKernelContractError(f"{name} must be an array")
    return value


def _path(root: Path, value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CognitiveKernelContractError(f"{name} needs a relative POSIX path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise CognitiveKernelContractError(f"{name} escapes the corpus root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise CognitiveKernelContractError(f"{name} escapes the corpus root")
    return value, path


def _checked_bytes(root: Path, ref: dict, name: str, *, open_payload: bool) -> tuple[str, str]:
    path, absolute = _path(root, ref.get("path"), name)
    digest = _digest(ref.get("sha256"), f"{name}.sha256")
    if open_payload:
        try:
            actual = sha256(absolute.read_bytes()).hexdigest()
        except OSError as exc:
            raise CognitiveKernelContractError(f"{name} cannot be read") from exc
        if actual != digest:
            raise CognitiveKernelContractError(f"{name} bytes differ from manifest")
    return path, digest


@dataclass(frozen=True)
class PayloadRef:
    path: str
    sha256: str


@dataclass(frozen=True)
class AdmittedCase:
    case_id: str
    split: str
    source_payloads: tuple[PayloadRef, ...]
    target_payload: PayloadRef
    rights_receipts: tuple[PayloadRef, ...]


@dataclass(frozen=True)
class CorpusAdmission:
    corpus_id: str
    manifest_sha256: str
    train: tuple[AdmittedCase, ...]
    development: tuple[AdmittedCase, ...]
    final_metadata: tuple[AdmittedCase, ...]
    root: Path

    def audit_handoff(self, *, gradient_paths: tuple[str, ...],
                      development_paths: tuple[str, ...]) -> dict[str, object]:
        """Recheck bytes at handoff; reject every unlisted or FINAL path/digest.

        Development may select a checkpoint but may never supply a gradient.
        A caller must audit its actual file list, not a reassuring split flag.
        """
        def refs(cases: tuple[AdmittedCase, ...]) -> dict[str, str]:
            result: dict[str, str] = {}
            for case in cases:
                for item in (*case.source_payloads, case.target_payload):
                    previous = result.setdefault(item.path, item.sha256)
                    if previous != item.sha256:
                        raise CognitiveKernelContractError("path has conflicting digests")
            return result

        training = refs(self.train)
        development = refs(self.development)
        final = refs(self.final_metadata)
        forbidden_digests = set(final.values())
        if set(training) & set(development) or set(training) & set(final) or set(development) & set(final):
            raise CognitiveKernelContractError("corpus payload path crosses splits")
        if (set(training.values()) | set(development.values())) & forbidden_digests:
            raise CognitiveKernelContractError("FINAL payload digest crosses splits")
        # Rights receipts stay outside optimizer inputs, but their frozen bytes
        # must still be present when the dataset is handed off.
        for case in (*self.train, *self.development):
            for receipt in case.rights_receipts:
                _checked_bytes(self.root, {"path": receipt.path,
                                           "sha256": receipt.sha256},
                               "source rights receipt", open_payload=True)
        def audit(paths: tuple[str, ...], allowed: dict[str, str], role: str) -> list[dict[str, str]]:
            if len(paths) != len(set(paths)):
                raise CognitiveKernelContractError(f"duplicate {role} payload")
            rows = []
            for path in paths:
                name, absolute = _path(self.root, path, role)
                if name not in allowed or name in final:
                    raise CognitiveKernelContractError(f"{role} contains unadmitted or FINAL payload")
                if allowed[name] in forbidden_digests:
                    raise CognitiveKernelContractError(f"{role} contains FINAL payload digest")
                try:
                    actual = sha256(absolute.read_bytes()).hexdigest()
                except OSError as exc:
                    raise CognitiveKernelContractError(f"{role} payload cannot be read") from exc
                if actual != allowed[name]:
                    raise CognitiveKernelContractError(f"{role} payload changed")
                rows.append({"path": name, "sha256": actual})
            return rows
        return {"schema": "mfm-optimizer-inputs-v1", "corpus_id": self.corpus_id,
                "corpus_manifest_sha256": self.manifest_sha256,
                "gradient_inputs": audit(gradient_paths, training, "gradient"),
                "development_inputs": audit(development_paths, development, "development"),
                "final_payloads_excluded": True,
                "final_exclusion_paths": sorted(final),
                "final_exclusion_digests": sorted(forbidden_digests)}


def admit_formation_corpus(manifest_path: str | Path, *,
                           expected_sha256: str, teacher_training: bool = False,
                           owner_authorization_ref: str | None = None) -> CorpusAdmission:
    """Admit exact bytes; the opt-in teacher lane is training-only, not gold.

    The default signed-review corpus policy is unchanged. Teacher training
    uses the same source rights and connected-lineage checks but no invented
    independent reviews or FINAL. Owner authorization is an attestation, not
    cryptographic proof of consent or semantic correctness.
    """
    file = Path(manifest_path)
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise CognitiveKernelContractError("corpus manifest cannot be read") from exc
    digest = sha256(raw).hexdigest()
    if digest != _digest(expected_sha256, "expected_sha256"):
        raise CognitiveKernelContractError("corpus manifest differs from frozen digest")
    manifest = _unique_json(raw, "corpus manifest")
    if manifest.get("schema") != (TEACHER_SCHEMA if teacher_training else SCHEMA):
        raise CognitiveKernelContractError("unsupported formation corpus schema")
    if teacher_training:
        authorization = _identifier(owner_authorization_ref, "owner_authorization_ref")
        if (manifest.get("authorization_id") != authorization or
                manifest.get("status") != "owner-authorized-teacher-training-only-unqualified" or
                type(manifest.get("case_count")) is not int or manifest["case_count"] < 1):
            raise CognitiveKernelContractError("teacher corpus lacks pinned owner authorization or count")
    elif owner_authorization_ref is not None:
        raise CognitiveKernelContractError("signed corpus does not use teacher authorization")
    corpus_id = _identifier(manifest.get("corpus_id"), "corpus_id")
    root = file.parent.resolve()
    cases = _list(manifest.get("cases"), "cases")
    if not cases:
        raise CognitiveKernelContractError("corpus has no cases")
    if teacher_training and len(cases) != manifest["case_count"]:
        raise CognitiveKernelContractError("teacher corpus count differs from pinned manifest")
    seen_cases: set[str] = set()
    case_groups: dict[str, set[tuple[str, str]]] = {}
    case_parents: dict[str, tuple[str, ...]] = {}
    source_owners: dict[str, tuple[str, str, str]] = {}
    source_lineage: dict[str, tuple[str, ...]] = {}
    source_parents: list[tuple[str, str]] = []
    memberships: dict[tuple[str, str], set[str]] = {}
    admitted: dict[str, list[AdmittedCase]] = {split: [] for split in SPLITS}
    for row_value in cases:
        row = _object(row_value, "case")
        case_id = _identifier(row.get("case_id"), "case_id")
        if case_id in seen_cases:
            raise CognitiveKernelContractError("duplicate case ID")
        seen_cases.add(case_id)
        split = row.get("split")
        if split not in (SPLITS - {"final"} if teacher_training else SPLITS):
            raise CognitiveKernelContractError("unsupported corpus split")
        if teacher_training and (row.get("authorization_id") != authorization or
                not isinstance(row.get("target_origin"), str) or
                row["target_origin"] not in {
                    "owner-authorized-service-teacher",
                    "licensed-deterministic-generator"}):
            raise CognitiveKernelContractError("teacher target lacks origin and owner authorization")
        author = _identifier(row.get("author_id"), "author_id")
        groups: set[tuple[str, str]] = set()
        for field in ("host_family", "source_family", "generator_family",
                      "scenario_family", "duplicate_group"):
            groups.add((field, _identifier(row.get(field), field)))
        parents = tuple(_identifier(x, "parent_case_id") for x in
                        _list(row.get("parent_case_ids"), "parent_case_ids"))
        if len(set(parents)) != len(parents) or case_id in parents:
            raise CognitiveKernelContractError("invalid parent case references")
        case_groups[case_id] = groups
        case_parents[case_id] = parents
        source_refs = _list(row.get("sources"), "sources")
        if not source_refs:
            raise CognitiveKernelContractError("case has no source")
        source_payloads: list[PayloadRef] = []
        source_digests: list[str] = []
        rights_receipts: list[PayloadRef] = []
        local_source_ids: set[str] = set()
        for source_value in source_refs:
            source = _object(source_value, "source")
            source_id = _identifier(source.get("source_id"), "source_id")
            if source_id in local_source_ids:
                raise CognitiveKernelContractError("duplicate source ID in case")
            local_source_ids.add(source_id)
            path, source_digest = _checked_bytes(root, source, "source", open_payload=split != "final")
            if source_id in source_owners and source_owners[source_id][:2] != (path, source_digest):
                raise CognitiveKernelContractError("source ID has conflicting bytes")
            source_owners[source_id] = (path, source_digest, case_id)
            groups.add(("source_id", source_id))
            groups.add(("source_sha256", source_digest))
            parents_of_source = tuple(_identifier(parent, "parent_source_id") for parent in
                                      _list(source.get("parent_source_ids"), "parent_source_ids"))
            if len(set(parents_of_source)) != len(parents_of_source):
                raise CognitiveKernelContractError("duplicate source parent")
            if source_id in source_lineage and source_lineage[source_id] != parents_of_source:
                raise CognitiveKernelContractError("source ID has conflicting ancestry")
            source_lineage[source_id] = parents_of_source
            for parent_id in parents_of_source:
                if parent_id == source_id:
                    raise CognitiveKernelContractError("source is its own parent")
                source_parents.append((source_id, parent_id))
            if split != "final":
                receipt_path, receipt_digest = _checked_bytes(
                    root, {"path": source.get("rights_path"),
                           "sha256": source.get("rights_sha256")},
                    "source rights receipt", open_payload=True)
                rights = _unique_json((root / receipt_path).read_bytes(),
                                      "source rights receipt")
                if rights.get("schema") != RIGHTS_SCHEMA:
                    raise CognitiveKernelContractError("unsupported source rights receipt")
                for field, value in (("source_id", source_id),
                                     ("source_sha256", source_digest),
                                     ("host_family", row["host_family"])):
                    if rights.get(field) != value:
                        raise CognitiveKernelContractError("source rights receipt has wrong binding")
                _identifier(rights.get("issuer_id"), "rights issuer_id")
                _identifier(rights.get("authority_ref"), "rights authority_ref")
                if rights.get("revoked") is not False:
                    raise CognitiveKernelContractError("source authorization is revoked or unknown")
                permitted = (rights.get("formation_training") is True and
                             rights.get("model_distribution") is True) if split == "train" else (
                             rights.get("formation_evaluation") is True)
                if not permitted:
                    raise CognitiveKernelContractError("source rights do not permit this split")
                groups.add(("rights_receipt_sha256", receipt_digest))
                rights_receipts.append(PayloadRef(receipt_path, receipt_digest))
            else:
                _digest(source.get("rights_sha256"), "final rights_sha256")
                _path(root, source.get("rights_path"), "final rights_path")
            source_payloads.append(PayloadRef(path, source_digest))
            source_digests.append(source_digest)
        target_row = _object(row.get("target"), "target")
        target_path, target_digest = _checked_bytes(
            root, target_row, "target", open_payload=split != "final")
        if teacher_training:
            provenance = _object(row.get("target_provenance"), "target_provenance")
            if set(provenance) != {"producer_id", "producer_version", "input",
                                   "output", "conversion"} or \
                    _identifier(provenance.get("producer_id"), "producer_id") != author:
                raise CognitiveKernelContractError("teacher target producer differs from author")
            _identifier(provenance.get("producer_version"), "producer_version")
            if row["target_origin"] == "owner-authorized-service-teacher":
                # Do not let differently named generator families place one
                # service teacher's output on both sides of diagnostic split.
                # Steward authentication is still required for an asserted ID.
                groups.add(("service_teacher_producer_id", author))
            for field in ("input", "output", "conversion"):
                _checked_bytes(root, _object(provenance[field], field),
                               f"target provenance {field}", open_payload=True)
            source_paths = {_path(root, source.path, "source")[1]
                            for source in source_payloads}
            target_absolute = _path(root, target_path, "target")[1]
            provenance_paths = {field: _path(root, provenance[field]["path"],
                                             f"target provenance {field}")[1]
                                for field in ("input", "output", "conversion")}
            if row["target_origin"] == "owner-authorized-service-teacher":
                if (len(set(provenance_paths.values())) != 3 or
                        set(provenance_paths.values()) & (source_paths | {target_absolute})):
                    raise CognitiveKernelContractError(
                        "teacher provenance paths must be distinct from each other, sources and target")
            else:
                if (provenance["output"]["path"] != target_path or
                        provenance["output"]["sha256"] != target_digest):
                    raise CognitiveKernelContractError(
                        "deterministic generator output must bind exact target bytes")
                if (provenance_paths["input"] == provenance_paths["conversion"] or
                        {provenance_paths["input"], provenance_paths["conversion"]} &
                        (source_paths | {target_absolute})):
                    raise CognitiveKernelContractError(
                        "deterministic provenance input and conversion paths must be distinct")
            groups.add(("target_provenance_input_sha256", provenance["input"]["sha256"]))
            groups.add(("target_provenance_output_sha256", provenance["output"]["sha256"]))
        reviews = _list(row.get("reviews"), "reviews")
        reviewers: set[str] = set()
        for review_value in reviews:
            review = _object(review_value, "review")
            reviewer = _identifier(review.get("reviewer_id"), "reviewer_id")
            if reviewer == author or reviewer in reviewers:
                raise CognitiveKernelContractError("reviewers must be independent of author and one another")
            if (review.get("blind") is not True or review.get("decision") != "accept" or
                    review.get("target_sha256") != target_digest or
                    review.get("source_sha256s") != source_digests):
                raise CognitiveKernelContractError("reviewers did not independently accept exact target and sources")
            reviewers.add(reviewer)
        if teacher_training and reviews:
            raise CognitiveKernelContractError("teacher training must not claim independent review")
        if not teacher_training and len(reviewers) < 2:
            raise CognitiveKernelContractError("case requires two independent reviewers")
        admitted[split].append(AdmittedCase(
            case_id, split, tuple(source_payloads), PayloadRef(target_path, target_digest),
            tuple(rights_receipts)))
    # Group equivalence includes identities, raw and derivative sources,
    # duplication, scenarios, generator and explicit parent-case lineage.
    for case_id, groups in case_groups.items():
        for key in groups:
            memberships.setdefault(key, set()).add(case_id)
    for case_id, parents in case_parents.items():
        for parent in parents:
            if parent not in seen_cases:
                raise CognitiveKernelContractError("unregistered parent case")
            memberships.setdefault(("case_parent", parent), set()).update((case_id, parent))
    for child, parent in source_parents:
        if parent not in source_owners:
            raise CognitiveKernelContractError("unregistered parent source")
        memberships.setdefault(("source_parent", parent), set()).update((
            source_owners[child][2], source_owners[parent][2]))
    def reject_cycles(graph: dict[str, tuple[str, ...]]) -> None:
        active: set[str] = set()
        complete: set[str] = set()
        for start in graph:
            stack = [(start, False)]
            while stack:
                node, closing = stack.pop()
                if closing:
                    active.remove(node)
                    complete.add(node)
                elif node not in complete:
                    if node in active:
                        raise CognitiveKernelContractError("corpus parent lineage cycle")
                    active.add(node)
                    stack.append((node, True))
                    stack.extend((parent, False) for parent in reversed(graph[node]))
    reject_cycles(case_parents)
    reject_cycles(source_lineage)
    split_by_case = {case.case_id: case.split for part in admitted.values() for case in part}
    for members in memberships.values():
        if len({split_by_case[case_id] for case_id in members}) > 1:
            raise CognitiveKernelContractError("connected source/person/generator lineage leaks across splits")
    result = CorpusAdmission(corpus_id, digest, tuple(admitted["train"]),
                             tuple(admitted["development"]), tuple(admitted["final"]), root)
    if teacher_training:
        if not result.train or not result.development or result.final_metadata:
            raise CognitiveKernelContractError("teacher corpus needs train and diagnostic development only")
    elif not result.train or not result.development or not result.final_metadata:
        raise CognitiveKernelContractError("corpus requires train, development and sealed FINAL metadata")
    # Metadata-level exclusion happens even if nobody requests a handoff.
    result.audit_handoff(gradient_paths=(), development_paths=())
    return result
