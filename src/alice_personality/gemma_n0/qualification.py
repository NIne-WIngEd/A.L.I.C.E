"""Verify a separately reviewed, complete public personality-N0 role report.

There is deliberately no report writer or qualification issuer. The current
forward diagnostic and FewRel experiment do not satisfy this protocol.

TRUST BOUNDARY: hashes, reviewer names, timestamps and JSON assertions are not
signatures or scientific evidence by themselves. ``expected_file_sha256`` must
come from a trusted independent owner-review decision, not be calculated from
an unreviewed candidate by its author. This verifier checks consistency and
measured comparisons; it cannot establish reviewer identity, honest execution,
label correctness or genuine historical precommitment from self-attestation.
Passing does not authorize private gradients or qualify Alice's identity.
Preparation verification rechecks publisher custody; run it on admitted
compute, never as a substitute for the caller's private execution boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from src.alice_foundation import gemma4_v1 as foundation


SCHEMA = "alice-personality-n0-role-qualification-v1"
PROTOCOL_SCHEMA = "alice-personality-n0-role-protocol-v1"
OUTCOME_SCHEMA = "alice-personality-n0-role-outcome-v1"
SOURCE_SCHEMA = "alice-personality-n0-public-target-review-v1"
RATIFICATION_SCHEMA = "alice-personality-n0-protocol-ratification-v1"
OWNER_REVIEW_SCHEMA = "alice-personality-n0-owner-role-review-v1"
RECIPE_SCHEMA = "alice-personality-n0-role-evaluation-recipe-v1"
RUN_SCHEMA = "alice-personality-n0-role-evaluation-run-v1"
TRAINING_SCHEMA = "alice-personality-n0-public-readout-training-evidence-v1"
OUTPUT_SCHEMA = "alice-personality-n0-public-role-observation-output-v1"
# Versioned coverage instruments, not a permanent ontology or numeric ceiling.
ROLE_CELLS = {
    "meaning_argument_roles": ("negation_conditional", "quantification_modality", "role_direction_reference",
                               "temporal_causal_composition", "paraphrase_overlap"),
    "evidence_uncertainty_plurality": ("fact_inference_unknown", "correction_historical_scope", "support_grounding",
                                      "decisive_irrelevant_removal", "conflict_plurality_calibration"),
    "pragmatics_social_context": ("implicature_presupposition", "indirect_act_ellipsis", "literal_intended_irony",
                                  "relationship_privacy_permissions", "belief_intent_emotion", "repair_disagreement_register"),
    "generic_voice_expression": ("prosody_meaning", "teasing_hostility_stakes", "uncertainty_delivery",
                                  "emphasis_pause_words_delivery", "turn_taking_repair", "uncertain_acoustic_cues"),
    "clone_source_host_self_separation": ("source_self_history", "direct_inference_synthetic", "host_not_identity_anchor",
                                           "relationship_missingness", "continuity_not_retroactive"),
    "inherited_preference_causality": ("relevant_rule_flip", "irrelevant_helpful_flattering", "agreement_pressure",
                                       "opposite_default_preference", "order_id_ablation", "co_valid_unknown"),
    "full_frame_grounding_scale": ("all_layer_codec", "source_span_graph_binding", "missing_fields_views",
                                    "raw_structured_conflict", "dynamic_cardinality", "context_relevance"),
}
CHECKS = {"baseline_review", "shortcut_review", "contamination_review", "target_independence_review",
          "source_family_split_review", "coverage_review"}
SOURCE_CLASSES = {"natural_human_annotations", "independently_reviewed_public_oracle", "independently_adjudicated_service_teacher"}
MAX_METADATA_BYTES = 16 * 1024 * 1024  # One parser admission limit, not corpus/capability limit.
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z\Z")


class QualificationError(ValueError):
    """The report cannot support its claimed public N0 role qualification."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise QualificationError(reason)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _text(value: object) -> str:
    _require(type(value) is str and bool(value.strip()) and value == value.strip()
             and not any(ord(char) < 32 for char in value), "review identifiers must be explicit nonempty text")
    return value


def _digest(value: object) -> str:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None, "an exact lowercase SHA256 binding is required")
    return value


def _exact(value: object, keys: set[str]) -> dict:
    _require(type(value) is dict and set(value) == keys, "unknown, missing or unsupported protocol fields")
    return value


def _integer(value: object, *, positive: bool = True) -> int:
    _require(type(value) is int and value >= (1 if positive else 0), "counts must be actual bounded nonnegative integers")
    return value


def _number(value: object) -> float:
    import math
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, "scientific values must be finite actual numbers")
    return value


def _time(value: object) -> datetime:
    _require(type(value) is str and _TIME.fullmatch(value) is not None, "timestamps must be explicit UTC instants")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise QualificationError("invalid UTC timestamp") from None


def _identifiers(value: object) -> set[str]:
    _require(type(value) is list and bool(value), "explicit review/source/case identities are required")
    items = [_text(item) for item in value]
    _require(len(set(items)) == len(items), "review/source/case identities must be unique")
    return set(items)


def _regular(value: str | Path) -> Path:
    _require(isinstance(value, (str, Path)), "report bindings require an absolute file path")
    path = Path(value)
    _require(path.is_absolute() and not any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction())
                for item in (path, *path.parents)), "report paths must not traverse links")
    _require(path.is_file() and path == path.resolve(strict=True), "report paths must be exact absolute regular files")
    return path


def _read(path: Path, expected: str, size: int | None = None) -> dict:
    _digest(expected)
    before = path.stat()
    _require(0 < before.st_size <= MAX_METADATA_BYTES and (size is None or before.st_size == size),
             "report metadata size differs or exceeds parser admission")
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA_BYTES + 1)
    after = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    _require(identity(before) == identity(after) and len(data) == before.st_size
             and sha256(data).hexdigest() == expected, "report bytes changed or differ from the external binding")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate report JSON keys")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(QualificationError("nonfinite JSON is forbidden")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, QualificationError):
            raise
        raise QualificationError("invalid report JSON") from None
    _require(type(value) is dict, "report metadata must be a JSON object")
    supplied = _digest(value.get("receipt_sha256"))
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    try:
        digest = sha256(_canonical(unsigned)).hexdigest()
    except (ValueError, TypeError, OverflowError):
        raise QualificationError("report values cannot contain nonfinite or unsupported JSON") from None
    _require(digest == supplied, "report canonical receipt digest differs")
    return value


def _pin_path(value: object, root: Path, seen: dict[Path, str]) -> tuple[Path, str, int]:
    pin = _exact(value, {"path", "bytes", "sha256"})
    raw = _text(pin["path"])
    relative = PurePosixPath(raw)
    _require(not relative.is_absolute() and "\\" not in raw and ":" not in raw
             and all(part not in ("", ".", "..") for part in raw.split("/")), "sidecar paths must be bounded relative members")
    path = _regular(root.joinpath(*relative.parts))
    _require(root in path.parents, "sidecar left its qualification report bundle")
    digest = _digest(pin["sha256"])
    _require(path not in seen, "one artifact cannot impersonate multiple independent protocol roles")
    seen[path] = digest
    return path, digest, _integer(pin["bytes"])


def _pin(value: object, root: Path, seen: dict[Path, str]) -> tuple[dict, str]:
    path, digest, size = _pin_path(value, root, seen)
    return _read(path, digest, size), digest


def _weight_bytes(path: Path, digest: str, size: int) -> None:
    """Hash an exact-size opaque state artifact; never deserialize its tensors."""
    before = path.stat()
    _require(before.st_size == size, "evaluated readout state size differs from its exact pin")
    observed, remaining = sha256(), size
    with path.open("rb") as stream:
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            _require(bool(data), "evaluated readout state ended before its exact byte bound")
            observed.update(data)
            remaining -= len(data)
        _require(not stream.read(1), "evaluated readout state exceeded its exact byte bound")
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
             and observed.hexdigest() == digest, "evaluated readout state changed or differs from its exact pin")


def current_qualification_code_sha256() -> dict:
    """Current provider, producer and identity contract code; no model imports."""
    root = Path(__file__).resolve().parents[1]
    paths = [path for name in ("gemma_n0", "identity") for path in sorted((root / name).glob("*.py"))]
    paths.append(root.parent / "alice_foundation/gemma4_v1.py")
    return {str(path.relative_to(root.parent)).replace("\\", "/"): sha256(_regular(path).read_bytes()).hexdigest() for path in paths}


def _bound(value: dict, prepared_file: str, prepared_receipt: str, source_inventory: str, code: dict) -> None:
    _require(value["prepared_receipt_file_sha256"] == prepared_file
             and value["prepared_receipt_sha256"] == prepared_receipt
             and value["source_inventory_sha256"] == source_inventory
             and value["code_sha256"] == code, "protocol source/preparation/current code bindings differ")
    _require(value["fixture_only"] is False and value["private_identity_data"] is False
             and value["final_payload_opened"] is False, "fixtures, private identity or opened FINAL cannot qualify public N0")


def verify_personality_n0_qualification(path: str | Path, *, expected_file_sha256: str,
        expected_prepared_file_sha256: str, expected_prepared_receipt_sha256: str) -> dict:
    """Read-only fail-closed verification, dependent on independent hash authority.

    No numeric scientific threshold is defined here. Thresholds and coverage
    counts must be explicitly ratified in the separately bound protocol before
    measurement. Sidecars stay inside the report directory; the preparation
    path is separately externally bound and custody-verified. No source case
    text or feature tensor is opened, printed or persisted by this function.
    """
    prepared_file, prepared_digest = (_digest(expected_prepared_file_sha256), _digest(expected_prepared_receipt_sha256))
    report_path = _regular(path)
    report = _read(report_path, expected_file_sha256)
    _exact(report, {"schema", "state", "role", "repository", "revision", "prepared_receipt_path",
        "prepared_receipt_file_sha256", "prepared_receipt_sha256", "protocol", "outcome", "ratification", "owner_review",
        "private_gradient_authorized", "identity_qualified", "receipt_sha256"})
    _require((report["schema"], report["state"], report["role"], report["repository"], report["revision"]) ==
        (SCHEMA, "QUALIFIED_PERSONALITY_N0", "personality", foundation.MODEL, foundation.REVISION),
        "only exact full public personality N0 role qualification reports are supported")
    _require(report["private_gradient_authorized"] is False and report["identity_qualified"] is False,
             "N0 role qualification cannot grant private gradients or Alice identity qualification")
    _require(report["prepared_receipt_file_sha256"] == prepared_file and report["prepared_receipt_sha256"] == prepared_digest,
             "qualification differs from the externally pinned prepared source")
    root, seen = report_path.parent, {report_path: expected_file_sha256}
    protocol, protocol_file = _pin(report["protocol"], root, seen)
    outcome, outcome_file = _pin(report["outcome"], root, seen)
    ratification, ratification_file = _pin(report["ratification"], root, seen)
    review, _ = _pin(report["owner_review"], root, seen)
    bound = {"prepared_receipt_file_sha256", "prepared_receipt_sha256", "source_inventory_sha256", "code_sha256",
             "fixture_only", "private_identity_data", "final_payload_opened", "receipt_sha256"}
    _exact(protocol, bound | {"schema", "state", "protocol_id", "author_ids", "created_at", "sources", "families", "checks", "evaluation_recipe"})
    _exact(outcome, bound | {"schema", "state", "protocol_file_sha256", "ratification_file_sha256", "started_at", "completed_at",
                           "families", "observed_model_geometry", "observed_runtime", "upstream_gradient", "upstream_tensor_mutation", "evaluation_runs"})
    _exact(ratification, {"schema", "state", "protocol_file_sha256", "owner_review_id", "reviewer_ids", "reviewed_at",
        "reviewed_thresholds", "reviewed_targets", "reviewed_baselines", "reviewed_split_and_contamination", "receipt_sha256"})
    _exact(review, {"schema", "state", "protocol_file_sha256", "outcome_file_sha256", "ratification_file_sha256",
        "owner_review_id", "reviewer_ids", "reviewed_at", "approved_families", "unresolved_material_failures",
        "source_acceptance_authority", "private_gradient_authorized", "identity_qualified", "receipt_sha256"})
    _require(protocol["schema"] == PROTOCOL_SCHEMA and protocol["state"] == "PRECOMMITTED_PUBLIC_ROLE_PROTOCOL"
             and outcome["schema"] == OUTCOME_SCHEMA and outcome["state"] == "PASSED_PUBLIC_ROLE_EVALUATION",
             "forward, FewRel-only or unpassed experiments are not full-role evidence")
    _text(protocol["protocol_id"])
    authors = _identifiers(protocol["author_ids"])
    _require(ratification["schema"] == RATIFICATION_SCHEMA and ratification["state"] == "OWNER_RATIFIED_BEFORE_MEASUREMENT"
             and ratification["protocol_file_sha256"] == protocol_file, "a separately bound premeasurement protocol ratification is required")
    _text(ratification["owner_review_id"])
    _require(not authors & _identifiers(ratification["reviewer_ids"]), "protocol author self-review is not independent ratification")
    _require(all(ratification[name] is True for name in ("reviewed_thresholds", "reviewed_targets", "reviewed_baselines", "reviewed_split_and_contamination")),
             "credible targets, thresholds, baselines and split reviews must be explicit")
    _require(outcome["protocol_file_sha256"] == protocol_file and outcome["ratification_file_sha256"] == ratification_file,
             "measurements differ from their precommitted protocol")
    _require(_time(protocol["created_at"]) <= _time(ratification["reviewed_at"]) < _time(outcome["started_at"])
             <= _time(outcome["completed_at"]) <= _time(review["reviewed_at"]) <= datetime.now(timezone.utc),
             "protocol/review/measurement chronology does not show prior completed ratification")
    _exact(protocol["checks"], CHECKS)
    _require(all(value is True for value in protocol["checks"].values()), "baseline, shortcut, contamination, split and coverage reviews are incomplete")
    _require(outcome["upstream_gradient"] is False and outcome["upstream_tensor_mutation"] is False,
             "public role evidence must preserve the frozen publisher route")
    code = current_qualification_code_sha256()
    inventory = _digest(protocol["source_inventory_sha256"])
    for artifact in (protocol, outcome):
        _bound(artifact, prepared_file, prepared_digest, inventory, code)
    _require(type(protocol["sources"]) is dict and bool(protocol["sources"]), "independently reviewed public target sources are absent")
    source_records, source_classes, source_authors = {}, {}, set(authors)
    global_families, global_case_hashes = {"TRAIN": set(), "DEV": set()}, {"TRAIN": set(), "DEV": set()}
    used_sources, source_pins = set(), {}
    for source_id, pin in protocol["sources"].items():
        _text(source_id)
        source, source_file = _pin(pin, root, seen)
        _exact(source, {"schema", "state", "source_id", "source_revision", "source_class", "task_scope", "author_ids", "reviewer_ids",
            "license_reviewed", "targets_independently_adjudicated", "source_family_derivative_closure_reviewed",
            "fixture_only", "private_identity_data", "final_payload_opened", "records", "receipt_sha256"})
        _require(source["schema"] == SOURCE_SCHEMA and source["state"] == "REVIEWED_PUBLIC_TARGET_SOURCE"
                 and source["source_id"] == source_id and type(source["source_class"]) is str and source["source_class"] in SOURCE_CLASSES,
                 "unreviewed, diagnostic or fixture target sources cannot qualify N0")
        _text(source["source_revision"])
        _require(type(source["task_scope"]) is str and source["task_scope"] in {"natural_relation_matching_only", "reviewed_full_public_role_cases"}, "unsupported target source task scope")
        if source_id == "thunlp_fewrel_1_0":
            _require(source["task_scope"] == "natural_relation_matching_only", "FewRel labels cannot be reclassified as full-role targets")
        source_author = _identifiers(source["author_ids"])
        source_authors |= source_author
        _require(not source_author & _identifiers(source["reviewer_ids"]), "target authors cannot self-attest independent gold review")
        _require(all(source[name] is True for name in ("license_reviewed", "targets_independently_adjudicated", "source_family_derivative_closure_reviewed"))
                 and all(source[name] is False for name in ("fixture_only", "private_identity_data", "final_payload_opened")),
                 "target/source rights, independence or heldout lineage checks are incomplete")
        _require(type(source["records"]) is list and bool(source["records"]), "source-bound target case records are absent")
        families, ids, split_hashes = {"TRAIN": set(), "DEV": set()}, set(), {"TRAIN": set(), "DEV": set()}
        for record in source["records"]:
            _exact(record, {"case_id", "family_id", "split", "case_sha256", "target_sha256"})
            case_id, family = _text(record["case_id"]), _text(record["family_id"])
            _require(type(record["split"]) is str and record["split"] in families and case_id not in ids, "source cases must be unique and exclusively TRAIN/DEV")
            ids.add(case_id)
            families[record["split"]].add(family)
            split_hashes[record["split"]].add(_digest(record["case_sha256"]))
            _digest(record["target_sha256"])
            source_records[(source_id, case_id)] = record
        _require(not families["TRAIN"] & families["DEV"] and not split_hashes["TRAIN"] & split_hashes["DEV"],
                 "target-source family/case leakage crosses heldout splits")
        for split in global_families:
            global_families[split] |= families[split]
            global_case_hashes[split] |= split_hashes[split]
        source_classes[source_id] = (source["source_class"], source["task_scope"])
        source_pins[source_id] = source_file
    _require(not global_families["TRAIN"] & global_families["DEV"]
             and not global_case_hashes["TRAIN"] & global_case_hashes["DEV"],
             "heldout source families/cases leak across separately named sources")
    _require(sha256(_canonical(source_pins)).hexdigest() == inventory, "public target source inventory binding differs")
    recipe, recipe_file = _pin(protocol["evaluation_recipe"], root, seen)
    _exact(recipe, bound | {"schema", "state", "recipe_id", "created_at", "readout_configurations", "procedure"})
    _require(recipe["schema"] == RECIPE_SCHEMA and recipe["state"] == "PRECOMMITTED_PUBLIC_ROLE_EVALUATION_RECIPE",
             "a separately pinned precommitted evaluated-readout recipe is required")
    _bound(recipe, prepared_file, prepared_digest, inventory, code)
    _text(recipe["recipe_id"])
    _text(recipe["procedure"])
    _require(_time(recipe["created_at"]) <= _time(protocol["created_at"]), "evaluation recipe was not precommitted with the protocol")
    configurations = recipe["readout_configurations"]
    _require(type(configurations) is dict and bool(configurations), "evaluated readout configurations are absent")
    for readout_id, configuration in configurations.items():
        _text(readout_id)
        _exact(configuration, {"configuration", "configuration_sha256", "weight_format"})
        _require(type(configuration["configuration"]) is dict and bool(configuration["configuration"]), "evaluated readout configuration must be explicit")
        _require(sha256(_canonical(configuration["configuration"])).hexdigest() == _digest(configuration["configuration_sha256"]),
                 "evaluated readout configuration digest differs")
        _require(type(configuration["weight_format"]) is str and configuration["weight_format"] in
                 {"safetensors", "torch_weights_only_state_dict"}, "unreviewed readout state formats are unsupported")
    _require(type(outcome["evaluation_runs"]) is dict and bool(outcome["evaluation_runs"]), "exact evaluated-run receipts are absent")
    runs, weight_sizes, used_runs, used_observations = {}, {}, set(), set()
    producer_code = {name: code["alice_personality/identity/" + name]
                     for name in ("codec.py", "feature_producer.py", "contracts.py", "__init__.py")}
    run_fields = bound | {"schema", "state", "run_id", "recipe_file_sha256", "protocol_file_sha256", "ratification_file_sha256",
        "started_at", "completed_at", "observed_model_geometry", "observed_runtime", "upstream_gradient", "upstream_tensor_mutation",
        "learned_readout_states", "training_receipt", "feature_production_receipts", "observations", "output_receipt"}
    training_fields = bound | {"schema", "state", "recipe_file_sha256", "run_id", "observed_updates", "cases",
        "learned_readout_state_sha256", "upstream_gradient", "upstream_tensor_mutation"}
    session_fields = {"schema", "source_class", "prepared_receipt_file_sha256", "prepared_receipt_sha256", "frame_binding_sha256",
        "implementation_sha256", "status", "session_source_recheck_passed", "mechanical_fixture_only", "qualification",
        "source_acceptance_authority", "persistent_feature_values_retained", "receipt_sha256"}
    for run_id, pin in outcome["evaluation_runs"].items():
        _text(run_id)
        run, _ = _pin(pin, root, seen)
        _exact(run, run_fields)
        _require(run["schema"] == RUN_SCHEMA and run["state"] == "COMPLETE_PUBLIC_ROLE_EVALUATION" and run["run_id"] == run_id,
                 "incomplete or differently identified evaluation runs cannot qualify")
        _bound(run, prepared_file, prepared_digest, inventory, code)
        _require(run["recipe_file_sha256"] == recipe_file and run["protocol_file_sha256"] == protocol_file
                 and run["ratification_file_sha256"] == ratification_file,
                 "evaluated run does not bind the actual ratified protocol and recipe")
        _require(_time(outcome["started_at"]) <= _time(run["started_at"]) <= _time(run["completed_at"])
                 <= _time(outcome["completed_at"]), "evaluated run falls outside its reported measurement interval")
        _require(run["upstream_gradient"] is False and run["upstream_tensor_mutation"] is False,
                 "evaluated runs must preserve the frozen publisher route")
        _require(_canonical(run["observed_model_geometry"]) == _canonical(outcome["observed_model_geometry"])
                 and _canonical(run["observed_runtime"]) == _canonical(outcome["observed_runtime"]),
                 "evaluated run and outcome geometry/runtime differ")
        _exact(run["learned_readout_states"], set(configurations))
        states = {}
        for readout_id, state_pin in run["learned_readout_states"].items():
            state_path, state_digest, state_size = _pin_path(state_pin, root, seen)
            _weight_bytes(state_path, state_digest, state_size)
            weight_sizes[state_path] = state_size
            states[readout_id] = state_digest
        training, _ = _pin(run["training_receipt"], root, seen)
        _exact(training, training_fields)
        _bound(training, prepared_file, prepared_digest, inventory, code)
        _require(training["schema"] == TRAINING_SCHEMA and training["state"] == "OPTIMIZED_PUBLIC_READOUT_UNQUALIFIED"
                 and training["run_id"] == run_id and training["recipe_file_sha256"] == recipe_file
                 and training["learned_readout_state_sha256"] == states
                 and training["upstream_gradient"] is False and training["upstream_tensor_mutation"] is False,
                 "evaluated learned states do not bind their exact frozen-source training receipt")
        _integer(training["observed_updates"])
        _require(type(training["cases"]) is list and bool(training["cases"]), "readout training source cases are absent")
        train_keys = set()
        for item in training["cases"]:
            _exact(item, {"source_id", "case_id", "case_sha256", "target_sha256"})
            key = (_text(item["source_id"]), _text(item["case_id"]))
            _require(key in source_records and key not in train_keys and source_records[key]["split"] == "TRAIN"
                     and source_records[key]["case_sha256"] == item["case_sha256"]
                     and source_records[key]["target_sha256"] == item["target_sha256"],
                     "evaluated readout training must bind reviewed TRAIN targets without heldout leakage")
            train_keys.add(key)
        sessions = run["feature_production_receipts"]
        _require(type(sessions) is list and bool(sessions), "evaluated run lacks closed full-frame source-production receipts")
        produced_frames = set()
        for session_pin in sessions:
            session, _ = _pin(session_pin, root, seen)
            _exact(session, session_fields)
            _require(session["schema"] == "alice-personality-feature-session-v1" and session["status"] == "CLOSED_SOURCE_VERIFIED"
                     and session["source_class"] == "prepared_frozen_gemma_features_unqualified"
                     and session["prepared_receipt_file_sha256"] == prepared_file and session["prepared_receipt_sha256"] == prepared_digest
                     and session["implementation_sha256"] == producer_code
                     and session["session_source_recheck_passed"] is True and session["mechanical_fixture_only"] is False
                     and session["qualification"] == "UNQUALIFIED" and session["source_acceptance_authority"] is False
                     and session["persistent_feature_values_retained"] is False,
                     "evaluated frame production must close against exact prepared source and current producer code")
            members = _identifiers(session["frame_binding_sha256"])
            for member in members:
                _digest(member)
            produced_frames |= members
        observations = run["observations"]
        _require(type(observations) is dict and bool(observations), "source-bound measured observations are absent")
        outputs, _ = _pin(run["output_receipt"], root, seen)
        _exact(outputs, bound | {"schema", "state", "run_id", "recipe_file_sha256", "observations"})
        _bound(outputs, prepared_file, prepared_digest, inventory, code)
        _require(outputs["schema"] == OUTPUT_SCHEMA and outputs["state"] == "RECORDED_UNQUALIFIED_PUBLIC_OUTPUTS"
                 and outputs["run_id"] == run_id and outputs["recipe_file_sha256"] == recipe_file,
                 "measured outputs do not bind the exact evaluated run and recipe")
        _exact(outputs["observations"], set(observations))
        for observation_id, observation in observations.items():
            _text(observation_id)
            _exact(observation, {"family", "cell", "cases", "metrics", "passed", "frame_binding_sha256", "output_sha256"})
            _digest(observation["output_sha256"])
            values = outputs["observations"][observation_id]
            _exact(values, {"cases", "output_values"})
            _require(type(values["output_values"]) is dict and bool(values["output_values"])
                     and _canonical(values["cases"]) == _canonical(observation["cases"])
                     and sha256(_canonical(values["output_values"])).hexdigest() == observation["output_sha256"],
                     "observed case/output bytes differ from their separately pinned output receipt")
            family, cell = _text(observation["family"]), _text(observation["cell"])
            _require(family in ROLE_CELLS and cell in ROLE_CELLS[family], "an evaluated observation uses an unsupported role instrument")
            frames = _identifiers(observation["frame_binding_sha256"])
            _require(frames <= produced_frames, "measured observations are absent from closed feature-production inventory")
            for frame in frames:
                _digest(frame)
        runs[run_id] = run
    _exact(protocol["families"], set(ROLE_CELLS))
    _exact(outcome["families"], set(ROLE_CELLS))
    for family, cells in ROLE_CELLS.items():
        expected, observed = protocol["families"][family], outcome["families"][family]
        _exact(expected, set(cells))
        _exact(observed, set(cells))
        for cell in cells:
            spec, measured = expected[cell], observed[cell]
            _exact(spec, {"source_ids", "minimum_cases", "metrics", "scientific_basis"})
            _text(spec["scientific_basis"])
            sources = _identifiers(spec["source_ids"])
            _require(sources <= set(source_classes), "coverage refers to an unreviewed target source")
            _require(type(spec["metrics"]) is dict and bool(spec["metrics"]), "credible precommitted scientific thresholds are absent")
            _exact(measured, {"cases", "metrics", "passed", "run_id", "observation_id"})
            run_id, observation_id = _text(measured["run_id"]), _text(measured["observation_id"])
            _require(run_id in runs and observation_id in runs[run_id]["observations"], "outcome references an absent exact evaluated observation")
            observation = runs[run_id]["observations"][observation_id]
            _require(observation["family"] == family and observation["cell"] == cell
                     and all(_canonical(observation[name]) == _canonical(measured[name]) for name in ("cases", "metrics", "passed")),
                     "reported metrics/cases differ from the source-bound evaluated-run receipt")
            used_runs.add(run_id)
            used_observations.add((run_id, observation_id))
            _require(measured["passed"] is True and type(measured["cases"]) is list
                     and len(measured["cases"]) >= _integer(spec["minimum_cases"]), "unpassed or insufficient coverage observations")
            case_keys = set()
            for item in measured["cases"]:
                _exact(item, {"source_id", "case_id", "case_sha256", "target_sha256"})
                key = (_text(item["source_id"]), _text(item["case_id"]))
                _require(key not in case_keys and key in source_records and key[0] in sources,
                         "observed cases do not match their reviewed target source")
                case_keys.add(key)
                source = source_records[key]
                _require(source["split"] == "DEV" and source["case_sha256"] == item["case_sha256"]
                         and source["target_sha256"] == item["target_sha256"], "qualification measurements must use exact source-bound heldout targets")
                if source_classes[key[0]][1] == "natural_relation_matching_only":
                    _require(family == "meaning_argument_roles" and cell == "role_direction_reference", "FewRel-only relation evidence cannot cover other public role gates")
                used_sources.add(key[0])
            _exact(measured["metrics"], set(spec["metrics"]))
            for metric, criterion in spec["metrics"].items():
                _text(metric)
                _exact(criterion, {"comparison", "threshold", "unit", "ratified_basis"})
                _text(criterion["unit"])
                _text(criterion["ratified_basis"])
                value, threshold = _number(measured["metrics"][metric]), _number(criterion["threshold"])
                _require(type(criterion["comparison"]) is str and criterion["comparison"] in {">=", "<="}, "unsupported measured threshold comparison")
                _require(value >= threshold if criterion["comparison"] == ">=" else value <= threshold,
                         "a measured role criterion did not reach its precommitted threshold")
    _require(used_sources == set(source_classes) and any(source_classes[name][0] == "natural_human_annotations" for name in used_sources),
             "qualification needs used natural public annotation plus role-complete reviewed targets")
    _require(used_runs == set(runs) and used_observations == {(run_id, observation_id) for run_id, run in runs.items()
             for observation_id in run["observations"]}, "evaluation runs cannot hide unreported role observations")
    _require(review["schema"] == OWNER_REVIEW_SCHEMA and review["state"] == "OWNER_APPROVED_PUBLIC_N0_ROLE"
             and review["protocol_file_sha256"] == protocol_file and review["outcome_file_sha256"] == outcome_file
             and review["ratification_file_sha256"] == ratification_file, "separate owner review does not bind the actual protocol and measured outcome")
    _text(review["owner_review_id"])
    _require(not source_authors & _identifiers(review["reviewer_ids"]), "authors cannot self-approve full N0 role qualification")
    _require(_identifiers(review["approved_families"]) == set(ROLE_CELLS) and review["unresolved_material_failures"] == []
             and type(review["unresolved_material_failures"]) is list
             and all(review[name] is False for name in ("source_acceptance_authority", "private_gradient_authorized", "identity_qualified")),
             "owner role review is incomplete or exceeds public N0 authority")
    prepared_path = _regular(report["prepared_receipt_path"])
    _read(prepared_path, prepared_file)
    from .preparation import verify_prepared
    prepared = verify_prepared(prepared_path)
    _require(prepared["receipt_sha256"] == prepared_digest and prepared["repository"] == foundation.MODEL
             and prepared["revision"] == foundation.REVISION and prepared["role"] == "personality"
             and prepared["state"] == "PREPARED_UNQUALIFIED" and prepared["behavior_qualification"] is None,
             "qualification source differs from the actual verified personality preparation")
    _require(type(outcome["observed_model_geometry"]) is dict and type(outcome["observed_runtime"]) is dict
             and _canonical(outcome["observed_model_geometry"]) == _canonical(prepared["model_geometry"])
             and _canonical(outcome["observed_runtime"]) == _canonical(prepared["runtime"]),
             "measured outcome geometry/runtime differ from the actual prepared route")
    _read(prepared_path, prepared_file)
    for artifact, digest in seen.items():
        if artifact in weight_sizes:
            _weight_bytes(_regular(artifact), digest, weight_sizes[artifact])
        else:
            _read(_regular(artifact), digest)
    _require(current_qualification_code_sha256() == code, "qualification code changed during verification")
    return report
