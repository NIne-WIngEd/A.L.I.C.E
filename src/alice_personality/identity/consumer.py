"""Actual optimized EIPM judgments to words and portable expressive intent.

No generator or renderer receives authority to replace the judgment. Evidence
is development construction, never release qualification or owner approval.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from hashlib import sha256
import math
import re

import torch

from .codec import canonical, fingerprint
from .contracts import IdentityError
from .feature_producer import ProducedFrame


@dataclass(frozen=True)
class CheckpointPin:
    path: str
    manifest_sha256: str


@dataclass(frozen=True)
class RenderedResponse:
    text: str
    delivery: dict
    renderer_id: str
    artifact_sha256: str | None


class DecisionConsumer:
    def __init__(self, model, checkpoints: tuple[CheckpointPin, ...]):
        if type(checkpoints) is not tuple or not checkpoints or any(type(p) is not CheckpointPin for p in checkpoints):
            raise IdentityError("response consumer needs exact optimized checkpoint pins")
        self.model, self.checkpoints = model, checkpoints
        self._evidence()

    def _evidence(self):
        from ..teaching.trainer import CHECKPOINT_SCHEMA, _hash, _json, _path, current_code_hashes
        previous, fields, labels, manifests, source_classes = None, set(), {}, [], set()
        previous_phase = None
        prepared_source = None
        for pin in self.checkpoints:
            path = _path(pin.path, directory=True)
            if {p.name for p in path.iterdir()} != {"manifest.json", "resume.pt"}:
                raise IdentityError("optimized checkpoint membership changed")
            manifest = _json(_path(path / "manifest.json"))
            body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
            if manifest.get("manifest_sha256") != pin.manifest_sha256 or sha256(canonical(body)).hexdigest() != pin.manifest_sha256:
                raise IdentityError("optimized checkpoint manifest changed")
            if (manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("training_state") != "OPTIMIZED_UNQUALIFIED"
                    or manifest.get("qualification") != "UNQUALIFIED" or manifest.get("acceptance_authority") is not False
                    or type(manifest.get("step")) is not int or manifest["step"] < 1
                    or len(manifest.get("events", [])) != manifest["step"]
                    or manifest.get("binding", {}).get("code_sha256") != current_code_hashes()):
                raise IdentityError("actual optimized construction evidence is required")
            phase = manifest.get("phase")
            source_pin = manifest["binding"]["artifacts"]["prepared_receipt"]
            if (manifest["binding"].get("model_config_sha256") != sha256(canonical(asdict(self.model.config))).hexdigest()
                    or (prepared_source is not None and source_pin["file_sha256"] != prepared_source)):
                raise IdentityError("construction checkpoint model configuration or prepared source changed")
            prepared_source = source_pin["file_sha256"]
            order = {"n1": 0, "n2": 1, "calibration": 2}
            if phase not in order or (previous_phase is not None and order[phase] <= order[previous_phase]):
                raise IdentityError("construction checkpoints must preserve governing stage order")
            if previous is not None and manifest["binding"]["model_initial_state_sha256"] != previous:
                raise IdentityError("construction checkpoints do not form the actual model lineage")
            payload = _path(path / "resume.pt")
            if _hash(payload) != manifest.get("resume_sha256"):
                raise IdentityError("optimized checkpoint payload changed")
            state = torch.load(payload, map_location="cpu", weights_only=True)
            if fingerprint(state["model"]) != manifest.get("model_state_sha256"):
                raise IdentityError("optimized checkpoint model state binding changed")
            previous, previous_phase = manifest["model_state_sha256"], phase
            for index, event in enumerate(manifest["events"], 1):
                if (event.get("step") != index or event.get("phase") != phase
                        or event.get("acceptance_authority") is not False or event.get("qualification") != "UNQUALIFIED"
                        or event.get("admission_binding_sha256") != sha256(canonical(manifest["binding"])).hexdigest()):
                    raise IdentityError("checkpoint optimization events differ from their admitted construction")
                source_classes.add(event["source_class"])
                if phase == "n2":
                    fields.update(name for name, count in event["reviewed_target_coverage"].items() if count > 0)
                    for name, ids in event.get("reviewed_label_coverage", {}).items():
                        labels.setdefault(name, set()).update(ids)
            manifests.append(pin.manifest_sha256)
        if fingerprint(self.model.state_dict()) != previous:
            raise IdentityError("actual model state differs from optimized checkpoint")
        return {"model_state_sha256": previous, "checkpoint_manifest_sha256": manifests,
                "prepared_pin": manifest["binding"]["artifacts"]["prepared_receipt"],
                "reviewed_fields": fields, "reviewed_labels": labels, "source_classes": sorted(source_classes)}

    def respond(self, document, produced: ProducedFrame, closed_session, *, renderer=None, tie_tolerance=1e-6):
        from ..teaching.materialize import _document_frame
        document.validate()
        if type(produced) is not ProducedFrame:
            raise IdentityError("response needs the actual produced document features")
        for value, key in ((produced.binding, "binding_sha256"), (closed_session, "receipt_sha256")):
            if sha256(canonical({k: v for k, v in value.items() if k != key})).hexdigest() != value.get(key):
                raise IdentityError("response feature/session binding changed")
        if (produced.binding.get("document_sha256") != document.sha256
                or produced.binding.get("frame_sha256") != fingerprint(produced.frame)
                or produced.binding.get("binding_sha256") not in closed_session.get("frame_binding_sha256", [])):
            raise IdentityError("response document/features differ from the closed session")
        _document_frame(document, produced.frame)
        evidence = self._evidence()
        from .feature_store import verify_store_custody
        verify_store_custody(produced.binding, closed_session)
        from ..teaching.trainer import _json, _path, _production
        from .feature_producer import FrozenFrameProducer
        if evidence["source_classes"] == ["public_mechanical_fixture"]:
            if (produced.binding.get("source_class") != "public_mechanical_fixture"
                    or closed_session.get("status") != "PUBLIC_MECHANICS_ONLY"
                    or closed_session.get("mechanical_fixture_only") is not True
                    or produced.binding.get("implementation_sha256") != FrozenFrameProducer._code_hashes()):
                raise IdentityError("public construction must preserve actual public feature provenance")
        else:
            pin = evidence["prepared_pin"]
            prepared = _json(_path(pin["path"]))
            _production(produced.frame, produced.binding, closed_session,
                prepared_file_sha256=pin["file_sha256"], prepared_receipt_sha256=prepared.get("receipt_sha256"),
                implementation=FrozenFrameProducer._code_hashes())
        if type(tie_tolerance) not in (int, float) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise IdentityError("choice tolerance must be explicitly finite and nonnegative")
        self.model.set_phase("inference")
        with torch.no_grad():
            packet = self.model(produced.frame)
        intents = []
        for index, candidate in enumerate(document.candidates):
            if not bool(packet.candidate_mask[0, index]):
                continue
            heads, controls = {}, {}
            for family, head in packet.heads.items():
                taught = evidence["reviewed_labels"].get("head:" + family, set())
                heads[family] = [{"label_id": spec.label_id, "name": spec.name, "description": spec.description,
                    "activation": float(head.activations[0, index, col])} for col, spec in enumerate(head.specs[0])
                    if spec is not None and spec.label_id in taught and bool(head.mask[0, index, col])]
            for col, spec in enumerate(packet.heads["voice"].specs[0]):
                if (spec is not None and bool(packet.heads["voice"].mask[0, index, col])
                        and spec.label_id in evidence["reviewed_labels"].get("voice_control_values", set())
                        and spec.label_id in evidence["reviewed_labels"].get("voice_control_confidence", set())):
                    controls[spec.label_id] = {"name": spec.name, "description": spec.description, "unit": spec.unit,
                        "value": float(packet.voice_control_values[0, index, col]),
                        "confidence": float(packet.voice_control_confidence[0, index, col])}
            intents.append({"candidate_id": candidate.candidate_id, "text": candidate.text,
                "preference": float(packet.preferences[0, index]), "heads": heads, "voice_controls": controls,
                "co_valid_probability": float(packet.co_valid_probabilities[0, index])
                    if "co_valid_probabilities" in evidence["reviewed_fields"] else None})
        taught_preference = bool({"preferences", "preference_margins"} & evidence["reviewed_fields"]) \
            and bool(packet.identity_grounding_available[0])
        best = max((item["preference"] for item in intents), default=0)
        choices = [item for item in intents if best - item["preference"] <= tie_tolerance] if taught_preference else []
        # Reviewed co-valid choices remain alternatives even when ranking differs.
        if taught_preference:
            choices += [item for item in intents if item not in choices and item["co_valid_probability"] is not None
                        and item["co_valid_probability"] >= .5]
        status = "SELECTED" if len(choices) == 1 else "UNRESOLVED_PLURALITY" if choices else "UNRESOLVED_UNTAUGHT"
        text = choices[0]["text"] if len(choices) == 1 else None
        rendering = {"status": "NOT_RENDERED", "text": None, "delivery": None,
                     "renderer_id": None, "artifact_sha256": None}
        if renderer is not None and len(choices) == 1:
            rendered = renderer(deepcopy(choices[0]))
            if (type(rendered) is not RenderedResponse or not isinstance(rendered.text, str)
                    or not rendered.text.strip() or type(rendered.delivery) is not dict
                    or not isinstance(rendered.renderer_id, str) or not rendered.renderer_id.strip()):
                raise IdentityError("renderer must return an explicit actual realization")
            canonical(rendered.delivery)
            if rendered.artifact_sha256 is not None and (not isinstance(rendered.artifact_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", rendered.artifact_sha256)):
                raise IdentityError("renderer artifact requires an exact digest or explicit absence")
            # Reported realization is not acoustic verification or owner review.
            rendering = {"status": "TEXT_REALIZED_DELIVERY_UNVERIFIED" if rendered.text == text
                         and rendered.delivery == choices[0]["voice_controls"] else "UNRESOLVED_RENDERER",
                         **vars(rendered)}
        result = {"schema": "alice-personality-response-construction-v1", "status": status,
            "document_sha256": document.sha256, "frame_sha256": produced.binding["frame_sha256"],
            "feature_binding_sha256": produced.binding["binding_sha256"], "closed_session_sha256": closed_session["receipt_sha256"],
            "packet_sha256": fingerprint(packet), "model_state_sha256": evidence["model_state_sha256"],
            "checkpoint_manifest_sha256": evidence["checkpoint_manifest_sha256"], "source_classes": evidence["source_classes"],
            "selected_candidate_ids": [item["candidate_id"] for item in choices], "response_text": text,
            "intents": intents, "rendering": rendering, "tie_tolerance": tie_tolerance,
            "training_state": "OPTIMIZED_UNQUALIFIED", "qualification": "UNQUALIFIED",
            "acceptance_authority": False, "owner_fidelity_approved": False, "spoken_fidelity_verified": False}
        result["receipt_sha256"] = sha256(canonical(result)).hexdigest()
        return result
