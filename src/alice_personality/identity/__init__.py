"""First-party identity tensor mechanics; no trained personality is provided."""
from .contracts import (CALIBRATION_FAMILIES, CALIBRATION_SCHEMA, calibration_candidate_mask,
                        CONCEPT_VIEWS, EDGE_ROLES, FRAME_SCHEMA, HEAD_FAMILIES,
                        PACKET_SCHEMA, CalibrationBatch, ConceptRecord, FrozenTokenBank, HeadOutput,
                        IdentityDecisionPacket, IdentityError, IdentityFrame,
                        IdentityModelConfig, IdentityRepresentation, LabelSpec,
                        SourceRecord, SupportGraph)
from .model import IdentityCalibration, IdentityJudgmentModel, IdentityModel, IdentityRepresentationModel
from .checkpoints import load_untrained_snapshot, save_untrained_snapshot

__all__ = ["CONCEPT_VIEWS", "EDGE_ROLES", "FRAME_SCHEMA", "HEAD_FAMILIES", "PACKET_SCHEMA",
           "CALIBRATION_FAMILIES", "CALIBRATION_SCHEMA", "calibration_candidate_mask",
           "CalibrationBatch", "ConceptRecord", "FrozenTokenBank", "HeadOutput", "IdentityDecisionPacket", "IdentityError",
           "IdentityFrame", "IdentityModelConfig", "IdentityRepresentation", "LabelSpec", "SourceRecord",
           "SupportGraph", "IdentityCalibration", "IdentityJudgmentModel", "IdentityModel",
           "IdentityRepresentationModel", "load_untrained_snapshot", "save_untrained_snapshot"]
