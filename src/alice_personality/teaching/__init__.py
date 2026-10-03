"""Explicit reviewed-target teaching; mechanics never grant identity qualification."""
from .contracts import (ContrastConstraint, ReviewedTarget, TeachingBatch, TeachingError,
                        TrainingRecipe, TARGET_SCHEMA)
from .objectives import IdentityObjectives
from .trainer import (ArtifactPin, TeachingAdmission, IdentityTrainer, current_code_hashes,
                      model_fingerprint, recipe_fingerprint)
from .cases import CASE_SCHEMA, NamedTarget, ReviewedTeachingCase, TargetFact
from .materialize import MaterializedCase, materialize_case

__all__ = ["ContrastConstraint", "ReviewedTarget", "TeachingBatch", "TeachingError", "TrainingRecipe",
           "TARGET_SCHEMA", "IdentityObjectives", "ArtifactPin", "TeachingAdmission", "IdentityTrainer",
           "current_code_hashes", "model_fingerprint", "recipe_fingerprint", "CASE_SCHEMA",
           "NamedTarget", "ReviewedTeachingCase", "TargetFact", "MaterializedCase", "materialize_case"]
