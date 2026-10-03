"""Explicit reviewed-target teaching; mechanics never grant identity qualification."""
from .contracts import (ContrastConstraint, ReviewedTarget, TeachingBatch, TeachingError,
                        TrainingRecipe, TARGET_SCHEMA)
from .objectives import IdentityObjectives
from .trainer import (ArtifactPin, TeachingAdmission, IdentityTrainer, current_code_hashes,
                      model_fingerprint, recipe_fingerprint)

__all__ = ["ContrastConstraint", "ReviewedTarget", "TeachingBatch", "TeachingError", "TrainingRecipe",
           "TARGET_SCHEMA", "IdentityObjectives", "ArtifactPin", "TeachingAdmission", "IdentityTrainer",
           "current_code_hashes", "model_fingerprint", "recipe_fingerprint"]
