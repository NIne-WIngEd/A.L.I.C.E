#!/usr/bin/env python3
from __future__ import annotations

import torch

import train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale as base
from alice_personality.n0.adaptive_multi_view_latent_pool_objectives_v0_2 import (
    LatentPoolV02ObjectiveWeights,
    adaptive_latent_pool_v0_2_objective,
    counterfactual_target_margin_loss,
)
from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
)


def corrected_loss_bundle(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    batch: dict[str, torch.Tensor],
    weights: LatentPoolV02ObjectiveWeights,
) -> dict[str, torch.Tensor]:
    output = base.latent_forward(model, batch)
    losses = adaptive_latent_pool_v0_2_objective(
        latent_slots=output["latent_slots"],
        pooled_state=output["pooled_state"],
        target_semantic=batch["semantic_target"],
        view_attention_mass=output["view_attention_mass"],
        view_available=output["view_available"],
        channel_attention_mass=output["channel_attention_mass"],
        counterfactual_slots=None,
        weights=weights,
    )
    decisive, best_view = base.decisive_view_mask(batch)
    counterfactual_output = base.counterfactual_forward(model, batch, decisive, best_view)
    if counterfactual_output is None:
        counterfactual = output["latent_slots"].sum() * 0.0
    else:
        counterfactual = counterfactual_target_margin_loss(
            output["latent_slots"][decisive],
            counterfactual_output["latent_slots"][decisive],
            batch["semantic_target"][decisive],
        )
    total = losses["total"] + weights.counterfactual_margin * counterfactual
    return {
        **losses,
        "counterfactual_margin": counterfactual,
        "total": total,
    }


# The base trainer resolves its module-level loss_bundle dynamically inside
# main(), so replacing it here preserves one implementation of every other
# lineage/evaluation/checkpoint rule while correcting the decisive-row scope.
base.loss_bundle = corrected_loss_bundle


if __name__ == "__main__":
    base.main()
