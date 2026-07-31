"""
Training objective — Section III-E.

    L_gt    = sum_t (S_t - S_hat_t)^2                       (Eq. 14, regression MSE)
    L_rec   = MSE(reconstructed_fusion, F_fusion)            (Eq. 15)
    L_KL    = -1/2 * sum_i (1 + log(sigma_i^2) - sigma_i^2 - mu_i^2)   (Eq. 16)
    L_ELBO  = L_rec + L_KL                                   (Eq. 17)
    L_total = L_gt + L_ELBO                                  (Eq. 18)

We additionally expose an optional smoothness penalty (mentioned in the
abstract: "a smoothness constraint ... to suppress transient
disturbances") as a small temporal-difference regularizer on consecutive
predicted scores. It is off by default in the strict-reproduction path
(weight can be set to 0) and documented separately for the "improvements"
phase.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def regression_loss(pred: torch.Tensor, target: torch.Tensor,
                     sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. 14. Both pred and target: (B,) or (B, T).

    If `sample_weights` (B,) is given, computes a weighted mean instead of
    a plain mean — used to counteract severe class imbalance (e.g. DAiSEE
    has very few "Not Engaged" samples), where an unweighted MSE lets the
    model minimize error by collapsing toward the majority region instead
    of actually discriminating rare classes.
    """
    if sample_weights is None:
        return F.mse_loss(pred, target, reduction="mean")
    per_sample = F.mse_loss(pred, target, reduction="none")
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Eq. 16, averaged over the batch (summed over latent dims)."""
    kl = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=-1)
    return kl.mean()


def elbo_loss(recon: torch.Tensor, target: torch.Tensor,
              mu_trend: torch.Tensor, logvar_trend: torch.Tensor,
              mu_cycle: torch.Tensor, logvar_cycle: torch.Tensor,
              kl_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """Eq. 15-17, for the dual-branch VAE (trend + cycle KL terms summed)."""
    rec = F.mse_loss(recon, target, reduction="mean")
    kl_t = kl_divergence(mu_trend, logvar_trend)
    kl_c = kl_divergence(mu_cycle, logvar_cycle)
    kl = kl_t + kl_c
    elbo = rec + kl_weight * kl
    return {"L_rec": rec, "L_KL": kl, "L_ELBO": elbo}


def smoothness_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Optional temporal smoothness regularizer over consecutive window
    predictions within a session: mean squared first difference.
    pred: (B, T) sequence of predicted scores for consecutive windows in a
    session batch (T >= 2); returns 0 if T < 2."""
    if pred.dim() != 2 or pred.shape[1] < 2:
        return pred.new_zeros(())
    diffs = pred[:, 1:] - pred[:, :-1]
    return diffs.pow(2).mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor, vae_out: Dict[str, torch.Tensor],
                kl_weight: float = 1.0, smoothness_weight: float = 0.0,
                pred_sequence: Optional[torch.Tensor] = None,
                sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Eq. 18, with the optional smoothness term added on top.

    pred, target: (B,) — per-window continuous engagement scores.
    vae_out: dict returned by TrendCycleVAE.forward().
    pred_sequence: optional (B, T) if you're scoring multiple consecutive
        windows per batch item and want the smoothness penalty applied.
    sample_weights: optional (B,) per-sample weights for the regression
        term (see `regression_loss`), e.g. inverse class frequency to
        counter imbalance in the training labels.
    """
    l_gt = regression_loss(pred, target, sample_weights=sample_weights)
    elbo = elbo_loss(
        vae_out["recon"], vae_out["target"],
        vae_out["mu_trend"], vae_out["logvar_trend"],
        vae_out["mu_cycle"], vae_out["logvar_cycle"],
        kl_weight=kl_weight,
    )
    loss = l_gt + elbo["L_ELBO"]

    smooth = pred.new_zeros(())
    if smoothness_weight > 0 and pred_sequence is not None:
        smooth = smoothness_penalty(pred_sequence)
        loss = loss + smoothness_weight * smooth

    return {
        "loss": loss,
        "L_gt": l_gt,
        "L_rec": elbo["L_rec"],
        "L_KL": elbo["L_KL"],
        "L_ELBO": elbo["L_ELBO"],
        "L_smooth": smooth,
    }
