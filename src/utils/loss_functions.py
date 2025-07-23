# utils/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# simple rmse loss
def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))


def spectral_fidelity_loss(L1_pred, L1_model):
    mse_loss = nn.MSELoss()
    return mse_loss(L1_pred, L1_model)


def sam_loss(pred, target, eps=1e-8):
    """


    SAM measures the spectral angle between predicted and target spectra,
    focusing on spectral shape rather than magnitude differences.

    Args:
        pred: Predicted spectra [B, N_wavelengths]
        target: Target spectra [B, N_wavelengths]
        eps: Small value to prevent division by zero

    Returns:
        SAM loss (lower is better, 0 = perfect match)
    """
    # pred, target: tensors of shape [B, N_wavelengths]
    num = (pred * target).sum(dim=1)  # dot product for each sample
    denom = torch.norm(pred, dim=1) * torch.norm(target, dim=1)
    cos = torch.clamp(num / (denom + eps), -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)
    return angle.mean()


def cosine_similarity_loss(pred, target, eps=1e-8):
    """
    Computes 1 - cosine similarity as a loss (lower is better).
    Args:
        pred: Predicted spectra [B, N_wavelengths]
        target: Target spectra [B, N_wavelengths]
        eps: Small value to prevent division by zero
    Returns:
        Mean cosine similarity loss over the batch
    """
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_norm = target / (target.norm(dim=1, keepdim=True) + eps)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    loss = 1.0 - cos_sim
    return loss.mean()


def daily_correlation_loss(totcol, amf: torch.Tensor):
    # Assuming totcol and amf are batches of daily values
    # totcol and amf should be normalized before this computation

    # If totcol has near-zero variance, correlation is undefined/zero.
    # Return 0.0 to avoid NaN issues and correctly reflect no correlation.
    if torch.all(torch.std(totcol, dim=0) < 1e-6):
        return torch.tensor(0.0, device=totcol.device, dtype=totcol.dtype)

    mean_totcol_per_column = torch.mean(totcol, dim=0)
    mean_amf = amf.mean()
    totcol_normalized = totcol - mean_totcol_per_column
    amf_normalized = amf - mean_amf
    corr_numerator = torch.sum(totcol_normalized * amf_normalized, dim=0)
    corr_denominator = torch.sqrt(torch.sum(totcol_normalized ** 2, dim=0) * torch.sum(amf_normalized ** 2, dim=0))
    correlation = corr_numerator / (corr_denominator + 1e-8)  # Avoid division by zero
    # Penalize the absolute value of correlation
    return torch.mean(torch.abs(correlation))


def combined_loss(L1_pred, L1_model, totcol, amf, mse_loss_w=0.1, corr_loss_w=0.1, sam_loss_w=0.1, cos_loss_w=0.1):
    """
    Combine spectral fidelity loss and daily correlation penalty.

    Args:
        L1_pred: Predicted L1 spectra
        L1_model: Model L1 spectra
        totcol: Total column (latent space)
        amf: Air mass factors
        alpha: Weight for correlation penalty
        beta: Weight for SAM (spectral angle mapper) loss
        gamma: Weight for cosine similarity loss

    Returns:
        Combined loss value
    """
    loss_spec = 0
    if mse_loss_w > 0:
        loss_spec = spectral_fidelity_loss(L1_pred, L1_model)

    loss_day = 0
    if corr_loss_w > 0:
        loss_day = daily_correlation_loss(totcol, amf)

    # Apply SAM loss if beta > 0
    loss_sam = 0.0
    if sam_loss_w > 0:
        loss_sam = sam_loss(L1_pred, L1_model)

    # Apply cosine similarity loss if gamma > 0
    loss_cosine = 0.0
    if cos_loss_w > 0:
        loss_cosine = cosine_similarity_loss(L1_pred, L1_model)

    return (mse_loss_w * loss_spec) + (corr_loss_w * loss_day) + (sam_loss_w * loss_sam) + (cos_loss_w * loss_cosine)

