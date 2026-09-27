"""Diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from onesun.model import COMMON_WAVELENGTHS, InstrumentFit

MODEL_COLOUR = "#f04486"
REFERENCE_COLOUR = "#222222"


def plot_fitted_terms(fit: InstrumentFit, title: str, destination: Path) -> None:
    """Solar prior with the shared correction, response, NO2 template, basis."""
    wavelength = COMMON_WAVELENGTHS
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, layout="constrained")
    prior = np.exp(fit.physical_log_f0)
    corrected = np.exp(fit.physical_log_f0 + fit.shared_log_correction)
    axes[0].plot(wavelength, prior / prior.mean(), color="0.5", lw=0.8, label="solar prior")
    axes[0].plot(wavelength, corrected / corrected.mean(), color=MODEL_COLOUR, lw=0.8,
                 label="prior + shared correction C")
    axes[0].set_ylabel("normalised F0")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(wavelength, 1e3 * fit.instrument_log_response, color="#1f5be0", lw=0.8)
    axes[1].set_ylabel("response R [×10⁻³]")
    axes[2].plot(wavelength, fit.no2_template, color="0.5", lw=0.8, label="NO2 optical depth")
    axes[2].plot(wavelength, fit.no2_residual_template, color=MODEL_COLOUR, lw=0.8,
                 label="orthogonal to nuisance basis")
    axes[2].set_ylabel("NO2 template")
    axes[2].legend(frameon=False, fontsize=8)
    for column in range(fit.nuisance_basis.shape[1]):
        axes[3].plot(wavelength, fit.nuisance_basis[:, column], lw=0.8)
    axes[3].set_ylabel("nuisance basis B")
    axes[3].set_xlabel("wavelength [nm]")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{title}: start {fit.wavelength_start_nm:.3f} nm, step "
        f"{fit.wavelength_step_nm:.5f} nm/px, extra FWHM {fit.extra_fwhm_nm:.3f} nm"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_days(
    aligned: pd.DataFrame,
    title: str,
    destination: Path,
    utc_offset: pd.Timedelta = pd.Timedelta(0),
) -> None:
    """Model NO2 against the reference, one panel per local day."""
    local = aligned.set_axis(aligned.index + utc_offset)
    days = sorted(local.index.normalize().unique())
    if not days:
        return
    columns = min(5, len(days))
    rows = int(np.ceil(len(days) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.8 * rows),
                             sharey=True, squeeze=False, layout="constrained")
    for axis, day in zip(axes.flat, days):
        subset = local.loc[local.index.normalize() == day]
        hour = (subset.index - day).total_seconds() / 3600.0
        if "reference_no2_mmol_m2" in subset:
            axis.scatter(hour, subset["reference_no2_mmol_m2"], s=5, lw=0,
                         color=REFERENCE_COLOUR, label="reference L2")
        axis.scatter(hour, subset["no2_mmol_m2"], s=5, lw=0, color=MODEL_COLOUR,
                     label="label-free")
        axis.set_title(pd.Timestamp(day).strftime("%Y-%m-%d"), fontsize=9)
        hours = utc_offset.total_seconds() / 3600
        axis.set_xlabel(f"local hour (UTC{hours:+g})" if hours else "UTC hour")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(days):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("NO₂ [mmol m⁻²]")
    axes[0, 0].legend(frameon=False, fontsize=7, markerscale=2)
    fig.suptitle(title)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_scatter(aligned: pd.DataFrame, metrics: dict, title: str, destination: Path) -> None:
    if "correlation" not in metrics:
        return
    x, y = aligned["reference_no2_mmol_m2"], aligned["no2_mmol_m2"]
    low = min(0.0, float(min(x.min(), y.min())))
    high = float(max(x.max(), y.max()))
    fig, axis = plt.subplots(figsize=(5, 5), layout="constrained")
    axis.scatter(x, y, s=4, lw=0, alpha=0.5, color=MODEL_COLOUR)
    axis.plot([low, high], [low, high], color="0.3", lw=0.8, ls="--")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("reference NO₂ [mmol m⁻²]")
    axis.set_ylabel("label-free NO₂ [mmol m⁻²]")
    axis.set_title(
        f"{title}\nr={metrics['correlation']:.3f}  bias={metrics['bias_mmol_m2']:.3f}  "
        f"slope={metrics['slope']:.3f}  RMSE={metrics['rmse_mmol_m2']:.3f}",
        fontsize=9,
    )
    axis.grid(alpha=0.2)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
