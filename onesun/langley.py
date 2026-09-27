"""Langley intercepts for the NO2 slant amplitude.

The fitted slant amplitude ``q`` contains a constant offset ``b`` (reference
structure projected onto the NO2 template) that does not scale with air mass.
``fit_model`` estimates ``b`` by ordinary regression of ``q`` on AMF over the
training rows. For clean or high-latitude sites a lower-envelope Langley,
a low quantile of ``q`` in air-mass bins, is more robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from onesun.model import NO2_REFERENCE_MOL_M2


@dataclass(frozen=True)
class EnvelopeLangley:
    intercept_scale: float
    envelope_slope_scale: float  # diagnostic background; not subtracted
    quantile: float
    amf_bins: int
    input_rows: int
    quality_rows: int
    amf_min: float
    amf_max: float
    quality_threshold_log_rms: float
    envelope: pd.DataFrame


def regression_intercept(slant_scale, amf) -> float:
    """Ordinary Langley: intercept of a straight line of q against AMF."""
    return float(np.polyfit(np.asarray(amf, float), np.asarray(slant_scale, float), 1)[1])


def lower_envelope_intercept(
    retrieval: pd.DataFrame,
    *,
    quantile: float = 0.02,
    quality_mad_multiplier: float = 3.0,
    amf_range: tuple[float, float] = (1.0, 5.0),
    target_rows_per_bin: int = 100,
    maximum_bins: int = 40,
    minimum_bins: int = 6,
) -> EnvelopeLangley:
    """Lower-envelope Langley from training-row retrievals (L0 only).

    ``retrieval`` is the output of ``onesun.model.retrieve`` for the training
    spectra. Rows are screened by reconstruction log-RMS (median + k MAD),
    split into equal-population AMF bins, and the ``quantile`` of ``q`` per
    bin is regressed on the bin's median AMF (weighted by sqrt(count)).
    """
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie between zero and 0.5")
    valid = retrieval.loc[
        np.isfinite(retrieval["slant_scale"])
        & np.isfinite(retrieval["amf"])
        & retrieval["amf"].between(*amf_range)
    ].copy()
    if valid.empty:
        raise ValueError("No finite rows in the requested AMF range")
    rms = valid["l0_reconstruction_log_rms"]
    median = float(rms.median())
    threshold = median + quality_mad_multiplier * float(np.nanmedian(np.abs(rms - median)))
    screened = valid.loc[rms <= threshold].copy()
    bins = min(maximum_bins, len(screened) // target_rows_per_bin)
    if bins < minimum_bins:
        raise ValueError("Too few quality-screened rows for the Langley envelope")
    screened["amf_bin"] = pd.qcut(screened["amf"], bins, duplicates="drop")
    envelope = screened.groupby("amf_bin", observed=True).agg(
        amf=("amf", "median"),
        slant=("slant_scale", lambda values: values.quantile(quantile)),
        count=("slant_scale", "size"),
    )
    envelope = envelope.loc[envelope["count"] >= max(20, target_rows_per_bin // 2)]
    if len(envelope) < minimum_bins or float(np.ptp(envelope["amf"])) < 1.0:
        raise ValueError("Insufficient AMF leverage for the Langley envelope")
    slope, intercept = np.polyfit(
        envelope["amf"], envelope["slant"], 1, w=np.sqrt(envelope["count"].to_numpy(float))
    )
    return EnvelopeLangley(
        intercept_scale=float(intercept),
        envelope_slope_scale=float(slope),
        quantile=float(quantile),
        amf_bins=int(len(envelope)),
        input_rows=int(len(valid)),
        quality_rows=int(len(screened)),
        amf_min=float(envelope["amf"].min()),
        amf_max=float(envelope["amf"].max()),
        quality_threshold_log_rms=float(threshold),
        envelope=envelope.reset_index(drop=True),
    )


def apply_intercept(retrieval: pd.DataFrame, intercept_scale: float) -> pd.DataFrame:
    """Recompute vertical columns with a different Langley intercept."""
    result = retrieval.copy()
    result["no2_mmol_m2"] = (
        1e3 * (result["slant_scale"] - intercept_scale) * NO2_REFERENCE_MOL_M2 / result["amf"]
    )
    return result
