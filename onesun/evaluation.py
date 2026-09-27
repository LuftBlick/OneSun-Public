"""Comparison with an operational L2 product, used only after the fit.

The label-free model never sees L2 values. These helpers align frozen
retrievals with PGN rnvs3 NO2 and compute agreement statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GOOD_REFERENCE_FLAGS = (0, 1, 10, 11)


def reconstruction_screen(log_rms: pd.Series, mad_multiplier: float = 10.0) -> pd.Series:
    """True for spectra with L0 reconstruction log-RMS <= median + k MAD."""
    median = float(log_rms.median())
    mad = float((log_rms - median).abs().median())
    return log_rms <= median + mad_multiplier * mad


def align_with_reference(
    retrieval: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tolerance_seconds: float = 30.0,
    good_flags=GOOD_REFERENCE_FLAGS,
) -> pd.DataFrame:
    """Match each reference row to at most one retrieval.

    L0 timestamps mark the start of a measurement and L2 timestamps its
    centre, so the reference is searched forward within ``tolerance_seconds``.
    ``reference`` is the output of ``onesun.pgn.read_l2``.
    """
    reference = reference.loc[reference["qflag"].isin(good_flags)]
    reference = reference.add_prefix("reference_").rename_axis("reference_timestamp")
    aligned = pd.merge_asof(
        retrieval.rename_axis("timestamp").reset_index().sort_values("timestamp"),
        reference.reset_index().sort_values("reference_timestamp"),
        left_on="timestamp",
        right_on="reference_timestamp",
        direction="forward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).dropna(subset=["reference_timestamp"])
    aligned["reference_time_offset_seconds"] = (
        aligned["reference_timestamp"] - aligned["timestamp"]
    ).dt.total_seconds()
    aligned["reference_no2_mmol_m2"] = 1e3 * aligned["reference_vcd_mol_m2"]
    return (
        aligned.sort_values("reference_time_offset_seconds", kind="mergesort")
        .drop_duplicates("reference_timestamp", keep="first")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def comparison_metrics(
    frame: pd.DataFrame,
    model: str = "no2_mmol_m2",
    reference: str = "reference_no2_mmol_m2",
) -> dict[str, float]:
    """Correlation, bias, OLS slope, reference SD, affine residual and RMSE."""
    x = frame[reference].to_numpy(float)
    y = frame[model].to_numpy(float)
    if len(x) < 3:
        return {"rows": int(len(x))}
    error = y - x
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "rows": int(len(frame)),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "bias_mmol_m2": float(error.mean()),
        "slope": float(slope),
        "reference_sd_mmol_m2": float(x.std(ddof=1)),
        "affine_residual_mmol_m2": float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2))),
        "rmse_mmol_m2": float(np.sqrt(np.mean(error**2))),
    }


def pair_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_seconds: float = 120.0,
) -> dict[str, float]:
    """Consistency of two collocated instruments.

    Pairs each spectrum of ``left`` with the nearest unused spectrum of
    ``right`` within the tolerance, then compares the model difference with
    the reference difference on the same pairs. Inputs are aligned frames
    from ``align_with_reference``.
    """
    a = left.rename_axis("timestamp").reset_index().sort_values("timestamp")
    b = right.rename_axis("timestamp").reset_index().sort_values("timestamp")
    b["right_timestamp"] = b["timestamp"]
    pairs = pd.merge_asof(
        a, b, on="timestamp", direction="nearest", suffixes=("_left", "_right"),
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).dropna(subset=["right_timestamp"])
    pairs["gap"] = (pairs["timestamp"] - pairs["right_timestamp"]).abs()
    pairs = pairs.sort_values("gap").drop_duplicates("right_timestamp")
    model = pairs["no2_mmol_m2_left"] - pairs["no2_mmol_m2_right"]
    reference = pairs["reference_no2_mmol_m2_left"] - pairs["reference_no2_mmol_m2_right"]
    return {
        "pairs": int(len(pairs)),
        "model_correlation": float(pairs["no2_mmol_m2_left"].corr(pairs["no2_mmol_m2_right"])),
        "model_difference_rmse_mmol_m2": float(np.sqrt(np.mean(model**2))),
        "reference_correlation": float(
            pairs["reference_no2_mmol_m2_left"].corr(pairs["reference_no2_mmol_m2_right"])
        ),
        "reference_difference_rmse_mmol_m2": float(np.sqrt(np.mean(reference**2))),
    }
