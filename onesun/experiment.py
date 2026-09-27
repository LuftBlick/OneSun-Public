"""End-to-end label-free experiment driven by a JSON configuration.

Order of operations (L2 is never used for fitting):

1. build or extend the L0 archive of every instrument,
2. split complete days into training and holdout,
3. fit the model on training L0 only and save it,
4. retrieve NO2 on holdout days and save the retrievals,
5. only then, optionally, download and compare the operational L2 product.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from onesun import evaluation, langley, pgn, plots
from onesun.dataset import (
    archive_timestamps,
    even_indices,
    load_archive,
    seasonally_balanced_days,
    split_days,
)
from onesun.model import FitSettings, Model, fit_model, retrieve


def _settings(overrides: dict[str, Any]) -> FitSettings:
    known = {item.name for item in fields(FitSettings)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown fit settings: {sorted(unknown)}")
    return FitSettings(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    )


def _instrument(spec: dict[str, Any]) -> pgn.Instrument:
    return pgn.Instrument(spec["location"], int(spec["pandora"]), int(spec.get("spectrometer", 1)))


def _site(spec: dict[str, Any], data_dir: Path, instrument: pgn.Instrument) -> pgn.Site:
    if {"latitude", "longitude", "altitude_m"} <= set(spec):
        return pgn.Site(spec["latitude"], spec["longitude"], spec["altitude_m"])
    return pgn.load_site(data_dir, instrument)


def _retrieve(archive, site, times, fit, chunk_rows: int = 20_000) -> pd.DataFrame:
    parts = []
    for first in range(0, len(times), chunk_rows):
        data = load_archive(archive, site.latitude, site.longitude, site.altitude_m,
                            timestamps=times[first:first + chunk_rows])
        parts.append(retrieve(data, fit))
    return pd.concat(parts).sort_index()


def run_experiment(config: dict[str, Any], *, download: bool = True, progress=print) -> Path:
    name = config["name"]
    data_dir = Path(config.get("data_dir", "data"))
    output = Path(config.get("output_dir", "runs")) / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    settings = _settings(config.get("fit", {}))
    split = config.get("split", {})
    langley_config = config.get("langley", {"method": "regression"})
    holdout_mode = config.get("holdout_retrieval", "example_days")
    example_days = int(config.get("example_days", 10))

    # Days are local calendar days: UTC shifted by ``utc_offset_hours``.
    offset = pd.Timedelta(hours=float(config.get("utc_offset_hours", 0)))

    def local_day(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return (pd.DatetimeIndex(times) + offset).normalize()

    # 1. Archives.
    prepared: dict[str, dict[str, Any]] = {}
    for key, spec in config["instruments"].items():
        instrument = _instrument(spec)
        if download:
            pgn.build_l0_archive(instrument, spec["start"], spec["end"], data_dir,
                                 progress=progress)
        archive = pgn.archive_dir(data_dir, instrument)
        first, last = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
        times = archive_timestamps(archive, str((first - pd.Timedelta(days=1)).date()),
                                   str((last + pd.Timedelta(days=1)).date()))
        days = local_day(times)
        prepared[key] = {
            "instrument": instrument,
            "archive": archive,
            "site": _site(spec, data_dir, instrument),
            "times": times[(days >= first) & (days <= last)],
        }

    # 2. Split complete local days, per instrument or common to all.
    fraction, seed = split.get("holdout_fraction", 0.2), split.get("seed", 42)
    if split.get("common", False):
        all_days = local_day(pd.DatetimeIndex(
            np.concatenate([item["times"].to_numpy() for item in prepared.values()])))
        common = split_days(all_days, fraction, seed)
    for key, item in prepared.items():
        times, days = item["times"], local_day(item["times"])
        training_days, holdout_days = common if split.get("common", False) else split_days(
            days, fraction, seed)
        training_times = times[days.isin(training_days)]
        selected = training_times[
            even_indices(len(training_times), settings.train_rows_per_instrument)
        ]
        holdout_counts = pd.Series(1, index=days[days.isin(holdout_days)])
        holdout_counts = holdout_counts.groupby(level=0).size()
        shown = seasonally_balanced_days(holdout_counts, min(example_days, len(holdout_counts)))
        progress(f"{key}: {len(times)} spectra on {len(training_days)} training and "
                 f"{len(holdout_days)} holdout days; fitting {len(selected)} rows")
        site = item["site"]
        item["training"] = load_archive(item["archive"], site.latitude, site.longitude,
                                        site.altitude_m, timestamps=selected)
        item["holdout_times"] = times[days.isin(holdout_days)]
        item["example_times"] = times[days.isin(shown)]
        item["example_days"] = shown
        (output / f"{key}_split.json").write_text(json.dumps({
            "utc_offset_hours": offset.total_seconds() / 3600,
            "training_days": training_days.strftime("%Y-%m-%d").tolist(),
            "holdout_days": holdout_days.strftime("%Y-%m-%d").tolist(),
            "example_days": shown.strftime("%Y-%m-%d").tolist(),
        }, indent=2))

    # 3. Fit and freeze.
    training = {key: item["training"] for key, item in prepared.items()}
    if config.get("joint", True):
        model = fit_model(training, settings)
        model.report["mode"] = "joint"
    else:
        instruments, reports = {}, {}
        for key, data in training.items():
            single = fit_model({key: data}, settings)
            instruments[key] = single.instruments[key]
            reports[key] = single.report
        model = Model(instruments, settings, report={"mode": "independent", "fits": reports})
    for key, item in prepared.items():
        if langley_config.get("method") == "lower_envelope":
            training_retrieval = retrieve(item["training"], model.instruments[key])
            envelope = langley.lower_envelope_intercept(
                training_retrieval,
                **{k: v for k, v in langley_config.items() if k != "method"},
            )
            model.instruments[key].slant_intercept_scale = envelope.intercept_scale
            model.report.setdefault("lower_envelope_langley", {})[key] = {
                k: v for k, v in asdict(envelope).items() if k != "envelope"
            }
    model_path = model.save(output / "model.npz")
    (output / "training_report.json").write_text(json.dumps(model.report, indent=2))
    progress(f"model saved to {model_path}")
    for key, fit in model.instruments.items():
        plots.plot_fitted_terms(fit, key, output / f"{key}_fitted_terms.png")

    # 4. Retrieve holdout spectra (no L2 involved).
    retrievals = {}
    for key, item in prepared.items():
        times = item["example_times"] if holdout_mode == "example_days" else item["holdout_times"]
        retrievals[key] = _retrieve(item["archive"], item["site"], times, model.instruments[key])
        retrievals[key].to_csv(output / f"{key}_holdout_retrieval.csv")

    # 5. Compare with the operational product, after the model is frozen.
    evaluate = config.get("evaluate")
    if not evaluate:
        for key, frame in retrievals.items():
            shown = frame.loc[local_day(frame.index).isin(prepared[key]["example_days"])]
            plots.plot_days(shown, key, output / f"{key}_example_days.png", offset)
        return output
    summary, screened_frames = {}, {}
    for key, item in prepared.items():
        spec = config["instruments"][key]
        code = evaluate.get("l2_code", "rnvs3")
        l2_path = data_dir / "L2" / f"{item['instrument'].name}_{code}_{spec['start']}_{spec['end']}.txt"
        if not l2_path.exists():
            pgn.download_l2(item["instrument"],
                            str((pd.Timestamp(spec["start"]) - pd.Timedelta(days=1)).date()),
                            str((pd.Timestamp(spec["end"]) + pd.Timedelta(days=1)).date()),
                            l2_path, code=code)
        aligned = evaluation.align_with_reference(retrievals[key], pgn.read_l2(l2_path))
        aligned["passes_screen"] = evaluation.reconstruction_screen(
            aligned["l0_reconstruction_log_rms"], evaluate.get("screen_mad_multiplier", 10.0)
        )
        aligned.to_csv(output / f"{key}_holdout_vs_{code}.csv")
        screened = screened_frames[key] = aligned.loc[aligned["passes_screen"]]
        summary[key] = {
            "screened": evaluation.comparison_metrics(screened),
            "all": evaluation.comparison_metrics(aligned),
        }
        shown = screened.loc[local_day(screened.index).isin(item["example_days"])]
        plots.plot_days(shown, f"{key} vs {code}", output / f"{key}_example_days.png", offset)
        plots.plot_scatter(screened, summary[key]["screened"], f"{key} vs {code}",
                           output / f"{key}_scatter.png")
        progress(f"{key}: " + ", ".join(
            f"{k}={v:.4g}" for k, v in summary[key]["screened"].items()))
    for left, right in config.get("collocated_pairs", []):
        summary[f"{left} vs {right}"] = evaluation.pair_metrics(
            screened_frames[left], screened_frames[right]
        )
        progress(f"{left} vs {right}: " + ", ".join(
            f"{k}={v:.4g}" for k, v in summary[f"{left} vs {right}"].items()))
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return output


def retrieve_record(
    model_path: str | Path,
    key: str,
    location: str,
    pandora: int,
    spectrometer: int,
    start: str,
    end: str,
    data_dir: str | Path = "data",
    destination: str | Path | None = None,
) -> Path:
    """Apply a frozen model to every archived spectrum in a date range."""
    model = Model.load(model_path)
    instrument = pgn.Instrument(location, pandora, spectrometer)
    archive = pgn.archive_dir(data_dir, instrument)
    site = pgn.load_site(data_dir, instrument)
    frame = _retrieve(archive, site, archive_timestamps(archive, start, end),
                      model.instruments[key])
    destination = Path(destination or Path(model_path).with_name(f"{key}_{start}_{end}.csv"))
    frame.to_csv(destination)
    return destination


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())

