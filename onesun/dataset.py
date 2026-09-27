"""Spectra containers, day-level splits and solar geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pysolar.solar import get_altitude


@dataclass
class InstrumentData:
    """L0 spectra of one instrument (counts per ms, one row per spectrum)."""

    spectra: np.ndarray
    timestamps: pd.DatetimeIndex
    amf: np.ndarray
    filterwheel_pos1: np.ndarray | None = None
    filterwheel_pos2: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.timestamps)

    def subset(self, mask_or_indices) -> "InstrumentData":
        index = np.arange(len(self))[mask_or_indices]
        return InstrumentData(
            self.spectra[index],
            self.timestamps[index],
            self.amf[index],
            None if self.filterwheel_pos1 is None else self.filterwheel_pos1[index],
            None if self.filterwheel_pos2 is None else self.filterwheel_pos2[index],
        )

    def on_days(self, days) -> "InstrumentData":
        days = pd.DatetimeIndex(days).normalize()
        return self.subset(self.timestamps.normalize().isin(days))


def even_indices(length: int, maximum: int) -> np.ndarray:
    """At most ``maximum`` evenly spaced, chronologically ordered positions."""
    if length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=int)


def gas_air_mass(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    altitude_m: float,
) -> np.ndarray:
    """Direct-sun trace-gas air mass for an effective layer at 21 km (Komhyr).

    The solar zenith angle is computed with pysolar for each UTC timestamp.
    """
    times = pd.DatetimeIndex(timestamps)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    altitude = np.fromiter(
        (get_altitude(latitude, longitude, t.to_pydatetime()) for t in times),
        dtype=np.float64,
        count=len(times),
    )
    sza = np.radians(90.0 - altitude)
    earth_radius_km, layer_km = 6371.229, 21.0
    return (earth_radius_km + layer_km) / np.sqrt(
        (earth_radius_km + layer_km) ** 2
        - (earth_radius_km + altitude_m / 1000.0) ** 2 * np.sin(sza) ** 2
    )


def _dataset(archive: str | Path) -> ds.Dataset:
    files = sorted(str(path) for path in Path(archive).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {archive}")
    return ds.dataset(files, format="parquet")


def _range_filter(start: str | None, end: str | None):
    expression = None
    if start is not None:
        expression = ds.field("time") >= pd.Timestamp(start)
    if end is not None:
        upper = ds.field("time") < pd.Timestamp(end) + pd.Timedelta(days=1)
        expression = upper if expression is None else expression & upper
    return expression


def archive_timestamps(
    archive: str | Path, start: str | None = None, end: str | None = None
) -> pd.DatetimeIndex:
    """All spectrum timestamps in an L0 archive, optionally in a date range."""
    table = _dataset(archive).to_table(
        columns=["time"], filter=_range_filter(start, end)
    )
    times = pd.DatetimeIndex(table.column("time").to_numpy())
    return pd.DatetimeIndex(times.unique()).sort_values()


def load_archive(
    archive: str | Path,
    latitude: float,
    longitude: float,
    altitude_m: float,
    *,
    start: str | None = None,
    end: str | None = None,
    days=None,
    timestamps=None,
) -> InstrumentData:
    """Load spectra from an archive written by ``onesun.pgn.build_l0_archive``.

    ``start``/``end`` select an inclusive date range, ``days`` a set of UTC
    days and ``timestamps`` exact spectra. Only selected rows are read.
    """
    selected = archive_timestamps(archive, start, end)
    if days is not None:
        selected = selected[selected.normalize().isin(pd.DatetimeIndex(days).normalize())]
    if timestamps is not None:
        selected = selected[selected.isin(pd.DatetimeIndex(timestamps))]
    if selected.empty:
        raise ValueError(f"No spectra selected from {archive}")
    dataset = _dataset(archive)
    days_needed = pd.DatetimeIndex(selected.normalize().unique())
    parts = []
    for day in days_needed:  # read day by day to keep the filter small
        day_times = selected[selected.normalize() == day]
        expression = _range_filter(str(day.date()), str(day.date())) & ds.field(
            "time"
        ).isin(day_times.as_unit("ns").to_numpy())
        parts.append(dataset.to_table(filter=expression).to_pandas())
    frame = pd.concat(parts).sort_values("time").drop_duplicates("time")
    times = pd.DatetimeIndex(frame["time"])
    spectra = np.stack(frame["data"].to_numpy()).astype(np.float64)
    return InstrumentData(
        spectra=spectra,
        timestamps=times,
        amf=gas_air_mass(times, latitude, longitude, altitude_m),
        filterwheel_pos1=frame["filterwheel_pos1"].to_numpy(),
        filterwheel_pos2=frame["filterwheel_pos2"].to_numpy(),
    )


def split_days(
    days,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Random split of complete days into training and holdout days."""
    days = pd.DatetimeIndex(pd.DatetimeIndex(days).normalize().unique()).sort_values()
    rng = np.random.default_rng(seed)
    shuffled = pd.DatetimeIndex(rng.permutation(days.values))
    count = min(max(int(round(len(days) * holdout_fraction)), 1), len(days) - 1)
    holdout = pd.DatetimeIndex(shuffled[:count]).sort_values()
    return days[~days.isin(holdout)], holdout


def _season(day: pd.Timestamp) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(pd.Timestamp(day).month, "SON")


def seasonally_balanced_days(daily_counts: pd.Series, count: int = 10) -> pd.DatetimeIndex:
    """Pick data-rich example days with an equal minimum share per season.

    Takes the highest-count days within each represented season, then fills
    remaining slots with the highest-count days overall. Uses L0 counts only.
    """
    candidates = daily_counts.rename("spectra").rename_axis("day").reset_index()
    candidates["day"] = pd.to_datetime(candidates["day"])
    candidates["season"] = candidates["day"].map(_season)
    seasons = [s for s in ("DJF", "MAM", "JJA", "SON") if s in set(candidates["season"])]
    per_season = count // max(len(seasons), 1)
    parts = [
        candidates.loc[candidates["season"].eq(season)]
        .sort_values(["spectra", "day"], ascending=[False, True])
        .head(per_season)
        for season in seasons
    ]
    selected = pd.concat(parts, ignore_index=True)
    remaining = candidates.loc[~candidates["day"].isin(selected["day"])].sort_values(
        ["spectra", "day"], ascending=[False, True]
    )
    selected = pd.concat(
        [selected, remaining.head(count - len(selected))], ignore_index=True
    )
    return pd.DatetimeIndex(selected["day"]).sort_values()
