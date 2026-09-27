"""Access to Pandonia Global Network (PGN) L0 and L2 files.

L0 spectra are read from the native daily PGN text files. The only
preprocessing is: select one routine (SQ by default), subtract the dark
measurement taken inside the same routine, drop the dark measurements
themselves and divide by integration time. No wavelength
calibration, no cloud screen and no calibration file are applied.
"""

from __future__ import annotations

import bz2
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests


API = "https://api.pandonia-global-network.org/v1"
DARK_FILTERWHEELS = (6, 3)  # closed filter-wheel combination used for darks

# Column descriptions in the L0 header (prefix match).
_L0_COLUMNS = {
    "routine_code": "Two letter code of measurement routine",
    "time": "UT date and time for beginning of measurement",
    "routine_count": "Routine count",
    "repetition_count": "Repetition count",
    "integration_time": "Integration time [ms]",
    "filterwheel_pos1": "Position of filterwheel #1",
    "filterwheel_pos2": "Position of filterwheel #2",
    "scale_factor": "Scale factor for data",
}
_L0_SPECTRUM = "Mean over all cycles of raw counts for each pixel"
_TIMESTAMP = re.compile(r"^\d{8}T\d{6}")


@dataclass(frozen=True)
class Instrument:
    """One PGN spectrometer, e.g. ``Instrument("Rome-SAP", 117, 1)``."""

    location: str
    pandora: int
    spectrometer: int = 1

    @property
    def name(self) -> str:
        return f"Pandora{self.pandora}s{self.spectrometer}_{self.location}"

    def l0_filename(self, day: pd.Timestamp) -> str:
        return f"{self.name}_{pd.Timestamp(day):%Y%m%d}_L0.txt.bz2"


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    altitude_m: float


def _get(url: str, params: dict | None = None, retries: int = 3, timeout: float = 300.0):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
    return None


def list_l0_files(instrument: Instrument, start: str, end: str) -> list[str]:
    """Names of the daily L0 files available from the PGN API, inclusive range."""
    url = (
        f"{API}/files/{instrument.location}/{instrument.pandora}/"
        f"{instrument.spectrometer}/L0"
    )
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    # The API requires start < end, so ask for one more day and filter.
    query_end = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    response = _get(url, params={"start": first.strftime("%Y-%m-%d"), "end": query_end})
    if response is None:
        return []
    names = []
    for item in response.json():
        match = re.search(r"_(\d{8})_L0", item["filename"])
        if match and first <= pd.Timestamp(match.group(1)) <= last:
            names.append(item["filename"])
    return sorted(names)


def download_l0_text(filename: str) -> str | None:
    response = _get(f"{API}/download/{filename}")
    if response is None:
        return None
    content = response.content
    if filename.endswith(".bz2"):
        content = bz2.decompress(content)
    return content.decode("latin-1")


def download_l2(
    instrument: Instrument,
    start: str,
    end: str,
    destination: str | Path,
    *,
    code: str = "rnvs3",
    chunk_days: int = 31,
) -> Path:
    """Download an operational L2 product (default: rnvs3 NO2) for evaluation.

    The bulk endpoint is queried in monthly chunks and written to one file.
    Only data rows of the chunks after the first are appended; the first
    chunk's header is kept so that ``read_l2`` can resolve the columns.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq=f"{chunk_days}D")
    stop = pd.Timestamp(end) + pd.Timedelta(days=1)
    with destination.open("w", encoding="latin-1") as handle:
        header_written = False
        for chunk_start in days:
            chunk_stop = min(chunk_start + pd.Timedelta(days=chunk_days), stop)
            response = _get(
                f"{API}/download/bulk_l2",
                params={
                    "start_datetime": chunk_start.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end_datetime": chunk_stop.strftime("%Y-%m-%dT%H:%M:%S"),
                    "pan_id": instrument.pandora,
                    "spectrometer": str(instrument.spectrometer),
                    "location": instrument.location,
                    "code": code,
                },
            )
            if response is None:
                continue
            for line in response.content.decode("latin-1").splitlines():
                if _TIMESTAMP.match(line):
                    handle.write(line + "\n")
                elif not header_written:
                    handle.write(line + "\n")
            header_written = True
    return destination


def _header(lines: list[str]) -> tuple[dict[str, str], dict[str, int], tuple[int, int]]:
    separators = [i for i, line in enumerate(lines[:400]) if line.startswith("-----")]
    if len(separators) < 2:
        raise ValueError("Not a PGN L0 file: missing header separators")
    metadata = {}
    for line in lines[: separators[0]]:
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    positions: dict[str, int] = {}
    spectrum: tuple[int, int] | None = None
    for line in lines[separators[0] + 1 : separators[1]]:
        if not line.startswith("Column"):
            continue
        label, description = line.split(": ", 1)
        number = label.split()[1]
        if "-" in number:
            if description.startswith(_L0_SPECTRUM):
                first, last = map(int, number.split("-"))
                spectrum = (first - 1, last)
            continue
        for key, prefix in _L0_COLUMNS.items():
            if description.startswith(prefix) and key not in positions:
                positions[key] = int(number) - 1
    missing = set(_L0_COLUMNS) - set(positions)
    if missing or spectrum is None:
        raise ValueError(f"L0 header lacks required columns: {sorted(missing)}")
    return metadata, positions, spectrum


def parse_l0(text: str, routine: str = "SQ") -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse one routine from an L0 file into raw, unscaled counts.

    Returns a frame indexed by measurement start time with the columns
    ``routine_count``, ``repetition_count``, ``integration_time``,
    ``filterwheel_pos1``, ``filterwheel_pos2`` and ``data`` (counts per pixel),
    plus the file's header metadata.
    """
    lines = text.splitlines()
    metadata, positions, (first, last) = _header(lines)
    prefix = f"{routine} "
    rows, spectra = [], []
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line.split()
        if len(fields) < last or fields[4] == "#":  # comment/INFO/ERROR lines
            continue
        try:
            scale = float(fields[positions["scale_factor"]])
            if scale <= 0:
                continue
            spectrum = np.asarray(fields[first:last], dtype=np.float64) / scale
        except ValueError:
            continue
        rows.append(
            (
                fields[positions["time"]],
                int(fields[positions["routine_count"]]),
                int(fields[positions["repetition_count"]]),
                float(fields[positions["integration_time"]]),
                int(fields[positions["filterwheel_pos1"]]),
                int(fields[positions["filterwheel_pos2"]]),
            )
        )
        spectra.append(spectrum)
    columns = [
        "time", "routine_count", "repetition_count", "integration_time",
        "filterwheel_pos1", "filterwheel_pos2",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["time"] = pd.to_datetime(
        frame["time"].str.replace(r"(?<=\d{6})Z$", ".0Z", regex=True),
        format="%Y%m%dT%H%M%S.%fZ",
    )
    frame["data"] = spectra
    return frame.set_index("time").sort_index(), metadata


def dark_correct_and_normalise(
    frame: pd.DataFrame,
    dark_filterwheels: tuple[int, int] = DARK_FILTERWHEELS,
) -> pd.DataFrame:
    """Subtract the routine's dark, drop darks, divide by integration time [ms].

    Within each (UTC day, routine count) the dark with the highest repetition
    count is used. Routines without a dark measurement stay uncorrected.
    """
    frame = frame.copy()
    is_dark = (frame["filterwheel_pos1"] == dark_filterwheels[0]) & (
        frame["filterwheel_pos2"] == dark_filterwheels[1]
    )
    days = pd.DatetimeIndex(frame.index).normalize()
    spectra = np.stack(frame["data"].to_numpy()) if len(frame) else np.empty((0, 0))
    for _, positions in frame.groupby([days, "routine_count"]).indices.items():
        darks = positions[is_dark.to_numpy()[positions]]
        if len(darks):
            chosen = darks[frame["repetition_count"].to_numpy()[darks].argmax()]
            spectra[positions] = spectra[positions] - spectra[chosen]
    keep = ~is_dark.to_numpy()
    spectra = spectra[keep] / frame["integration_time"].to_numpy()[keep, None]
    frame = frame.loc[keep].drop(columns="data")
    frame["data"] = list(spectra)
    return frame


def _write_day(frame: pd.DataFrame, path: Path) -> None:
    table = frame.reset_index()
    table["data"] = [np.asarray(values, dtype=np.float32) for values in table["data"]]
    temporary = path.with_suffix(".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def archive_dir(data_dir: str | Path, instrument: Instrument) -> Path:
    return Path(data_dir) / "L0" / instrument.name


def build_l0_archive(
    instrument: Instrument,
    start: str,
    end: str,
    data_dir: str | Path = "data",
    *,
    routine: str = "SQ",
    progress=print,
) -> Path:
    """Download, preprocess and store L0 spectra, one parquet file per day.

    Days already in the archive are skipped, so an interrupted download can
    simply be restarted. Spectra are stored as float32 counts per ms. The
    site coordinates from the L0 header are written to ``site.json``.
    """
    directory = archive_dir(data_dir, instrument)
    directory.mkdir(parents=True, exist_ok=True)
    filenames = list_l0_files(instrument, start, end)
    if not filenames:
        raise RuntimeError(f"No L0 files for {instrument.name} between {start} and {end}")
    progress(f"{instrument.name}: {len(filenames)} daily L0 files available")
    for number, filename in enumerate(filenames, start=1):
        day = re.search(r"_(\d{8})_L0", filename).group(1)
        path = directory / f"{day}.parquet"
        if path.exists():
            continue
        text = download_l0_text(filename)
        if text is None:
            progress(f"  [{number}/{len(filenames)}] {day}: not downloadable")
            continue
        frame, metadata = parse_l0(text, routine)
        site_file = directory / "site.json"
        if not site_file.exists():
            site = Site(
                float(metadata["Location latitude [deg]"]),
                float(metadata["Location longitude [deg]"]),
                float(metadata["Location altitude [m]"]),
            )
            site_file.write_text(json.dumps(asdict(site), indent=2))
        frame = dark_correct_and_normalise(frame)
        _write_day(frame, path)
        progress(f"  [{number}/{len(filenames)}] {day}: {len(frame)} {routine} spectra")
    return directory


def load_site(data_dir: str | Path, instrument: Instrument) -> Site:
    return Site(**json.loads((archive_dir(data_dir, instrument) / "site.json").read_text()))


def read_l2(path: str | Path, gas: str = "nitrogen dioxide") -> pd.DataFrame:
    """Read timestamp, quality flag, vertical column, effective temperature and
    AMF of one gas from a PGN L2 file (columns resolved from the header)."""
    wanted = {
        "qflag": f"L2 data quality flag for {gas}",
        "vcd_mol_m2": f"{gas.capitalize()} total vertical column amount",
        "temperature_k": f"{gas.capitalize()} effective temperature",
        "amf": f"Direct {gas} air mass factor",
    }
    positions: dict[str, int] = {}
    rows = []
    with Path(path).open(encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("Column") and ": " in line:
                label, description = line.split(": ", 1)
                for key, prefix in wanted.items():
                    if key not in positions and description.startswith(prefix):
                        positions[key] = int(label.split()[1]) - 1
            elif _TIMESTAMP.match(line):
                values = line.split()
                rows.append([values[0]] + [values[positions[key]] for key in wanted])
    missing = set(wanted) - set(positions)
    if missing:
        raise ValueError(f"L2 file lacks columns for {gas}: {sorted(missing)}")
    frame = pd.DataFrame(rows, columns=["timestamp", *wanted])
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], format="ISO8601", utc=True, errors="coerce"
    ).dt.tz_localize(None)
    frame[list(wanted)] = frame[list(wanted)].astype(float)
    frame["qflag"] = frame["qflag"].astype(int)
    return (
        frame.dropna(subset=["timestamp"])
        .drop_duplicates("timestamp")
        .set_index("timestamp")
        .sort_index()
    )
