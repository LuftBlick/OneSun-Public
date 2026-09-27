import numpy as np
import pandas as pd

from onesun import evaluation, pgn

PIXELS = 4


def l0_text(lines):
    header = [
        "File name: Pandora999s1_Test_20240715_L0.txt",
        "Location latitude [deg]: 41.9017",
        "Location longitude [deg]: 12.5158",
        "Location altitude [m]: 75",
        "-" * 40,
        "Column 1: Two letter code of measurement routine (** for manual operation)",
        "Column 2: UT date and time for beginning of measurement, yyyymmddThhmmssZ (ISO 8601)",
        "Column 3: Routine count (1 for the first routine of the day, 2 for the second, etc.)",
        "Column 4: Repetition count (1 for the first set in the routine, 2 for the second, etc.)",
        "Column 5: Total duration of measurement set in seconds (=# if the line is a comment line)",
        "Column 6: Integration time [ms]",
        "Column 7: Position of filterwheel #1, 0=filterwheel not used, 1-9 are valid positions",
        "Column 8: Position of filterwheel #2, 0=filterwheel not used, 1-9 are valid positions",
        "Column 9: Scale factor for data (to obtain unscaled data and uncertainty divide then by this number)",
        f"Columns 10-{9 + PIXELS}: Mean over all cycles of raw counts for each pixel",
        "-" * 40,
    ]
    return "\n".join(header + lines) + "\n"


def row(code, time, routine, repetition, integration, fw1, fw2, scale, counts):
    values = " ".join(str(value) for value in counts)
    return f"{code} {time} {routine} {repetition} 10.0 {integration} {fw1} {fw2} {scale} {values}"


def test_parse_dark_correct_and_normalise():
    text = l0_text([
        row("SQ", "20240715T080000.0Z", 3, 1, 50.0, 2, 2, 2.0, [220, 420, 620, 820]),
        row("SQ", "20240715T080010.5Z", 3, 2, 50.0, 2, 2, 2.0, [240, 440, 640, 840]),
        row("SQ", "20240715T080020Z", 3, 3, 50.0, 6, 3, 2.0, [20, 20, 20, 20]),
        "SQ 20240715T081000.0Z 4 1 # INFO Skipped routine SQ",
        row("SS", "20240715T082000.0Z", 5, 1, 50.0, 2, 2, 1.0, [1, 2, 3, 4]),
        row("SQ", "20240715T090000.0Z", 6, 1, 10.0, 2, 2, 1.0, [100, 200, 300, 400]),
    ])
    raw, metadata = pgn.parse_l0(text, "SQ")
    assert metadata["Location altitude [m]"] == "75"
    assert len(raw) == 4
    frame = pgn.dark_correct_and_normalise(raw)
    assert len(frame) == 3  # dark dropped
    spectra = np.stack(frame["data"].to_numpy())
    # (counts/scale - dark/scale) / integration time
    np.testing.assert_allclose(spectra[0], (np.array([220, 420, 620, 820]) - 20) / 2 / 50)
    np.testing.assert_allclose(spectra[1], (np.array([240, 440, 640, 840]) - 20) / 2 / 50)
    # routine 6 has no dark and stays uncorrected
    np.testing.assert_allclose(spectra[2], np.array([100, 200, 300, 400]) / 10)


def test_read_l2_and_alignment(tmp_path):
    columns = {36: "L2 data quality flag for nitrogen dioxide, 0=assured high quality",
               39: "Nitrogen dioxide total vertical column amount [moles per square meter]",
               45: "Nitrogen dioxide effective temperature [K]",
               50: "Direct nitrogen dioxide air mass factor"}
    header = ["File name: test", "-" * 40]
    header += [f"Column {i}: {columns.get(i, f'other {i}')}" for i in range(1, 51)]
    header += ["-" * 40]

    def data_line(time, flag, vcd):
        values = ["0"] * 50
        values[0], values[35], values[38], values[44], values[49] = time, str(flag), str(vcd), "250.0", "1.5"
        return " ".join(values)

    path = tmp_path / "l2.txt"
    path.write_text("\n".join(header + [
        data_line("20240715T080005.0Z", 0, 1e-4),
        data_line("20240715T080025.0Z", 12, 2e-4),
    ]) + "\n")
    reference = pgn.read_l2(path)
    assert list(reference["qflag"]) == [0, 12]
    retrieval = pd.DataFrame(
        {"no2_mmol_m2": [0.09, 0.19]},
        index=pd.DatetimeIndex(["2024-07-15 08:00:00", "2024-07-15 08:00:20"]),
    )
    aligned = evaluation.align_with_reference(retrieval, reference)
    assert len(aligned) == 1  # flag 12 is excluded
    assert aligned["reference_no2_mmol_m2"].iloc[0] == 0.1
    assert aligned["reference_time_offset_seconds"].iloc[0] == 5.0
