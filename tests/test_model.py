import numpy as np
import pandas as pd
import pytest

from onesun import langley
from onesun.dataset import InstrumentData, split_days
from onesun.model import (
    FitSettings,
    Model,
    _blurred,
    fit_model,
    load_reference,
    retrieve,
)

START, STEP, EXTRA_FWHM = 280.3, 0.1305, 0.3


def synthetic_instrument(rows=1500, seed=0):
    """Raw-pixel spectra built from the reference itself with a known NO2 load."""
    rng = np.random.default_rng(seed)
    reference = load_reference()
    wavelength = reference["wavelength_nm"].to_numpy()
    grid_step = float(np.median(np.diff(wavelength)))
    log_f0 = np.log(_blurred(reference["solar_f0"].to_numpy(), EXTRA_FWHM, grid_step))
    no2 = _blurred(reference["no2_optical_depth"].to_numpy(), EXTRA_FWHM, grid_step)
    pixels = START + STEP * np.arange(2048)
    log_f0, no2 = np.interp(pixels, wavelength, log_f0), np.interp(pixels, wavelength, no2)
    scaled = (pixels - 435.0) / 35.0
    times = pd.date_range("2024-06-01 05:00", periods=rows, freq="7min")
    amf = 1.0 + 3.0 * rng.random(rows)
    vertical = 5.0 + 20.0 * rng.random(rows)  # in units of the 1.3434e16 template
    slant = vertical * amf
    continuum = (
        rng.normal(0, 0.05, rows)[:, None]
        - 0.02 * amf[:, None] * scaled[None, :]
        + 0.005 * rng.normal(0, 1, rows)[:, None] * scaled[None, :] ** 2
    )
    log_counts = log_f0[None, :] + continuum - slant[:, None] * no2[None, :]
    log_counts += rng.normal(0, 1e-4, log_counts.shape)
    data = InstrumentData(np.exp(log_counts), pd.DatetimeIndex(times), amf)
    return data, slant


@pytest.fixture(scope="module")
def fitted():
    data, slant = synthetic_instrument()
    model = fit_model({"synthetic": data}, FitSettings(train_rows_per_instrument=1000))
    return data, slant, model


def test_calibration_recovers_dispersion_and_slit(fitted):
    _, _, model = fitted
    fit = model.instruments["synthetic"]
    pixels = np.arange(900, 1401)  # about 398-463 nm; start and step trade off
    fitted_scale = fit.wavelength_start_nm + fit.wavelength_step_nm * pixels
    assert np.abs(fitted_scale - (START + STEP * pixels)).max() < 0.03
    assert fit.extra_fwhm_nm == pytest.approx(EXTRA_FWHM, abs=0.05)


def test_slant_amplitude_tracks_injected_no2(fitted):
    data, slant, model = fitted
    result = retrieve(data, model.instruments["synthetic"])
    assert np.corrcoef(result["slant_scale"], slant)[0, 1] > 0.999
    assert np.polyfit(slant, result["slant_scale"], 1)[0] == pytest.approx(1.0, abs=0.05)


def test_save_and_load_roundtrip(fitted, tmp_path):
    data, _, model = fitted
    loaded = Model.load(model.save(tmp_path / "model"))
    pd.testing.assert_frame_equal(
        retrieve(data, model.instruments["synthetic"]),
        retrieve(data, loaded.instruments["synthetic"]),
    )
    assert loaded.settings == model.settings


def test_lower_envelope_langley_and_reanchoring(fitted):
    data, _, model = fitted
    result = retrieve(data, model.instruments["synthetic"])
    envelope = langley.lower_envelope_intercept(result, quantile=0.02)
    assert envelope.amf_bins >= 6
    assert np.isfinite(envelope.intercept_scale)
    same = langley.apply_intercept(result, model.instruments["synthetic"].slant_intercept_scale)
    np.testing.assert_allclose(same["no2_mmol_m2"], result["no2_mmol_m2"], rtol=1e-12)


def test_split_is_by_complete_day_and_reproducible():
    days = pd.date_range("2024-01-01", periods=100, freq="D")
    training, holdout = split_days(days, 0.2, seed=42)
    assert len(holdout) == 20 and training.intersection(holdout).empty
    assert split_days(days, 0.2, seed=42)[1].equals(holdout)
