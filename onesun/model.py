"""The label-free v0 NO2 model.

For instrument ``s`` and spectrum ``i`` the log count rate is modelled as

    log I_si(λ) = log F0_s(λ) + C(λ) + R_s(λ) + B_s(λ) a_si − q_si σ_NO2,s(λ) + ε_si(λ)

* ``log F0_s``: literature solar prior, convolved to the fitted instrument slit
* ``C``: broad correction to the solar reference, shared by all instruments
* ``R_s``: fine instrument response, orthogonal to the differential NO2 template
* ``B_s a_si``: nuisance basis (polynomial orders 0-4 plus the first two
  wavelength derivatives of log F0) with per-spectrum amplitudes
* ``q_si σ_NO2,s``: NO2 absorption; ``q`` is the dimensionless slant amplitude

Fitting uses only L0 spectra, timestamps and solar geometry. No L1 spectrum
and no L2 column enter the fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from onesun.dataset import InstrumentData, even_indices


REFERENCE_FILE = Path(__file__).with_name("reference_data") / "reference_conv05.csv"
COMMON_WAVELENGTHS = np.arange(400.0, 470.0 + 0.05, 0.1)
AVOGADRO = 6.02214076e23
NO2_REFERENCE_MOLECULES_CM2 = 1.3434e16
NO2_REFERENCE_MOL_M2 = NO2_REFERENCE_MOLECULES_CM2 * 1e4 / AVOGADRO


@dataclass(frozen=True)
class FitSettings:
    """Fixed choices of the v0 fit."""

    train_rows_per_instrument: int = 20_000
    calibration_rows: int = 3_000
    # Search box for the linear dispersion (first-pixel wavelength, step) and
    # the additional Gaussian FWHM. The defaults suit Pandora s1 spectrometers;
    # s2 or other instruments need their own box.
    wavelength_start_bounds: tuple[float, float] = (276.0, 286.0)
    wavelength_step_bounds: tuple[float, float] = (0.125, 0.136)
    extra_fwhm_bounds: tuple[float, float] = (0.0, 0.65)
    calibration_wavelength_range: tuple[float, float] = (390.0, 480.0)
    calibration_min_valid_pixels: int = 500
    # Pixels used to normalise each calibration spectrum before the median.
    calibration_normalisation_pixels: tuple[int, int] = (700, 1700)
    polynomial_order: int = 4
    iterations: int = 3
    shared_smoothing: tuple[int, int] = (101, 3)  # Savitzky-Golay, about 10 nm
    response_smoothing: tuple[int, int] = (7, 2)  # Savitzky-Golay, about 0.7 nm
    seed: int = 42


@dataclass
class InstrumentFit:
    """Everything needed to retrieve NO2 from one instrument's L0 spectra."""

    wavelength_start_nm: float
    wavelength_step_nm: float
    extra_fwhm_nm: float
    calibration_correlation: float
    physical_log_f0: np.ndarray
    no2_template: np.ndarray
    shared_log_correction: np.ndarray
    instrument_log_response: np.ndarray
    nuisance_basis: np.ndarray
    no2_residual_template: np.ndarray
    slant_intercept_scale: float

    @property
    def baseline(self) -> np.ndarray:
        """Effective log solar reference: log F0 + C + R."""
        return (
            self.physical_log_f0
            + self.shared_log_correction
            + self.instrument_log_response
        )


@dataclass
class Model:
    """A frozen v0 model for one or several jointly fitted instruments."""

    instruments: dict[str, InstrumentFit]
    settings: FitSettings = field(default_factory=FitSettings)
    wavelengths_nm: np.ndarray = field(
        default_factory=lambda: COMMON_WAVELENGTHS.copy()
    )
    report: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write the model as a portable ``.npz`` file (no pickling)."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"wavelengths_nm": self.wavelengths_nm}
        scalars: dict[str, dict[str, float]] = {}
        for key, fit in self.instruments.items():
            scalars[key] = {}
            for item in fields(fit):
                value = getattr(fit, item.name)
                if isinstance(value, np.ndarray):
                    arrays[f"{key}/{item.name}"] = value
                else:
                    scalars[key][item.name] = float(value)
        metadata = {
            "format": "onesun-v0",
            "instruments": list(self.instruments),
            "scalars": scalars,
            "settings": asdict(self.settings),
            "report": self.report,
        }
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        with np.load(Path(path), allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"]))
            instruments = {}
            for key in metadata["instruments"]:
                values: dict[str, Any] = dict(metadata["scalars"][key])
                for item in fields(InstrumentFit):
                    if item.name not in values:
                        values[item.name] = stored[f"{key}/{item.name}"]
                instruments[key] = InstrumentFit(**values)
            settings = metadata["settings"]
            settings = FitSettings(
                **{
                    name: tuple(value) if isinstance(value, list) else value
                    for name, value in settings.items()
                }
            )
            return cls(
                instruments=instruments,
                settings=settings,
                wavelengths_nm=stored["wavelengths_nm"],
                report=metadata["report"],
            )


def load_reference() -> pd.DataFrame:
    """Solar prior and NO2 optical depth (254.5 K, 1.3434e16 molec cm-2),
    both convolved with a 0.5 nm Gaussian on a 0.1 nm grid, 300-500 nm."""
    return pd.read_csv(REFERENCE_FILE, float_precision="round_trip")


def _blurred(values: np.ndarray, extra_fwhm_nm: float, step_nm: float) -> np.ndarray:
    sigma_pixels = extra_fwhm_nm / 2.354820045 / step_nm
    if sigma_pixels <= 1e-8:
        return values.copy()
    return gaussian_filter1d(values, sigma_pixels, mode="nearest")


def fit_wavelength_and_slit(
    spectra: np.ndarray,
    reference: pd.DataFrame | None = None,
    settings: FitSettings = FitSettings(),
) -> dict[str, float]:
    """Stage 1: linear dispersion and extra Gaussian slit width.

    Maximises the correlation between the high-passed log median spectrum
    and the high-passed, blurred log solar prior.
    """
    reference = load_reference() if reference is None else reference
    indices = even_indices(len(spectra), settings.calibration_rows)
    sample = np.asarray(spectra[indices], dtype=np.float64)
    first, last = settings.calibration_normalisation_pixels
    normalization = np.nanmedian(sample[:, first:last], axis=1)
    sample = sample / np.maximum(normalization[:, None], 1e-8)
    median = np.nanmedian(sample, axis=0)
    log_median = np.log(np.clip(median, 1e-7, None))
    observed_high_pass = log_median - savgol_filter(log_median, 101, 3)
    f0_wavelength = reference["wavelength_nm"].to_numpy(dtype=float)
    f0_values = reference["solar_f0"].to_numpy(dtype=float)
    f0_step = float(np.median(np.diff(f0_wavelength)))
    low, high = settings.calibration_wavelength_range

    def objective(parameters: np.ndarray) -> float:
        start, step, extra_fwhm = parameters
        pixel_wavelength = start + step * np.arange(median.size)
        prediction = np.interp(
            pixel_wavelength,
            f0_wavelength,
            _blurred(f0_values, extra_fwhm, f0_step),
            left=np.nan,
            right=np.nan,
        )
        valid = (
            np.isfinite(prediction)
            & (pixel_wavelength >= low)
            & (pixel_wavelength <= high)
        )
        if valid.sum() < settings.calibration_min_valid_pixels:
            return 2.0
        predicted_log = np.log(np.clip(prediction[valid], 1e-8, None))
        predicted_high_pass = predicted_log - savgol_filter(predicted_log, 101, 3)
        correlation = np.corrcoef(observed_high_pass[valid], predicted_high_pass)[0, 1]
        return float(-correlation)

    result = differential_evolution(
        objective,
        bounds=(
            settings.wavelength_start_bounds,
            settings.wavelength_step_bounds,
            settings.extra_fwhm_bounds,
        ),
        seed=settings.seed,
        maxiter=45,
        popsize=10,
        tol=1e-7,
        polish=True,
    )
    return {
        "wavelength_start_nm": float(result.x[0]),
        "wavelength_step_nm": float(result.x[1]),
        "extra_fwhm_nm": float(result.x[2]),
        "calibration_correlation": float(-result.fun),
    }


def resample_log_spectra(
    spectra: np.ndarray,
    wavelength_start_nm: float,
    wavelength_step_nm: float,
) -> np.ndarray:
    """Linear interpolation of raw pixels onto the common 0.1 nm grid, in log."""
    positions = (COMMON_WAVELENGTHS - wavelength_start_nm) / wavelength_step_nm
    lower = np.floor(positions).astype(int)
    lower = np.clip(lower, 0, spectra.shape[1] - 2)
    fraction = positions - lower
    values = (
        np.asarray(spectra[:, lower], dtype=np.float64) * (1.0 - fraction)[None, :]
        + np.asarray(spectra[:, lower + 1], dtype=np.float64) * fraction[None, :]
    )
    return np.log(np.clip(values, 1e-6, None))


def instrument_templates(
    calibration: Mapping[str, float],
    reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """log F0 prior and NO2 optical depth after the fitted extra slit."""
    reference = load_reference() if reference is None else reference
    wavelength = reference["wavelength_nm"].to_numpy(dtype=float)
    f0_step = float(np.median(np.diff(wavelength)))
    physical_f0 = np.interp(
        COMMON_WAVELENGTHS,
        wavelength,
        _blurred(
            reference["solar_f0"].to_numpy(dtype=float),
            calibration["extra_fwhm_nm"],
            f0_step,
        ),
    )
    no2 = np.interp(
        COMMON_WAVELENGTHS,
        wavelength,
        _blurred(
            reference["no2_optical_depth"].to_numpy(dtype=float),
            calibration["extra_fwhm_nm"],
            f0_step,
        ),
    )
    return np.log(np.clip(physical_f0, 1e-8, None)), no2


def nuisance_basis(physical_log_f0: np.ndarray, polynomial_order: int = 4) -> np.ndarray:
    """Stage 2: polynomials 0..order plus d/dλ and d²/dλ² of log F0."""
    scaled = (
        2.0
        * (COMMON_WAVELENGTHS - COMMON_WAVELENGTHS.min())
        / np.ptp(COMMON_WAVELENGTHS)
        - 1.0
    )
    columns = [scaled**order for order in range(polynomial_order + 1)]
    first = np.gradient(physical_log_f0, COMMON_WAVELENGTHS)
    second = np.gradient(first, COMMON_WAVELENGTHS)
    for derivative in (first, second):
        derivative = derivative - derivative.mean()
        columns.append(derivative / max(derivative.std(), 1e-12))
    return np.column_stack(columns)


def residualize_template(no2_template: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """NO2 template with its projection onto the nuisance basis removed."""
    nuisance_fit = basis @ np.linalg.lstsq(basis, no2_template, rcond=None)[0]
    residual = no2_template - nuisance_fit
    if np.linalg.norm(residual) < 1e-8:
        raise RuntimeError("NO2 template vanished into the nuisance basis.")
    return residual


def fit_amplitudes(
    log_spectra: np.ndarray,
    baseline: np.ndarray,
    no2_template: np.ndarray,
    basis: np.ndarray,
    no2_residual_template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-spectrum least squares via the Frisch-Waugh-Lovell identity.

    Returns the NO2 slant amplitude ``q``, the nuisance amplitudes ``a``
    (one row per spectrum) and the spectral residual ``ε``.
    """
    centered = log_spectra - baseline[None, :]
    denominator = float(no2_residual_template @ no2_residual_template)
    slant_scale = -(centered @ no2_residual_template) / denominator
    continuum_target = centered + slant_scale[:, None] * no2_template
    nuisance_coefficients = np.linalg.lstsq(basis, continuum_target.T, rcond=None)[0]
    fitted = (
        nuisance_coefficients.T @ basis.T
        - slant_scale[:, None] * no2_template[None, :]
    )
    return slant_scale, nuisance_coefficients.T, centered - fitted


def _orthogonalize(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return values - direction * float(values @ direction) / float(direction @ direction)


def _prepare_instrument(
    data: InstrumentData,
    reference: pd.DataFrame,
    settings: FitSettings,
    spectral_preprocessor: Callable[[np.ndarray], np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = even_indices(len(data), settings.train_rows_per_instrument)
    spectra = np.asarray(data.spectra[indices], dtype=float)
    if spectral_preprocessor is not None:
        spectra = spectral_preprocessor(spectra)
    calibration = fit_wavelength_and_slit(spectra, reference, settings)
    physical_log_f0, no2_template = instrument_templates(calibration, reference)
    basis = nuisance_basis(physical_log_f0, settings.polynomial_order)
    work = {
        "calibration": calibration,
        "log_spectra": resample_log_spectra(
            spectra,
            calibration["wavelength_start_nm"],
            calibration["wavelength_step_nm"],
        ),
        "physical_log_f0": physical_log_f0,
        "no2_template": no2_template,
        "nuisance_basis": basis,
        "no2_residual_template": residualize_template(no2_template, basis),
        "amf": np.asarray(data.amf[indices], dtype=float),
        "shared_log_correction": np.zeros_like(physical_log_f0),
        "instrument_log_response": np.zeros_like(physical_log_f0),
    }
    report = {
        **calibration,
        "available_training_rows": int(len(data)),
        "fit_training_rows": int(len(indices)),
    }
    return work, report


def _finalize(item: dict[str, Any]) -> tuple[InstrumentFit, np.ndarray]:
    """Stage 4: ordinary Langley regression of slant amplitude on AMF."""
    baseline = (
        item["physical_log_f0"]
        + item["shared_log_correction"]
        + item["instrument_log_response"]
    )
    slant, _, residual = fit_amplitudes(
        item["log_spectra"],
        baseline,
        item["no2_template"],
        item["nuisance_basis"],
        item["no2_residual_template"],
    )
    # A template-shaped reference error is constant in slant space, whereas
    # real absorption scales with AMF, so the intercept fixes the absolute
    # NO2/reference split without any L1 or L2 target.
    intercept = float(np.polyfit(item["amf"], slant, deg=1)[1])
    fit = InstrumentFit(
        **item["calibration"],
        physical_log_f0=item["physical_log_f0"],
        no2_template=item["no2_template"],
        shared_log_correction=item["shared_log_correction"],
        instrument_log_response=item["instrument_log_response"],
        nuisance_basis=item["nuisance_basis"],
        no2_residual_template=item["no2_residual_template"],
        slant_intercept_scale=intercept,
    )
    return fit, residual


def fit_model(
    training: Mapping[str, InstrumentData],
    settings: FitSettings = FitSettings(),
    *,
    spectral_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Model:
    """Fit one instrument, or several jointly with a shared correction C.

    Stage 1 calibrates wavelength and slit per instrument, stage 2 builds the
    nuisance basis, stage 3 alternates shared/instrument updates for
    ``settings.iterations`` passes and stage 4 fits the Langley intercept.
    """
    reference = load_reference()
    work: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "training_inputs": [
            "L0 spectra",
            "timestamps",
            "gas AMF geometry",
            "physical F0 prior",
            "fixed 254.5 K NO2 template",
        ],
        "instruments": {},
    }
    for key, data in training.items():
        work[key], report["instruments"][key] = _prepare_instrument(
            data, reference, settings, spectral_preprocessor
        )

    iteration_reports = []
    for iteration in range(settings.iterations):
        mean_residuals: dict[str, np.ndarray] = {}
        residual_rms: dict[str, float] = {}
        for key, item in work.items():
            baseline = (
                item["physical_log_f0"]
                + item["shared_log_correction"]
                + item["instrument_log_response"]
            )
            _, _, residual = fit_amplitudes(
                item["log_spectra"],
                baseline,
                item["no2_template"],
                item["nuisance_basis"],
                item["no2_residual_template"],
            )
            mean_residuals[key] = residual.mean(axis=0)
            residual_rms[key] = float(np.sqrt(np.mean(np.square(residual))))

        mean_across = np.mean(list(mean_residuals.values()), axis=0)
        shared_update = savgol_filter(mean_across, *settings.shared_smoothing)
        for item in work.values():
            item["shared_log_correction"] += shared_update
        for key, item in work.items():
            update = savgol_filter(
                mean_residuals[key] - shared_update, *settings.response_smoothing
            )
            item["instrument_log_response"] += _orthogonalize(
                update, item["no2_residual_template"]
            )
        iteration_reports.append(
            {"iteration": iteration + 1, "residual_rms": residual_rms}
        )

    instruments = {}
    for key, item in work.items():
        instruments[key], residual = _finalize(item)
        report["instruments"][key]["slant_intercept_scale"] = (
            instruments[key].slant_intercept_scale
        )
        report["instruments"][key]["final_log_rms"] = float(
            np.sqrt(np.mean(np.square(residual)))
        )
    report["iterations"] = iteration_reports
    return Model(instruments=instruments, settings=settings, report=report)


def fit_with_frozen_shared_correction(
    data: InstrumentData,
    shared_log_correction: np.ndarray,
    settings: FitSettings = FitSettings(),
) -> InstrumentFit:
    """Add a new instrument to an existing model while keeping C fixed.

    Only the wavelength scale, slit, response R and Langley intercept of the
    new instrument are fitted.
    """
    reference = load_reference()
    item, _ = _prepare_instrument(data, reference, settings, None)
    item["shared_log_correction"] = np.asarray(shared_log_correction, dtype=float).copy()
    for _ in range(settings.iterations):
        baseline = (
            item["physical_log_f0"]
            + item["shared_log_correction"]
            + item["instrument_log_response"]
        )
        _, _, residual = fit_amplitudes(
            item["log_spectra"],
            baseline,
            item["no2_template"],
            item["nuisance_basis"],
            item["no2_residual_template"],
        )
        update = savgol_filter(residual.mean(axis=0), *settings.response_smoothing)
        item["instrument_log_response"] += _orthogonalize(
            update, item["no2_residual_template"]
        )
    fit, _ = _finalize(item)
    return fit


def retrieve(
    data: InstrumentData,
    fit: InstrumentFit,
    *,
    spectral_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
    include_nuisance: bool = False,
) -> pd.DataFrame:
    """Apply a frozen instrument fit to L0 spectra.

    ``no2_mmol_m2`` is the vertical column
    ``(q - b) * N_ref / AMF`` with the fitted Langley intercept ``b``.
    ``slant_scale`` is the raw dimensionless amplitude ``q`` so that a
    different Langley intercept can be applied later (see ``onesun.langley``).
    """
    spectra = np.asarray(data.spectra, dtype=float)
    if spectral_preprocessor is not None:
        spectra = spectral_preprocessor(spectra)
    log_spectra = resample_log_spectra(
        spectra, fit.wavelength_start_nm, fit.wavelength_step_nm
    )
    slant, nuisance, residual = fit_amplitudes(
        log_spectra,
        fit.baseline,
        fit.no2_template,
        fit.nuisance_basis,
        fit.no2_residual_template,
    )
    amf = np.asarray(data.amf, dtype=float)
    vertical = (slant - fit.slant_intercept_scale) * NO2_REFERENCE_MOL_M2 / amf
    frame = pd.DataFrame(
        {
            "no2_mmol_m2": 1e3 * vertical,
            "amf": amf,
            "slant_scale": slant,
            "l0_reconstruction_log_rms": np.sqrt(np.mean(np.square(residual), axis=1)),
        },
        index=pd.DatetimeIndex(data.timestamps, name="timestamp"),
    )
    if include_nuisance:
        names = [f"poly_{order}" for order in range(nuisance.shape[1] - 2)]
        names += ["dlogF0_dlambda", "d2logF0_dlambda2"]
        for column, name in enumerate(names):
            frame[f"nuisance_{name}"] = nuisance[:, column]
    return frame
