# OneSun: label-free NO₂ from raw Pandora spectra

This repository contains the retrieval method developed in the ESA project
OneSun. It retrieves NO₂ columns from raw Pandora direct-sun L0 spectra,
using no calibration file, no L1 spectra and no L2 labels. Several
instruments can be fitted jointly with one shared correction to the solar
reference.

## Model

```
log I_si(λ) = log F0_s(λ) + C(λ) + R_s(λ) + B_s(λ)·a_si − q_si·σ_NO2,s(λ) + ε_si(λ)
```

- `F0_s`: literature solar spectrum, blurred to the fitted instrument slit
- `C`: broad solar correction shared by all instruments
- `R_s`: instrument response, orthogonal to the NO₂ template
- `B_s`: nuisance basis (polynomials 0-4 and two derivatives of log F0)
- `q_si`: NO₂ slant amplitude (254.5 K cross-section)

The fit has four steps, all on 400-470 nm:

1. Fit the wavelength scale and slit width of each instrument against the
   solar spectrum.
2. Orthogonalise the NO₂ template against `B`, so that `q` becomes a linear
   projection.
3. Refine `C` and `R_s` from the mean residuals over three passes.
4. Fit a Langley intercept `b` on the training spectra.

The vertical column is then `VCD = (q − b)·N_ref / AMF`.

## Usage

```bash
pip install -e .
python -m onesun run examples/quickstart.json
```

The quickstart downloads ten days of the two collocated Seoul-SNU
instruments, Pandora 149s1 and 163s1, from the PGN API (~0.8 GB download,
~100 MB stored). It fits both jointly on 80 % of the local days, with a
lower-envelope Langley, and retrieves NO₂ on the held-out days. It then
compares each instrument with the operational PGN product (rnvs3) and the
two instruments with each other. The label-free columns sit below rnvs3, but
agree more closely between the two instruments than the two rnvs3 products
do. The model, retrievals, metrics and figures are written to `runs/<name>/`.
L2 data are opened only after the model has been saved.

Other commands:

```bash
python -m onesun download Rome-SAP 117 2024-01-01 2024-12-31
python -m onesun retrieve runs/quickstart_seoul/model.npz seoul_snu_p149 Seoul-SNU 149 2021-06-01 2021-06-10
```

Further configs in `examples/`:

- individual fits at Rome-SAP, Tel-Aviv and Thessaloniki
- a joint Rome-SAP/Tel-Aviv fit
- a lower-envelope Langley at the background site Izaña

Config keys:

- `joint`: fit the instruments with a shared `C` (`true`) or one at a time (`false`)
- `fit`: overrides for `onesun.FitSettings`; the default dispersion bounds suit s1 spectrometers
- `split`: holdout fraction and seed, by complete day; `common: true` uses one split for all instruments
- `utc_offset_hours`: defines local days for the split (e.g. 9 for Seoul); dates are otherwise UTC
- `langley`: `regression` or `lower_envelope` (with `quantile`)
- `evaluate`: optional comparison with L2
- `collocated_pairs`: instrument pairs to compare with each other

## Pretrained model

`models/seoul_snu_p149_p163_v0.npz` is a joint model for Seoul-SNU Pandora
149s1 and 163s1, trained on data from 2020-04 to 2021-10. Its instrument
keys are `seoul_snu_p149` and `seoul_snu_p163`.

```bash
python -m onesun download Seoul-SNU 149 2021-06-01 2021-06-10
python -m onesun retrieve models/seoul_snu_p149_p163_v0.npz seoul_snu_p149 Seoul-SNU 149 2021-06-01 2021-06-10
```

## Limitations

- Only NO₂ is retrieved.
- With a fixed-temperature cross-section and a Gaussian slit, columns are
  lower than rnvs3, with a slope of about 0.7. The correlation is typically
  above 0.9.
- Without an independent absolute calibration, the offset depends on the
  Langley intercept and therefore on the training period.

## Tests

```bash
pip install -e ".[test]" && pytest
```

## Acknowledgement

Developed by LuftBlick (A. Kreuter, B. Eder, M. Tiefengraber, A. Cede) in the
ESA project OneSun. Data from the Pandonia Global Network.
