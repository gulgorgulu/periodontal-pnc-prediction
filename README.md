# Periodontal pocket non-closure prediction

Prediction model from:

> Görgülü NG, Mamaklıoğlu D, Kalkan Y, Güngörmek HS, Karaduman B, Doğan B.
> Predicting pocket non-closure after non-surgical periodontal therapy:
> multilevel analysis and externally validated machine-learning models.
> Manuscript submitted.

From baseline (pre-treatment) measurements only, the model predicts the
3-month outcome after non-surgical periodontal therapy of every site with
**baseline probing pocket depth (PPD) ≥ 5 mm**:

| Output | Level |
|---|---|
| Predicted 3-month PPD (mm) | site |
| P(BOP): probability of bleeding on probing at 3 months | site |
| P(PNC): probability of pocket non-closure at 3 months (3-month PPD ≥ 5 mm, or 4 mm with BOP) | site |
| Expected number of PNC sites (sum of the patient's P(PNC)) | patient |

This repository contains the final prediction model only. It does not
contain the study data or the analysis code.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, GPBoost 1.7.1.1, NumPy 2.5.2 and pandas 3.0.5. On
macOS, GPBoost needs the OpenMP runtime (`brew install libomp`).

## Use

Demo with three synthetic patients (`data/synthetic_baseline_chart.csv`):

```bash
python demo.py
```

```
=== Patient SYN-01 ===
Sites with baseline PPD >= 5 mm: 33
Expected number of PNC sites at 3 months: 15.5
Sites with P(PNC) >= 30%: 21

Tooth 47 mesiolingual  baseline  8 mm: predicted 3-month PPD 5.3 mm, P(BOP) 0.48, P(PNC) 0.86  <- high risk
...
Tooth 36 distolingual  baseline  5 mm: predicted 3-month PPD 3.4 mm, P(BOP) 0.33, P(PNC) 0.25
```

Your own chart (CSV or Excel, one row per site):

```bash
python predict_pnc_gpboost.py --input chart.csv --output site_predictions.csv --patient-output patients.csv
```

or from Python:

```python
from predict_pnc_gpboost import read_chart, predict_sites, patient_summary

sites = predict_sites(read_chart("chart.csv"))
patients = patient_summary(sites, high_risk=0.30)
```

The 30% high-risk mark is illustrative, not a validated decision threshold;
the probabilities themselves are the model output.

## Input

One row per site. Sites with baseline PPD < 5 mm may be included; they are
not scored. All columns are required for the sites with PPD ≥ 5 mm (no
missing values).

| Column | Coding |
|---|---|
| `Patient.Code` | patient identifier |
| `Site.No` | FDI tooth number (two digits) + surface: `DB`, `B`, `MB`, `MP`/`ML`, `P`/`L`, `DP`/`DL` (e.g. `36MB`) |
| `Age` | years |
| `Gender` | 1 female, 2 male |
| `Smoking` | 0 non-smoker, 1 smoker |
| `DMType` | 0 no diabetes, 1 well-controlled, 2 poorly controlled diabetes |
| `ToothType` | 1 incisor/canine, 2 premolar, 3 molar |
| `Buccal.Oral` | 1 mid-buccal/mid-oral, 2 interdental |
| `PD.0` | baseline PPD, mm |
| `CAL.0` | baseline clinical attachment level, mm |
| `BOP.0` | baseline bleeding on probing, 0/1 |
| `PI.0` | baseline Plaque Index, 0–3 |
| `Furcation` | furcation involvement, 0/1 |
| `Restoration` | 0 none, 1 filling/caries, 2 crown, 3 bridge abutment |
| `Vitality` | tooth vitality: 0 vital, 1 non-vital (endodontically treated) |

Other columns (as in `data/synthetic_baseline_chart.csv`) are ignored. A
code not used in the development data stops the prediction with an error.
A numeric value outside the development range (age 18–63 years, baseline
PPD 5–16 mm, baseline CAL 2–18 mm) gives a warning, as the prediction is
then an extrapolation.

## Model

- **3-month PPD** and **3-month BOP**: two GPBoost models (gradient-boosted
  trees with nested patient and tooth random intercepts) on 13 baseline
  predictors, fitted on all 235 development patients (12,396 sites with
  baseline PPD ≥ 5 mm).
- **New patients**: the patient being predicted was not in the development
  data, so the predictions are population averages over the patient and
  tooth random effects (for BOP, the logistic link averaged over the
  estimated random-intercept variance).
- **P(PNC)** = P(PPD ≥ 5 mm) + P(PPD = 4 mm) × P(BOP), with 3-month PPD read
  to the whole millimetre. The PPD probabilities come from the predicted PPD
  and the out-of-fold PPD prediction errors of the development data.

`models/` holds only what predicting a new patient needs: the boosted
trees of the two models (`*_gpboost_trees.txt`), the BOP random-intercept
variances and input coding (`model_metadata.json`), and the sorted
out-of-fold PPD prediction errors (`pd3_oof_residuals.npy`). No patient
record, identifier or training data is included.

## Performance

Derived PNC model in the study, with 95% CIs from a patient-level cluster
bootstrap:

| | Internal test (47 patients, 2,870 sites) | External cohort (35 patients, 1,966 sites) |
|---|---|---|
| AUC | 0.762 (0.717–0.805) | 0.722 (0.684–0.761) |
| Brier score | 0.189 (0.169–0.208) | 0.202 (0.186–0.219) |
| Calibration-in-the-large | −0.02 (−0.34 to 0.30) | −0.07 (−0.36 to 0.22) |
| Calibration slope | 0.94 (0.71–1.32) | 0.80 (0.62–1.00) |
| Expected PNC sites per patient, MAE | 10.3 (7.8–13.0) | 7.7 (6.0–9.6) |

The internal test used models fitted on the 188 training patients; the
external cohort, the models in this repository (fitted on all 235
development patients).

## Limitations

- **Not a clinical decision tool.** The model supports, and does not replace,
  clinical judgement; it is intended for research and for informing the
  discussion of expected treatment response.
- **No recalibration.** The model was applied to the external cohort
  unchanged. It was developed in 235 patients and externally validated in
  35; before use in another population, check its calibration there.
- **Scope.** Sites with baseline PPD ≥ 5 mm, 3 months after non-surgical
  periodontal therapy, measured and coded as above.
- Discrimination is moderate (AUC about 0.72 in the external cohort):
  individual site predictions are uncertain.

## Citation

If you use this model, please cite the article above (journal and DOI will
be added on publication). Citation metadata: `CITATION.cff`.

## License

MIT
