"""Predicts, before non-surgical periodontal therapy, the 3-month outcome of
every site with baseline probing pocket depth (PPD) >= 5 mm:

- predicted 3-month PPD (mm)
- P(BOP): probability of bleeding on probing at 3 months
- P(PNC): probability of pocket non-closure at 3 months
  (PNC = 3-month PPD >= 5 mm, or 4 mm with bleeding on probing)

and, per patient, the expected number of PNC sites (sum of P(PNC)).

Two GPBoost models (3-month PPD and 3-month BOP) were fitted on 235
development patients. The patient being predicted is new to the models, so
the predictions are population averages over the patient and tooth random
effects. P(PNC) combines the two models with the PNC definition:

    P(PNC) = P(PPD >= 5 mm) + P(PPD = 4 mm) x P(BOP)

where the PPD probabilities come from the predicted PPD and the models'
out-of-fold prediction-error distribution in the development data, with
3-month PPD read to the whole millimetre.

Usage:
    python predict_pnc_gpboost.py --input chart.csv --output predictions.csv
    python predict_pnc_gpboost.py --input chart.xlsx --output predictions.csv --patient-output patients.csv

The input is one row per site (CSV or Excel), with the columns listed in
REQUIRED_COLUMNS and coded as in README.md. Sites with baseline PPD < 5 mm
may be included; they are not scored.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

REQUIRED_COLUMNS = [
    "Patient.Code", "Site.No", "Age", "Gender", "Smoking", "DMType", "ToothType", "Buccal.Oral",
    "PD.0", "BOP.0", "PI.0", "CAL.0", "Furcation", "Restoration", "Vitality",
]
MIN_BASELINE_PPD = 5

SURFACES = {
    "DB": "distobuccal", "B": "mid-buccal", "MB": "mesiobuccal",
    "MP": "mesiopalatal", "P": "mid-palatal", "DP": "distopalatal",
    "ML": "mesiolingual", "L": "mid-lingual", "DL": "distolingual",
}

# Gauss-Hermite nodes and weights (standard normal) for averaging the
# logistic link over the random-intercept distribution of a new patient/tooth.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def load_models(model_dir: Path = MODEL_DIR) -> dict:
    import gpboost as gpb

    meta = json.loads((model_dir / "model_metadata.json").read_text())
    return {
        "meta": meta,
        "pd3": gpb.Booster(model_file=str(model_dir / "pd3_gpboost_trees.txt")),
        "bop3": gpb.Booster(model_file=str(model_dir / "bop3_gpboost_trees.txt")),
        "residuals": np.sort(np.load(model_dir / "pd3_oof_residuals.npy")),
        "bop3_re_sd": float(np.sqrt(sum(meta["bop3_random_intercept_variance"].values()))),
    }


def read_chart(path) -> pd.DataFrame:
    path = Path(path)
    read = pd.read_excel if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    return read(path, dtype={"Patient.Code": str, "Site.No": str})


def prepare_features(chart: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites with baseline PPD >= 5 mm, model feature matrix). Checks the
    columns, missing values and the coding of the categorical predictors."""
    missing = [c for c in REQUIRED_COLUMNS if c not in chart.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    sites = chart[chart["PD.0"] >= MIN_BASELINE_PPD].reset_index(drop=True).copy()
    if sites.empty:
        raise ValueError(f"No site has baseline PPD >= {MIN_BASELINE_PPD} mm")
    na = sites[REQUIRED_COLUMNS].isna().sum()
    if na.any():
        raise ValueError(f"Missing values in sites with PPD >= {MIN_BASELINE_PPD} mm: {na[na > 0].to_dict()}")

    sites["IsMolar"] = (sites["ToothType"] == 3).astype(int)
    X = sites[meta["numeric_features"] + meta["categorical_features"]].copy()
    for col in meta["numeric_features"]:
        X[col] = X[col].astype(float)
    for col, levels in meta["categories"].items():
        unknown = sorted(set(X[col].astype(int)) - set(levels))
        if unknown:
            raise ValueError(f"{col}: codes {unknown} not used in the development data (allowed: {levels})")
        X[col] = pd.Categorical(X[col].astype(int), categories=levels)
    return sites, X


def derived_pnc_probability(pd3_pred, bop3_proba, residuals) -> np.ndarray:
    """P(PPD >= 4.5) + P(3.5 <= PPD < 4.5) x P(BOP), with P(PPD >= t) the
    share of development prediction errors r for which pred + r >= t."""
    def p_at_least(t):
        return 1.0 - np.searchsorted(residuals, t - np.asarray(pd3_pred), side="left") / len(residuals)

    p5 = p_at_least(4.5)
    return p5 + (p_at_least(3.5) - p5) * np.asarray(bop3_proba)


def predict_sites(chart: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """One row per site with baseline PPD >= 5 mm: predicted 3-month PPD,
    P(BOP) and P(PNC)."""
    models = models or load_models()
    sites, X = prepare_features(chart, models["meta"])
    pd3 = models["pd3"].predict(X, pred_latent=True)
    eta = models["bop3"].predict(X, pred_latent=True)
    bop3 = (1.0 / (1.0 + np.exp(-(eta[:, None] + models["bop3_re_sd"] * _GH_NODES[None, :])))) @ _GH_WEIGHTS
    pnc = derived_pnc_probability(pd3, bop3, models["residuals"])

    site_no = sites["Site.No"].astype(str)
    out = pd.DataFrame({
        "Patient.Code": sites["Patient.Code"].values,
        "Site.No": site_no.values,
        "Tooth": site_no.str[:2].values,
        "Surface": site_no.str[2:].map(SURFACES).fillna(site_no.str[2:]).values,
        "Baseline PPD (mm)": sites["PD.0"].values,
        "Predicted 3-month PPD (mm)": pd3,
        "P(BOP)": bop3,
        "P(PNC)": pnc,
    })
    _warn_out_of_range(sites, models["meta"])
    return out


def patient_summary(site_predictions: pd.DataFrame, high_risk: float = 0.30) -> pd.DataFrame:
    """Per patient: number of sites with baseline PPD >= 5 mm, expected
    number of PNC sites (sum of P(PNC)), and sites with P(PNC) >= high_risk."""
    g = site_predictions.groupby("Patient.Code", sort=False)["P(PNC)"]
    return pd.DataFrame({
        "Sites with baseline PPD >= 5 mm": g.size(),
        "Expected number of PNC sites": g.sum(),
        f"Sites with P(PNC) >= {high_risk:.0%}": g.apply(lambda p: int((p >= high_risk).sum())),
    }).reset_index()


def _warn_out_of_range(sites: pd.DataFrame, meta: dict) -> None:
    for col, (lo, hi) in meta.get("numeric_ranges", {}).items():
        outside = ((sites[col] < lo) | (sites[col] > hi)).sum()
        if outside:
            warnings.warn(f"{outside} site(s) have {col} outside the development range {lo}-{hi}; "
                          "predictions there are extrapolations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="baseline chart, one row per site (.csv or .xlsx)")
    parser.add_argument("--output", required=True, help="site-level predictions (.csv)")
    parser.add_argument("--patient-output", help="optional patient-level summary (.csv)")
    parser.add_argument("--high-risk", type=float, default=0.30,
                        help="P(PNC) at or above which a site is counted as high risk (default 0.30)")
    args = parser.parse_args()

    preds = predict_sites(read_chart(args.input))
    preds.round(3).to_csv(args.output, index=False)
    summary = patient_summary(preds, args.high_risk)
    if args.patient_output:
        summary.round(2).to_csv(args.patient_output, index=False)
    print(f"{len(preds)} sites with baseline PPD >= {MIN_BASELINE_PPD} mm scored -> {args.output}")
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
