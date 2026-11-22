"""Demo: predictions for the synthetic patients in data/synthetic_baseline_chart.csv.

For each patient, prints every site with baseline PPD >= 5 mm (highest
P(PNC) first), marks sites at or above the high-risk threshold, and gives
the expected number of PNC sites. The patients are synthetic.

Usage: python demo.py [--high-risk 0.30]
"""

import argparse
from pathlib import Path

from predict_pnc_gpboost import load_models, patient_summary, predict_sites, read_chart

CHART = Path(__file__).resolve().parent / "data" / "synthetic_baseline_chart.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-risk", type=float, default=0.30)
    args = parser.parse_args()

    preds = predict_sites(read_chart(CHART), load_models())
    summary = patient_summary(preds, args.high_risk).set_index("Patient.Code")

    for patient, p in preds.groupby("Patient.Code", sort=False):
        s = summary.loc[patient]
        print(f"\n=== Patient {patient} ===")
        print(f"Sites with baseline PPD >= 5 mm: {int(s.iloc[0])}")
        print(f"Expected number of PNC sites at 3 months: {s.iloc[1]:.1f}")
        print(f"Sites with P(PNC) >= {args.high_risk:.0%}: {int(s.iloc[2])}\n")
        for _, r in p.sort_values("P(PNC)", ascending=False).iterrows():
            flag = "  <- high risk" if r["P(PNC)"] >= args.high_risk else ""
            print(f"Tooth {r['Tooth']} {r['Surface']:<13} baseline {r['Baseline PPD (mm)']:>2.0f} mm: "
                  f"predicted 3-month PPD {r['Predicted 3-month PPD (mm)']:.1f} mm, "
                  f"P(BOP) {r['P(BOP)']:.2f}, P(PNC) {r['P(PNC)']:.2f}{flag}")


if __name__ == "__main__":
    main()
