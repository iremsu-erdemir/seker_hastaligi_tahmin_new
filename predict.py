"""Kayıtlı model ile tek satır tahmin örneği (sunum / doğrulama)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from diabetes_adaboost.config import DATA_CSV  # noqa: E402
from diabetes_adaboost.inference import DiabetesModelBundle  # noqa: E402


def main() -> int:
    bundle_path = ROOT / "artifacts" / "diabetes_best_model.joblib"
    if not bundle_path.is_file():
        print("Once run: python src/diabetes_adaboost/training.py")
        return 1
    bundle: DiabetesModelBundle = joblib.load(bundle_path)
    thr = float(getattr(bundle, "decision_threshold", 0.5))
    df = pd.read_csv(DATA_CSV)
    row = df.iloc[[0]]
    proba = bundle.predict_proba(row)[0, 1]
    pred = bundle.predict(row)[0]
    print(
        "sample row 0:",
        "pred=",
        int(pred),
        "P(outcome=1)=",
        round(float(proba), 4),
        "threshold=",
        round(thr, 4),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
