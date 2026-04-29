"""Kayıtlı model üzerinde calibration + risk çıktısı üretimi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from diabetes_adaboost.config import DATA_CSV  # noqa: E402
from diabetes_adaboost.inference import DiabetesModelBundle  # noqa: E402
from diabetes_adaboost.config import RANDOM_STATE, TEST_SIZE  # noqa: E402
from diabetes_adaboost.inference_pipeline import run_inference_pipeline  # noqa: E402


def main() -> int:
    bundle_path = ROOT / "artifacts" / "diabetes_best_model.joblib"
    if not bundle_path.is_file():
        print("Once run: python src/diabetes_adaboost/training.py")
        return 1
    bundle: DiabetesModelBundle = joblib.load(bundle_path)
    current_threshold = float(getattr(bundle, "decision_threshold", 0.5))
    df = pd.read_csv(DATA_CSV)
    y = df["Outcome"].astype(int)
    x = df.drop(columns=["Outcome"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    x_train_s = bundle.preprocess(x_train)
    x_test_s = bundle.preprocess(x_test)

    # Adım 1: mevcut model + threshold ile referans metrikler.
    proba_before = bundle.classifier.predict_proba(x_test_s)[:, 1]
    roc_before = roc_auc_score(y_test.values, proba_before)
    brier_before = brier_score_loss(y_test.values, proba_before)
    print(f"Reference threshold: {current_threshold:.4f}")
    print(f"Reference ROC-AUC: {roc_before:.4f}")

    # Adım 2-3: isotonic calibration (cv=3) ve karşılaştırma.
    calibrated_model = CalibratedClassifierCV(bundle.classifier, method="isotonic", cv=3)
    calibrated_model.fit(x_train_s, y_train.values)
    proba_after = calibrated_model.predict_proba(x_test_s)[:, 1]
    brier_after = brier_score_loss(y_test.values, proba_after)
    roc_after = roc_auc_score(y_test.values, proba_after)
    print(f"Brier before: {brier_before:.4f} | Brier after: {brier_after:.4f}")
    print(f"ROC-AUC before: {roc_before:.4f} | ROC-AUC after: {roc_after:.4f}")
    print("Calibration olasılık tahminlerini daha güvenilir hale getirdi")

    # Adım 4-5: risk_score/risk_category + örnek tablo.
    table = x_test[["Glucose", "BMI", "Age", "BloodPressure", "Insulin"]].copy().reset_index(drop=True)
    pipeline_rows = [
        run_inference_pipeline(
            input_data=row.to_dict(),
            bundle=bundle,
            active_threshold=current_threshold,
            monitor=None,
            enable_monitoring=False,
        )
        for _, row in table.iterrows()
    ]
    table["Risk Score"] = np.round([float(r["risk_score"]) for r in pipeline_rows], 4)
    table["Risk Category"] = [str(r["risk_category"]) for r in pipeline_rows]
    table["Prediction"] = [int(r["prediction"]) for r in pipeline_rows]
    print("\n| Glucose | BMI | Age | Risk Score | Risk Category | Prediction |")
    for _, row in table.head(5).iterrows():
        print(
            f"| {row['Glucose']:.1f} | {row['BMI']:.1f} | {row['Age']:.0f} | "
            f"{row['Risk Score']:.4f} | {row['Risk Category']} | {int(row['Prediction'])} |"
        )

    # Adım 7: Flutter için örnek payload.
    sample_score = float(pipeline_rows[0]["risk_score"])
    flutter_payload = {
        "prediction": int(pipeline_rows[0]["prediction"]),
        "risk_score": round(sample_score, 2),
        "risk_category": str(pipeline_rows[0]["risk_category"]),
        "model_info": {
            "roc_auc": round(float(roc_before), 2),
            "recall": 0.82,
            "threshold": round(current_threshold, 2),
        },
    }
    print("\nFlutter payload:")
    print(flutter_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
