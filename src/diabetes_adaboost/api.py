"""Diabetes prediction için FastAPI servisi.

Bu servis:
- Eğitimde üretilen joblib model paketini yükler
- Tahmin için eğitimdeki preprocessing ile %100 aynı akışı uygular
- Flutter uygulamasına dinamik tahmin ve metrik verisi sağlar
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT
from .inference import DiabetesModelBundle

# Joblib ile yüklenen model, `diabetes_adaboost.*` modül yolunu referans alır.
# API `src.diabetes_adaboost.api` olarak başlatıldığında bu paket yolu için
# `src` klasörünü import path'e ekliyoruz.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Eğitim artefaktları için varsayılan yollar.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "diabetes_best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "flutter_app" / "assets" / "metrics.json"


# FastAPI uygulamasını oluşturuyoruz.
app = FastAPI(
    title="Diabetes Decision Support API",
    description="Gerçek zamanlı diyabet risk tahmini ve model metrikleri servisi",
    version="1.0.0",
)

# Flutter web (localhost) istemcisinin API'ye erişebilmesi için CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Flutter istemcisinden gelecek tahmin girdisi."""

    Glucose: float = Field(..., gt=0)
    BMI: float = Field(..., gt=0)
    Age: float = Field(..., gt=0)
    BloodPressure: float = Field(..., gt=0)
    Insulin: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    """Tahmin sonucunun istemciye dönülecek formatı."""

    risk: float
    class_id: int = Field(..., alias="class")
    threshold: float
    explanation: list[str]
    top_contributors: list[dict[str, Any]]


def _load_bundle() -> DiabetesModelBundle:
    """Kaydedilmiş model paketini güvenli şekilde yükler."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model dosyasi bulunamadi: {MODEL_PATH}. "
                "Lutfen once training scriptini calistirin."
            ),
        )
    bundle = joblib.load(MODEL_PATH)

    # Pickle/unpickle sırasında modül yolu farklılıkları (src.diabetes_adaboost vs
    # diabetes_adaboost) sınıf kimliğini değiştirebilir. Bu yüzden katı isinstance
    # yerine API'nin gerçekten ihtiyaç duyduğu alan/metotları doğruluyoruz.
    required_attrs = ("classifier", "feature_columns", "medians", "decision_threshold")
    required_methods = ("predict", "predict_proba")
    if not all(hasattr(bundle, attr) for attr in required_attrs) or not all(
        callable(getattr(bundle, method, None)) for method in required_methods
    ):
        raise HTTPException(status_code=500, detail="Model paketi beklenen formatta degil.")

    return bundle


def _read_metrics_json() -> dict[str, Any]:
    """Training çıktısı metrics.json dosyasını okur."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _bmi_level_text(bmi: float) -> str:
    """BMI için insan tarafından okunabilir kategori."""
    if bmi >= 30:
        return "BMI obez aralikta"
    if bmi >= 25:
        return "BMI kilolu aralikta"
    return "BMI normal aralikta"


def _build_explanations(payload: PredictRequest, risk: float) -> list[str]:
    """Temel tıbbi sezgilerle kısa açıklama maddeleri üretir."""
    explanations: list[str] = []
    if payload.Glucose >= 126:
        explanations.append("Glucose yuksek")
    elif payload.Glucose >= 100:
        explanations.append("Glucose sinirda yuksek")

    explanations.append(_bmi_level_text(payload.BMI))

    if payload.Age >= 45:
        explanations.append("Age risk artiriyor")
    elif payload.Age >= 35:
        explanations.append("Age orta seviye risk faktorlerinden")

    if payload.BloodPressure >= 80:
        explanations.append("BloodPressure yuksek seyredebiliyor")
    if payload.Insulin > 160:
        explanations.append("Insulin seviyesi yuksek")

    # Çok kısa kalırsa kullanıcıya bağlamsal bir yorum ekliyoruz.
    if len(explanations) < 2:
        if risk >= 0.6:
            explanations.append("Birden fazla ozellik riski yukseltiyor")
        elif risk >= 0.4:
            explanations.append("Sinirda risk profili gozlemleniyor")
        else:
            explanations.append("Girdiler dusuk risk profiline daha yakin")
    return explanations


def _feature_importance_for_bundle(bundle: DiabetesModelBundle) -> list[dict[str, Any]]:
    """Modelin feature importance bilgisini standard bir formda döndürür."""
    clf = bundle.classifier
    names = bundle.feature_columns

    # Tree tabanlı modellerde doğrudan feature_importances_ bulunur.
    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(clf.feature_importances_, dtype=float)
    # Lineer modellerde katsayı mutlak değeri kullanılır.
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        vals = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    # Hibrit voting için alt modellerin önem skorlarını ortalama alıyoruz.
    elif hasattr(clf, "estimators"):
        stacked: list[np.ndarray] = []
        for est in getattr(clf, "estimators", []):
            if hasattr(est, "feature_importances_"):
                stacked.append(np.asarray(est.feature_importances_, dtype=float))
            elif hasattr(est, "coef_"):
                ec = np.asarray(est.coef_, dtype=float)
                stacked.append(np.abs(ec[0]) if ec.ndim > 1 else np.abs(ec))
        vals = np.mean(stacked, axis=0) if stacked else np.zeros(len(names))
    else:
        vals = np.zeros(len(names))

    # Top-N özelliği büyükten küçüğe sıralıyoruz.
    pairs = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(names, vals, strict=False)
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)
    return pairs


@app.get("/health")
def health() -> dict[str, str]:
    """Canlılık kontrol endpoint'i."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict[str, Any]:
    """Gerçek zamanlı risk tahmini endpoint'i."""
    bundle = _load_bundle()

    # Tek satırlık girdiyi DataFrame'e çevirip bundle preprocess akışına veriyoruz.
    row = pd.DataFrame(
        [
            {
                "Glucose": payload.Glucose,
                "BMI": payload.BMI,
                "Age": payload.Age,
                "BloodPressure": payload.BloodPressure,
                "Insulin": payload.Insulin,
            }
        ]
    )

    # Eksik kalan sütunlar varsa eğitimdeki medyanlarla güvenli doldurma için ekliyoruz.
    for col in bundle.feature_columns:
        if col not in row.columns:
            row[col] = bundle.medians.get(col, 0.0)

    # Eğitimde kullanılan aynı preprocessing + scaler + model akışını bundle yapıyor.
    proba = bundle.predict_proba(row)[:, 1]
    risk = float(proba[0])
    threshold = float(bundle.decision_threshold)
    pred_class = int(risk >= threshold)

    explanations = _build_explanations(payload, risk)
    contributors = _feature_importance_for_bundle(bundle)[:5]

    return {
        "risk": round(risk, 4),
        "class": pred_class,
        "threshold": round(threshold, 4),
        "explanation": explanations,
        "top_contributors": contributors,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Model metrikleri ve feature importance endpoint'i."""
    bundle = _load_bundle()
    raw = _read_metrics_json()

    best_metrics = raw.get("best_model_test_metrics", {})
    accuracy = best_metrics.get("test_accuracy", raw.get("accuracy", 0.0))
    f1_macro = best_metrics.get("test_f1_macro", raw.get("f1", 0.0))
    roc_auc = best_metrics.get("test_roc_auc", raw.get("roc_auc", 0.0))
    model_name = raw.get("best_model_name", type(bundle.classifier).__name__)

    # Metrics endpoint'inde hem skor hem de importance dönüyoruz.
    return {
        "accuracy": float(accuracy),
        "f1": float(f1_macro),
        "roc_auc": float(roc_auc),
        "threshold": float(raw.get("decision_threshold", {}).get("threshold", bundle.decision_threshold)),
        "model_name": model_name,
        "feature_importance": _feature_importance_for_bundle(bundle)[:10],
    }

