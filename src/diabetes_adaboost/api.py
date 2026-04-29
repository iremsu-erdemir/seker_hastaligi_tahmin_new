"""Diabetes prediction için FastAPI servisi.

Bu servis:
- Eğitimde üretilen joblib model paketini yükler
- Tahmin için eğitimdeki preprocessing ile %100 aynı akışı uygular
- Flutter uygulamasına dinamik tahmin ve metrik verisi sağlar
"""

from __future__ import annotations

import json
import hashlib
import sys
import threading
from typing import Annotated
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import DATA_CSV, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
from .inference import DiabetesModelBundle
from .inference_pipeline import run_inference_pipeline
from .monitoring import ModelMonitor, ThresholdConfig, read_threshold_config, write_threshold_config

matplotlib.use("Agg")

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
CHARTS_DIR = PROJECT_ROOT / "flutter_app" / "assets" / "charts"
THRESHOLD_CONFIG_PATH = PROJECT_ROOT / "config" / "threshold_config.json"
DRIFT_LOG_PATH = PROJECT_ROOT / "logs" / "drift_log.jsonl"
MONITORING_STATE_PATH = ARTIFACTS_DIR / "monitoring_state.json"
_MONITOR: ModelMonitor | None = None
_CHART_RENDER_LOCK = threading.Lock()
_DYNAMIC_CHART_FALLBACKS = [
    "roc_best_model.png",
    "pr_best_model.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "model_comparison.png",
    "threshold.png",
    "eda_glucose_hist.png",
    "eda_bmi_hist.png",
    "eda_age_hist.png",
    "eda_outcome_distribution.png",
    "eda_correlation_heatmap.png",
    "eda_glucose_vs_outcome_box.png",
    "eda_bmi_vs_outcome_box.png",
]


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
    Outcome: int | None = Field(default=None, ge=0, le=1)


class PredictResponse(BaseModel):
    """Tahmin sonucunun istemciye dönülecek formatı."""

    prediction: int
    risk_score: float
    risk_category: str
    risk: float
    class_id: int = Field(..., alias="class")
    threshold: float
    model_info: dict[str, float]
    explanation: list[str]
    top_contributors: list[dict[str, Any]]
    model_health: dict[str, Any]
    drift_status: str
    risk_distribution: dict[str, float]
    active_threshold: float
    alert_level: str
    inference_id: str | None = None


class FeedbackRequest(BaseModel):
    inference_id: str = Field(..., min_length=10)
    Outcome: int = Field(..., ge=0, le=1)


class FeedbackResponse(BaseModel):
    accepted: bool
    inference_id: str
    model_health: dict[str, Any]


class ThresholdConfigResponse(BaseModel):
    default_threshold: float
    override_enabled: bool
    active_threshold: float
    min_threshold: float
    max_threshold: float


class ThresholdConfigUpdateRequest(BaseModel):
    threshold: float = Field(..., ge=0.35, le=0.50)
    override_enabled: bool = True


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


def _active_threshold(bundle: DiabetesModelBundle) -> float:
    cfg = read_threshold_config(THRESHOLD_CONFIG_PATH, fallback_default=0.4228)
    return float(cfg.active_threshold)


def _threshold_response(cfg: ThresholdConfig) -> dict[str, Any]:
    return {
        "default_threshold": round(float(cfg.default_threshold), 4),
        "override_enabled": bool(cfg.override_enabled),
        "active_threshold": round(float(cfg.active_threshold), 4),
        "min_threshold": round(float(cfg.min_threshold), 2),
        "max_threshold": round(float(cfg.max_threshold), 2),
    }


def _json_sanitize(obj: Any) -> Any:
    """NumPy tiplerini JSON ile uyumlu yerel tiplere dönüştürür."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_sanitize(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _humanize_chart_title(filename: str) -> str:
    """Chart dosya adından okunabilir bir başlık üretir."""
    stem = Path(filename).stem
    parts = stem.replace("-", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p)


def _list_chart_assets() -> list[dict[str, str]]:
    """Flutter assets/charts klasöründeki tüm png grafikleri listeler."""
    assets: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    chart_names: list[str] = []
    if CHARTS_DIR.exists():
        chart_names.extend(path.name for path in sorted(CHARTS_DIR.glob("*.png")))
    if not chart_names:
        chart_names.extend(_DYNAMIC_CHART_FALLBACKS)
    # Aynı korelasyon grafiğinin eski/yeni adlarını birlikte göstermeyelim.
    if "eda_correlation_heatmap.png" in chart_names and "heatmap_correlation.png" in chart_names:
        chart_names = [name for name in chart_names if name != "heatmap_correlation.png"]
    unique_names = sorted(set(chart_names))
    for chart_name in unique_names:
        chart_path = CHARTS_DIR / chart_name
        if chart_path.exists():
            digest = hashlib.sha256(chart_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                # Aynı görselin farklı isimle gelmesi durumunda tek kart göster.
                continue
            seen_hashes.add(digest)
        stem = Path(chart_name).stem.lower()
        if stem.startswith("eda_"):
            category = "eda"
        elif "roc" in stem or "pr_" in stem or "confusion" in stem or "threshold" in stem:
            category = "performance"
        elif "feature_importance" in stem or "model_comparison" in stem:
            category = "model"
        else:
            category = "analysis"
        assets.append(
            {
                "title": _humanize_chart_title(chart_name),
                "asset_path": f"assets/charts/{chart_name}",
                "category": category,
            }
        )
    return assets


def _test_split_scores(bundle: DiabetesModelBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_CSV)
    y = df["Outcome"].astype(int)
    x = df.drop(columns=["Outcome"])
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_score = bundle.predict_proba(x_test)[:, 1]
    return y_test.to_numpy(dtype=int), y_score, x_test.to_numpy(dtype=float)


def _fig_to_png_bytes(fig: Any) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _dynamic_chart_bytes(filename: str, bundle: DiabetesModelBundle) -> bytes | None:
    name = Path(filename).name.lower()
    threshold = _active_threshold(bundle)
    metrics = _read_metrics_json()

    if "roc" in name:
        y_true, y_score, _ = _test_split_scores(bundle)
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="Live ROC")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax.set_title(f"ROC Curve (active threshold={threshold:.3f})")
        ax.legend(loc="lower right")
        return _fig_to_png_bytes(fig)

    if "pr" in name:
        y_true, y_score, _ = _test_split_scores(bundle)
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="Live PR")
        ax.set_title(f"PR Curve (active threshold={threshold:.3f})")
        ax.legend(loc="lower left")
        return _fig_to_png_bytes(fig)

    if "confusion" in name:
        y_true, y_score, _ = _test_split_scores(bundle)
        y_pred = (y_score >= threshold).astype(np.int64)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_title(f"Confusion Matrix (thr={threshold:.3f})")
        return _fig_to_png_bytes(fig)

    if "feature_importance" in name:
        items = _feature_importance_for_bundle(bundle)[:10]
        fig, ax = plt.subplots(figsize=(7, 5))
        feats = [it["feature"] for it in items][::-1]
        vals = [float(it["importance"]) for it in items][::-1]
        ax.barh(feats, vals, color="#4C78A8")
        ax.set_title("Feature Importance (live model)")
        return _fig_to_png_bytes(fig)

    if "model_comparison" in name:
        models = metrics.get("models", [])
        if not models:
            return None
        labels = [m.get("name", "Model") for m in models]
        roc_vals = [float(m.get("test_roc_auc", 0.0)) for m in models]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, roc_vals, color="#72B7B2")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Model Comparison (ROC-AUC)")
        ax.tick_params(axis="x", rotation=35)
        return _fig_to_png_bytes(fig)

    if "threshold" in name:
        y_true, y_score, _ = _test_split_scores(bundle)
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, p[:-1], label="Precision")
        ax.plot(t, r[:-1], label="Recall")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Active {threshold:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold vs Precision/Recall")
        ax.legend(loc="best")
        return _fig_to_png_bytes(fig)

    df = pd.read_csv(DATA_CSV)
    if "eda_glucose_hist" in name:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Glucose"].dropna(), bins=30, color="#4C78A8")
        ax.set_title("Glucose Distribution")
        return _fig_to_png_bytes(fig)
    if "eda_bmi_hist" in name:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["BMI"].dropna(), bins=30, color="#F58518")
        ax.set_title("BMI Distribution")
        return _fig_to_png_bytes(fig)
    if "eda_age_hist" in name:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Age"].dropna(), bins=30, color="#54A24B")
        ax.set_title("Age Distribution")
        return _fig_to_png_bytes(fig)
    if "eda_outcome_distribution" in name:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Outcome"].value_counts().sort_index()
        ax.bar([str(int(i)) for i in counts.index], counts.values, color="#E45756")
        ax.set_title("Outcome Distribution")
        return _fig_to_png_bytes(fig)
    if "eda_correlation_heatmap" in name:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(
                    j,
                    i,
                    f"{corr.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )
        ax.set_title("Correlation Heatmap")
        return _fig_to_png_bytes(fig)
    if "eda_glucose_vs_outcome_box" in name:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped = [df[df["Outcome"] == c]["Glucose"].dropna() for c in [0, 1]]
        ax.boxplot(grouped, labels=["0", "1"])
        ax.set_title("Glucose by Outcome")
        return _fig_to_png_bytes(fig)
    if "eda_bmi_vs_outcome_box" in name:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped = [df[df["Outcome"] == c]["BMI"].dropna() for c in [0, 1]]
        ax.boxplot(grouped, labels=["0", "1"])
        ax.set_title("BMI by Outcome")
        return _fig_to_png_bytes(fig)

    return None


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


def _evaluate_bundle_at_threshold(bundle: DiabetesModelBundle, threshold: float) -> dict[str, Any]:
    """Modeli sabit test split üzerinde verilen eşikle değerlendirir."""
    df = pd.read_csv(DATA_CSV)
    y = df["Outcome"].astype(int)
    x = df.drop(columns=["Outcome"])
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    proba = bundle.predict_proba(x_test)[:, 1]
    y_pred = (proba >= threshold).astype(np.int64)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report_txt = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "confusion_matrix": _json_sanitize(confusion_matrix(y_test, y_pred).tolist()),
        "classification_report": report_txt,
        "classification_report_dict": _json_sanitize(report_dict),
    }


def _create_monitor(bundle: DiabetesModelBundle) -> ModelMonitor:
    global _MONITOR
    if _MONITOR is not None:
        return _MONITOR
    raw = _read_metrics_json()
    baseline_roc_auc = float(
        raw.get("best_model_test_metrics", {}).get("test_roc_auc", 0.0)
    )
    baseline_brier = float(raw.get("calibration", {}).get("brier_score_after", 0.0))
    df = pd.read_csv(DATA_CSV)
    y = df["Outcome"].astype(int)
    x = df.drop(columns=["Outcome"])
    _, x_test, _, _ = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    reference_probs = bundle.predict_proba(x_test)[:, 1]
    reference_hist, _ = np.histogram(reference_probs, bins=10, range=(0.0, 1.0), density=True)
    reference_hist = reference_hist / (reference_hist.sum() + 1e-9)
    _MONITOR = ModelMonitor(
        baseline_roc_auc=baseline_roc_auc,
        baseline_brier_score=baseline_brier,
        baseline_hist=reference_hist,
        metrics_path=METRICS_PATH,
        drift_log_path=DRIFT_LOG_PATH,
        state_path=MONITORING_STATE_PATH,
    )
    return _MONITOR


@app.get("/health")
def health() -> dict[str, str]:
    """Canlılık kontrol endpoint'i."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict[str, Any]:
    """Gerçek zamanlı risk tahmini endpoint'i."""
    bundle = _load_bundle()

    threshold = _active_threshold(bundle)
    pipeline_output = run_inference_pipeline(
        input_data={
            "Glucose": payload.Glucose,
            "BMI": payload.BMI,
            "Age": payload.Age,
            "BloodPressure": payload.BloodPressure,
            "Insulin": payload.Insulin,
        },
        bundle=bundle,
        active_threshold=threshold,
        monitor=_create_monitor(bundle),
        observed_label=None,
        enable_monitoring=True,
    )
    metrics_raw = _read_metrics_json()
    best_metrics = metrics_raw.get("best_model_test_metrics", {})
    model_info = {
        "roc_auc": float(best_metrics.get("test_roc_auc", 0.0)),
        "recall": float(best_metrics.get("test_recall_macro", 0.0)),
        "threshold": threshold,
    }
    explanations = _build_explanations(payload, float(pipeline_output["risk_score"]))
    contributors = _feature_importance_for_bundle(bundle)[:5]
    monitoring = pipeline_output["monitoring"]

    return {
        "prediction": pipeline_output["prediction"],
        "risk_score": pipeline_output["risk_score"],
        "risk_category": pipeline_output["risk_category"],
        "risk": pipeline_output["risk"],
        "class": pipeline_output["class"],
        "threshold": round(threshold, 4),
        "active_threshold": round(threshold, 4),
        "model_info": model_info,
        "explanation": explanations,
        "top_contributors": contributors,
        "model_health": monitoring["current_metrics"],
        "drift_status": monitoring["drift_status"],
        "alert_level": monitoring["alert_level"],
        "risk_distribution": monitoring["risk_distribution"],
        "inference_id": pipeline_output.get("inference_id"),
    }


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> dict[str, Any]:
    """Predict sonrası ground-truth etiketi ingest eder."""
    bundle = _load_bundle()
    threshold = _active_threshold(bundle)
    monitor = _create_monitor(bundle)
    try:
        summary = monitor.record_feedback(
            inference_id=payload.inference_id,
            observed_label=payload.Outcome,
            threshold_used=threshold,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="inference_id bulunamadi") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "accepted": True,
        "inference_id": payload.inference_id,
        "model_health": summary["current_metrics"],
    }


@app.get("/metrics")
def metrics(
    threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> dict[str, Any]:
    """Model metrikleri ve feature importance endpoint'i."""
    bundle = _load_bundle()
    raw = _read_metrics_json()
    threshold_raw = _active_threshold(bundle)
    active_threshold = float(threshold if threshold is not None else threshold_raw)
    eval_metrics = _evaluate_bundle_at_threshold(bundle, active_threshold)
    model_name = raw.get("best_model_name", type(bundle.classifier).__name__)

    # Metrics endpoint'inde hem skor hem de importance dönüyoruz.
    return {
        "accuracy": eval_metrics["accuracy"],
        "f1": eval_metrics["f1_macro"],
        "roc_auc": eval_metrics["roc_auc"],
        "balanced_accuracy": eval_metrics["balanced_accuracy"],
        "precision_macro": eval_metrics["precision_macro"],
        "recall_macro": eval_metrics["recall_macro"],
        "brier_score": eval_metrics["brier_score"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "threshold": active_threshold,
        "model_name": model_name,
        "feature_importance": _feature_importance_for_bundle(bundle)[:10],
        "models": raw.get("models", []),
        "preprocessing": raw.get("preprocessing", []),
        "classification_report_test_best": eval_metrics["classification_report"],
        "classification_report_test_best_dict": eval_metrics["classification_report_dict"],
        "feature_importance_by_model": raw.get("feature_importance", {}),
        "cv_train_roc_auc_mean": float(raw.get("cv_train_roc_auc_mean", 0.0)),
        "cv_train_roc_auc_std": float(raw.get("cv_train_roc_auc_std", 0.0)),
        "generated_at": raw.get("generated_at", ""),
        "charts": _list_chart_assets(),
        "clinical_optimization": raw.get("clinical_optimization", {}),
        "calibration": raw.get("calibration", {}),
        "monitoring": raw.get("monitoring", {}),
    }


@app.get("/charts/{filename}")
def chart_image(filename: str) -> FileResponse:
    """Grafikleri canlı üretir; yoksa statik dosyayı döndürür."""
    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Sadece .png grafik destekleniyor.")

    bundle = _load_bundle()
    with _CHART_RENDER_LOCK:
        dynamic_png = _dynamic_chart_bytes(safe_name, bundle=bundle)
    if dynamic_png is not None:
        return StreamingResponse(BytesIO(dynamic_png), media_type="image/png")

    chart_path = (CHARTS_DIR / safe_name).resolve()
    if chart_path.exists():
        return FileResponse(chart_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Grafik bulunamadi.")


@app.get("/model-health")
def model_health() -> dict[str, Any]:
    """Modelin son batch sağlık durumunu döndürür."""
    bundle = _load_bundle()
    threshold = _active_threshold(bundle)
    monitor = _create_monitor(bundle)
    summary = monitor.summarize(threshold_used=threshold)
    calibration = _read_metrics_json().get("calibration", {})
    return {
        "roc_auc": summary["current_metrics"]["roc_auc"],
        "recall": summary["current_metrics"]["recall"],
        "brier_score": summary["current_metrics"]["brier_score"],
        "drift_status": summary["drift_status"],
        "alert_level": summary.get("alert_level", summary["drift_status"]),
        "active_threshold": round(threshold, 4),
        "calibration_status": calibration.get("method", "unknown"),
        "risk_distribution": summary["risk_distribution"],
    }


@app.get("/threshold-config", response_model=ThresholdConfigResponse)
def get_threshold_config() -> dict[str, Any]:
    cfg = read_threshold_config(THRESHOLD_CONFIG_PATH, fallback_default=0.4228)
    return _threshold_response(cfg)


@app.put("/threshold-config", response_model=ThresholdConfigResponse)
def update_threshold_config(payload: ThresholdConfigUpdateRequest) -> dict[str, Any]:
    cfg = read_threshold_config(THRESHOLD_CONFIG_PATH, fallback_default=0.4228)
    cfg.override_enabled = bool(payload.override_enabled)
    cfg.override_threshold = float(payload.threshold)
    saved = write_threshold_config(THRESHOLD_CONFIG_PATH, cfg)

    bundle = _load_bundle()
    monitor = _create_monitor(bundle)
    monitor.log_threshold_change(
        threshold_used=float(saved["active_threshold"]),
        override_enabled=bool(saved["override_enabled"]),
        source="threshold-config-endpoint",
    )
    return {
        "default_threshold": float(saved["default_threshold"]),
        "override_enabled": bool(saved["override_enabled"]),
        "active_threshold": float(saved["active_threshold"]),
        "min_threshold": float(saved["min_threshold"]),
        "max_threshold": float(saved["max_threshold"]),
    }

