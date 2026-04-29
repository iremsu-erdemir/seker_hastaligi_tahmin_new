"""Single-source inference pipeline for API/CLI usage."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    # Package usage (e.g. `python -m src.diabetes_adaboost.api`)
    from .inference import DiabetesModelBundle
    from .monitoring import ModelMonitor
except ImportError:  # pragma: no cover - direct script/debug execution fallback
    # Direct file execution fallback in IDE/debugger.
    from inference import DiabetesModelBundle
    from monitoring import ModelMonitor


def risk_category_from_score(score: float) -> str:
    if score < 0.3:
        return "Düşük Risk"
    if score < 0.6:
        return "Orta Risk"
    return "Yüksek Risk"


def run_inference_pipeline(
    *,
    input_data: pd.DataFrame | dict[str, Any],
    bundle: DiabetesModelBundle,
    active_threshold: float,
    monitor: ModelMonitor | None = None,
    observed_label: int | None = None,
    enable_monitoring: bool = True,
) -> dict[str, Any]:
    """Runs preprocess -> predict_proba -> threshold -> risk -> monitoring."""
    row = input_data.copy() if isinstance(input_data, pd.DataFrame) else pd.DataFrame([input_data])

    for col in bundle.feature_columns:
        if col not in row.columns:
            row[col] = bundle.medians.get(col, 0.0)

    # bundle.predict_proba internally applies existing preprocessing and calibrated model.
    proba = bundle.predict_proba(row)[:, 1]
    risk_score = float(proba[0])
    threshold = float(active_threshold)
    pred_class = int(risk_score >= threshold)
    risk_category = risk_category_from_score(risk_score)

    monitoring_summary: dict[str, Any] = {
        "baseline_metrics": {},
        "current_metrics": {},
        "drift_status": "OK",
        "risk_distribution": {
            "low_risk_ratio": 0.0,
            "medium_risk_ratio": 0.0,
            "high_risk_ratio": 0.0,
        },
        "drift_score": 0.0,
        "alert_level": "OK",
    }
    inference_id: str | None = None
    if enable_monitoring and monitor is not None:
        inference_id = monitor.record_prediction(probability=risk_score)
        if observed_label is not None:
            monitoring_summary = monitor.record_feedback(
                inference_id=inference_id,
                observed_label=observed_label,
                threshold_used=threshold,
            )
        else:
            monitoring_summary = monitor.summarize(threshold_used=threshold)

    return {
        "inference_id": inference_id,
        "prediction": pred_class,
        "risk_score": round(risk_score, 4),
        "risk": round(risk_score, 4),
        "risk_category": risk_category,
        "class": pred_class,
        "threshold": round(threshold, 4),
        "monitoring": monitoring_summary,
    }
