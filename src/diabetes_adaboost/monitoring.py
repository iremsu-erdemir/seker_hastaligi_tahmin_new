"""Production monitoring utilities for diabetes inference service."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _coerce_binary_label(value: Any) -> int | None:
    """Converts heterogeneous label values to 0/1 when possible."""
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        iv = int(value)
        return iv if iv in (0, 1) else None
    if isinstance(value, (np.floating, float)):
        fv = float(value)
        if np.isnan(fv):
            return None
        if fv in (0.0, 1.0):
            return int(fv)
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"0", "0.0", "false", "no"}:
            return 0
        if text in {"1", "1.0", "true", "yes"}:
            return 1
        try:
            fv = float(text)
        except ValueError:
            return None
        if fv in (0.0, 1.0):
            return int(fv)
    return None


@dataclass
class ThresholdConfig:
    default_threshold: float
    override_enabled: bool
    override_threshold: float | None
    min_threshold: float
    max_threshold: float
    active_threshold_raw: float | None = None

    @property
    def active_threshold(self) -> float:
        if self.active_threshold_raw is not None:
            return float(np.clip(self.active_threshold_raw, self.min_threshold, self.max_threshold))
        if self.override_enabled and self.override_threshold is not None:
            return float(
                np.clip(self.override_threshold, self.min_threshold, self.max_threshold)
            )
        return self.default_threshold


class ModelMonitor:
    """Tracks drift and model health without changing model behavior."""

    def __init__(
        self,
        *,
        baseline_roc_auc: float,
        baseline_brier_score: float,
        baseline_hist: np.ndarray,
        metrics_path: Path,
        drift_log_path: Path,
        state_path: Path,
        max_history: int = 500,
        min_labeled_for_metrics: int = 20,
    ) -> None:
        self.baseline_roc_auc = baseline_roc_auc
        self.baseline_brier_score = baseline_brier_score
        self.baseline_hist = baseline_hist
        self.metrics_path = metrics_path
        self.drift_log_path = drift_log_path
        self.state_path = state_path
        self.max_history = max_history
        self.min_labeled_for_metrics = min_labeled_for_metrics
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"predictions": [], "labeled_predictions": [], "pending_inferences": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {"predictions": [], "labeled_predictions": [], "pending_inferences": {}}
            state.setdefault("predictions", [])
            state.setdefault("labeled_predictions", [])
            state.setdefault("pending_inferences", {})
            return state
        except json.JSONDecodeError:
            return {"predictions": [], "labeled_predictions": [], "pending_inferences": {}}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(_sanitize(self._state), indent=2), encoding="utf-8")

    def _latest_predictions(self) -> np.ndarray:
        values = self._state.get("predictions", [])[-self.max_history :]
        return np.asarray(values, dtype=float) if values else np.asarray([], dtype=float)

    def _latest_labeled(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self._state.get("labeled_predictions", [])[-self.max_history :]
        if not raw:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        labels_buf: list[int] = []
        probs_buf: list[float] = []
        for item in raw:
            lbl = _coerce_binary_label(item.get("label"))
            if lbl is None:
                continue
            try:
                prob = float(item.get("probability"))
            except (TypeError, ValueError):
                continue
            labels_buf.append(lbl)
            probs_buf.append(float(np.clip(prob, 0.0, 1.0)))
        if not labels_buf:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        labels = np.asarray(labels_buf, dtype=int)
        probs = np.asarray(probs_buf, dtype=float)
        return labels, probs

    def _prediction_shift(self, current_probs: np.ndarray) -> float:
        if current_probs.size == 0:
            return 0.0
        current_hist, _ = np.histogram(current_probs, bins=10, range=(0.0, 1.0), density=True)
        current_hist = current_hist / (current_hist.sum() + 1e-9)
        return float(np.abs(current_hist - self.baseline_hist).sum())

    def _risk_distribution(self, probs: np.ndarray) -> dict[str, float]:
        if probs.size == 0:
            return {"low_risk_ratio": 0.0, "medium_risk_ratio": 0.0, "high_risk_ratio": 0.0}
        low = float(np.mean(probs < 0.3))
        medium = float(np.mean((probs >= 0.3) & (probs < 0.6)))
        high = float(np.mean(probs >= 0.6))
        return {
            "low_risk_ratio": round(low, 4),
            "medium_risk_ratio": round(medium, 4),
            "high_risk_ratio": round(high, 4),
        }

    def get_drift_status(self, roc_drift: float, brier_drift: float) -> str:
        if roc_drift > 0.07 or brier_drift > 0.04:
            return "CRITICAL"
        if 0.03 <= roc_drift <= 0.07 or 0.02 <= brier_drift <= 0.04:
            return "WARNING"
        return "OK"

    def detect_drift(
        self, *, roc_auc: float, brier_score: float, distribution_shift: float
    ) -> dict[str, float | str]:
        roc_drift = abs(float(roc_auc) - self.baseline_roc_auc)
        brier_drift = abs(float(brier_score) - self.baseline_brier_score)
        drift_status = self.get_drift_status(roc_drift, brier_drift)
        drift_score = roc_drift + brier_drift + float(distribution_shift)
        return {
            "roc_drift": round(roc_drift, 4),
            "brier_drift": round(brier_drift, 4),
            "drift_status": drift_status,
            "drift_score": round(drift_score, 4),
        }

    def _update_metrics_json(
        self,
        *,
        latest_batch_metrics: dict[str, Any],
        drift_status: str,
        risk_distribution: dict[str, float],
        threshold_used: float,
    ) -> None:
        metrics: dict[str, Any] = {}
        if self.metrics_path.exists():
            try:
                metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        metrics["monitoring"] = {
            "updated_at": _utc_now(),
            "latest_batch": latest_batch_metrics,
            "drift_status": drift_status,
            "risk_distribution": risk_distribution,
            "active_threshold": threshold_used,
        }
        self.metrics_path.write_text(json.dumps(_sanitize(metrics), indent=2), encoding="utf-8")

    def _append_drift_log(
        self,
        *,
        metrics: dict[str, Any],
        drift_score: float,
        threshold_used: float,
        prediction: int | None = None,
        risk_score: float | None = None,
        alert_level: str = "OK",
    ) -> None:
        self.drift_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": _utc_now(),
            "metrics": metrics,
            "drift_score": round(float(drift_score), 6),
            "threshold_used": round(float(threshold_used), 4),
            "prediction": prediction,
            "risk_score": risk_score,
            "drift_status": alert_level,
            "alert_level": alert_level,
        }
        with self.drift_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(_sanitize(payload), ensure_ascii=False) + "\n")

    def summarize(self, *, threshold_used: float) -> dict[str, Any]:
        probs = self._latest_predictions()
        labels, labeled_probs = self._latest_labeled()
        distribution_shift = self._prediction_shift(probs)
        risk_distribution = self._risk_distribution(probs)

        roc_auc = self.baseline_roc_auc
        recall: float | None = None
        brier_score = self.baseline_brier_score
        classes_seen = sorted(int(v) for v in np.unique(labels)) if labels.size else []
        has_labeled_metrics = (
            labels.size >= self.min_labeled_for_metrics and len(np.unique(labels)) >= 2
        )
        if labels.size < self.min_labeled_for_metrics:
            recall_status = "collecting_samples"
        elif len(classes_seen) < 2:
            recall_status = "need_both_classes"
        else:
            recall_status = "ready"
        if has_labeled_metrics:
            roc_auc = float(roc_auc_score(labels, labeled_probs))
            brier_score = float(brier_score_loss(labels, labeled_probs))
            preds = (labeled_probs >= threshold_used).astype(int)
            recall = float(recall_score(labels, preds, zero_division=0))

        drift = self.detect_drift(
            roc_auc=roc_auc, brier_score=brier_score, distribution_shift=distribution_shift
        )
        drift_status = str(drift["drift_status"])
        return {
            "baseline_metrics": {
                "roc_auc": round(self.baseline_roc_auc, 4),
                "brier_score": round(self.baseline_brier_score, 4),
            },
            "current_metrics": {
                "roc_auc": round(roc_auc, 4),
                "recall": round(recall, 4) if recall is not None else None,
                "brier_score": round(brier_score, 4),
                "prediction_distribution_shift": round(distribution_shift, 4),
                "has_labeled_metrics": has_labeled_metrics,
                "labeled_sample_count": int(labels.size),
                "min_labeled_for_metrics": int(self.min_labeled_for_metrics),
                "classes_seen": classes_seen,
                "recall_status": recall_status,
            },
            "drift_status": drift_status,
            "risk_distribution": risk_distribution,
            "drift_score": drift["drift_score"],
            "alert_level": drift_status,
        }

    def record_prediction(self, *, probability: float) -> str:
        p = float(np.clip(probability, 0.0, 1.0))
        self._state.setdefault("predictions", []).append(p)
        self._state["predictions"] = self._state["predictions"][-self.max_history :]
        inference_id = str(uuid.uuid4())
        pending = self._state.setdefault("pending_inferences", {})
        if isinstance(pending, dict):
            pending[inference_id] = {"probability": p, "created_at": _utc_now()}
            pending_items = list(pending.items())[-self.max_history :]
            self._state["pending_inferences"] = {k: v for k, v in pending_items}
        else:
            self._state["pending_inferences"] = {
                inference_id: {"probability": p, "created_at": _utc_now()}
            }
        self._save_state()
        return inference_id

    def record_feedback(
        self, *, inference_id: str, observed_label: int, threshold_used: float
    ) -> dict[str, Any]:
        normalized_label = _coerce_binary_label(observed_label)
        if normalized_label is None:
            raise ValueError("observed_label must be 0 or 1")
        pending = self._state.setdefault("pending_inferences", {})
        entry = pending.pop(inference_id, None) if isinstance(pending, dict) else None
        if entry is None:
            raise KeyError("inference_id not found")
        probability = float(np.clip(float(entry.get("probability", 0.0)), 0.0, 1.0))
        self._state.setdefault("labeled_predictions", []).append(
            {"label": int(normalized_label), "probability": probability}
        )
        self._state["labeled_predictions"] = self._state["labeled_predictions"][
            -self.max_history :
        ]
        self._save_state()
        summary = self.summarize(threshold_used=threshold_used)
        current_metrics = summary["current_metrics"]
        baseline_metrics = summary["baseline_metrics"]
        drift_status = summary["drift_status"]
        risk_distribution = summary["risk_distribution"]
        drift_score = summary["drift_score"]

        self._update_metrics_json(
            latest_batch_metrics=current_metrics,
            drift_status=drift_status,
            risk_distribution=risk_distribution,
            threshold_used=threshold_used,
        )
        self._append_drift_log(
            metrics={
                "event": "feedback_ingested",
                "inference_id": inference_id,
                "observed_label": int(normalized_label),
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "risk_distribution": risk_distribution,
                "drift_status": drift_status,
            },
            drift_score=float(drift_score),
            threshold_used=threshold_used,
            prediction=None,
            risk_score=probability,
            alert_level=str(summary.get("alert_level", drift_status)),
        )
        return summary

    def record_inference(
        self,
        *,
        probability: float,
        threshold_used: float,
        observed_label: int | None = None,
        prediction: int | None = None,
    ) -> dict[str, Any]:
        p = float(np.clip(probability, 0.0, 1.0))
        inference_id = self.record_prediction(probability=p)
        normalized_label = _coerce_binary_label(observed_label)
        if normalized_label in (0, 1):
            return self.record_feedback(
                inference_id=inference_id,
                observed_label=int(normalized_label),
                threshold_used=threshold_used,
            )

        summary = self.summarize(threshold_used=threshold_used)
        current_metrics = summary["current_metrics"]
        baseline_metrics = summary["baseline_metrics"]
        drift_status = summary["drift_status"]
        risk_distribution = summary["risk_distribution"]
        drift_score = summary["drift_score"]

        self._update_metrics_json(
            latest_batch_metrics=current_metrics,
            drift_status=drift_status,
            risk_distribution=risk_distribution,
            threshold_used=threshold_used,
        )
        self._append_drift_log(
            metrics={
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "risk_distribution": risk_distribution,
                "drift_status": drift_status,
            },
            drift_score=float(drift_score),
            threshold_used=threshold_used,
            prediction=prediction,
            risk_score=p,
            alert_level=str(summary.get("alert_level", drift_status)),
        )
        return summary

    def log_threshold_change(
        self, *, threshold_used: float, override_enabled: bool, source: str = "api"
    ) -> None:
        self._append_drift_log(
            metrics={
                "event": "threshold_update",
                "override_enabled": override_enabled,
                "source": source,
            },
            drift_score=0.0,
            threshold_used=threshold_used,
            alert_level="OK",
        )


def read_threshold_config(path: Path, fallback_default: float = 0.4228) -> ThresholdConfig:
    if not path.exists():
        return ThresholdConfig(
            default_threshold=fallback_default,
            override_enabled=False,
            override_threshold=None,
            min_threshold=0.35,
            max_threshold=0.50,
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw = {}
    return ThresholdConfig(
        default_threshold=_safe_float(raw.get("default_threshold"), fallback_default),
        override_enabled=bool(raw.get("override_enabled", False)),
        override_threshold=(
            _safe_float(raw.get("override_threshold"))
            if raw.get("override_threshold") is not None
            else None
        ),
        min_threshold=_safe_float(raw.get("min_threshold"), 0.35),
        max_threshold=_safe_float(raw.get("max_threshold"), 0.50),
        active_threshold_raw=(
            _safe_float(raw.get("active_threshold"))
            if raw.get("active_threshold") is not None
            else None
        ),
    )


def write_threshold_config(path: Path, cfg: ThresholdConfig) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_threshold": round(float(cfg.default_threshold), 4),
        "override_enabled": bool(cfg.override_enabled),
        "override_threshold": (
            round(float(cfg.override_threshold), 4) if cfg.override_threshold is not None else None
        ),
        "active_threshold": round(float(cfg.active_threshold), 4),
        "min_threshold": round(float(cfg.min_threshold), 2),
        "max_threshold": round(float(cfg.max_threshold), 2),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
