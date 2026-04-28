"""Özellik mühendisliği: winsorization, log1p, etkileşim ve BMI kategorisi."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Eğitimde hesaplanan dilimler; log1p bu sütunlarda winsor sonrası uygulanır.
WINSOR_LOG_COLUMNS = ("Insulin", "DiabetesPedigreeFunction")


def _bmi_category(bmi: pd.Series) -> pd.Series:
    """WHO eşikleri: 0=Normal (<25), 1=Kilolu [25,30), 2=Obez (>=30)."""
    b = bmi.astype(float)
    cat = pd.Series(0, index=b.index, dtype=np.int64)
    cat[b >= 25.0] = 1
    cat[b >= 30.0] = 2
    return cat


def winsor_bounds_from_train(
    X_train: pd.DataFrame,
    columns: tuple[str, ...] = WINSOR_LOG_COLUMNS,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Eğitim kümesinden %1 / %99 sınırları."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        if col not in X_train.columns:
            continue
        lo = float(X_train[col].quantile(low_q))
        hi = float(X_train[col].quantile(high_q))
        if lo > hi:
            lo, hi = hi, lo
        bounds[col] = (lo, hi)
    return bounds


def apply_winsorize(X: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def apply_log1p_columns(X: pd.DataFrame, columns: tuple[str, ...] = WINSOR_LOG_COLUMNS) -> pd.DataFrame:
    out = X.copy()
    for col in columns:
        if col in out.columns:
            out[col] = np.log1p(out[col].astype(float))
    return out


def add_interaction_and_bmi_category(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    if "Glucose" in out.columns and "Age" in out.columns:
        out["Glucose_Age_Interaction"] = out["Glucose"].astype(float) * out["Age"].astype(float)
    if "BMI" in out.columns:
        out["BMI_Category"] = _bmi_category(out["BMI"])
    return out


def engineer_features_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    winsor_columns: tuple[str, ...] = WINSOR_LOG_COLUMNS,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Medyan doldurma sonrası çağrılır.
    Winsor sınırları yalnızca eğitimden; test aynı sınırlarla kırpılır.
    """
    bounds = winsor_bounds_from_train(X_train, columns=winsor_columns, low_q=low_q, high_q=high_q)
    Xt = apply_winsorize(X_train, bounds)
    Xs = apply_winsorize(X_test, bounds)
    Xt = apply_log1p_columns(Xt, columns=winsor_columns)
    Xs = apply_log1p_columns(Xs, columns=winsor_columns)
    Xt = add_interaction_and_bmi_category(Xt)
    Xs = add_interaction_and_bmi_category(Xs)
    meta = {
        "winsor_columns": list(winsor_columns),
        "winsor_quantiles": {"low": low_q, "high": high_q},
        "winsor_bounds": bounds,
    }
    return Xt, Xs, meta
