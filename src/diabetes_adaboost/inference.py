"""Kaydedilmiş model paketinden tahmin (tek veya çoklu satır)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import COLUMNS_IMPUTE, COLUMNS_ZERO_TO_NA
from .feature_engineering import (
    WINSOR_LOG_COLUMNS,
    add_interaction_and_bmi_category,
    apply_log1p_columns,
    apply_winsorize,
)


@dataclass
class DiabetesModelBundle:
    """training.py tarafından joblib ile kaydedilen nesne yapısı."""

    feature_columns: list[str]
    medians: dict[str, float]
    scaler: Any
    classifier: Any
    zero_to_na_cols: list[str] | None = None
    impute_cols: list[str] | None = None
    winsor_bounds: dict[str, tuple[float, float]] | None = None
    winsor_columns: tuple[str, ...] = WINSOR_LOG_COLUMNS
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.zero_to_na_cols = self.zero_to_na_cols or list(COLUMNS_ZERO_TO_NA)
        self.impute_cols = self.impute_cols or list(COLUMNS_IMPUTE)
        self.winsor_bounds = self.winsor_bounds or {}

    def __getattr__(self, name: str) -> Any:
        """Eski joblib paketleri için geriye dönük alanlar."""
        if name == "decision_threshold":
            return 0.5
        if name == "winsor_bounds":
            return {}
        if name == "winsor_columns":
            return WINSOR_LOG_COLUMNS
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _apply_feature_engineering(self, x: pd.DataFrame) -> pd.DataFrame:
        x = apply_winsorize(x, self.winsor_bounds)
        x = apply_log1p_columns(x, columns=self.winsor_columns)
        x = add_interaction_and_bmi_category(x)
        return x

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        x = df.copy()
        for c in self.zero_to_na_cols:
            if c in x.columns:
                x[c] = x[c].replace(0, np.nan)
        for c in self.impute_cols:
            if c in x.columns and c in self.medians:
                x[c] = x[c].fillna(self.medians[c])
        x = self._apply_feature_engineering(x)
        x = x[self.feature_columns]
        return self.scaler.transform(x.values.astype(float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)[:, 1]
        thr = float(self.decision_threshold)
        return (proba >= thr).astype(np.int64)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        Xs = self.preprocess(df)
        return self.classifier.predict_proba(Xs)
