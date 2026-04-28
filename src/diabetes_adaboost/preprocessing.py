"""Cleaning, NaN handling, train/test splits (matches notebook flow, with clearer imputation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COLUMNS_IMPUTE, COLUMNS_ZERO_TO_NA, RANDOM_STATE, TEST_SIZE


def replace_zeros_with_nan(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    cols = COLUMNS_ZERO_TO_NA if columns is None else columns
    out = df.copy()
    out[cols] = out[cols].replace(0, np.nan)
    return out


def split_xy(df: pd.DataFrame, target: str = "Outcome"):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def train_test_split_both_versions(df: pd.DataFrame, df_drop: pd.DataFrame):
    X, y = split_xy(df)
    X_drop, y_drop = split_xy(df_drop)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(
        X_drop, y_drop, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return (X_train, X_test, y_train, y_test), (
        X_train_drop,
        X_test_drop,
        y_train_drop,
        y_test_drop,
    )


def impute_train_test_medians(
    X_train: pd.DataFrame, X_test: pd.DataFrame, columns=None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fill NaNs with training-set medians; apply same medians to test (notebook logic)."""
    cols = COLUMNS_IMPUTE if columns is None else columns
    Xt = X_train.copy()
    Xs = X_test.copy()
    medians: dict[str, float] = {}
    for col in cols:
        medians[col] = float(Xt[col].median())
        Xt[col] = Xt[col].fillna(medians[col])
        Xs[col] = Xs[col].fillna(medians[col])
    return Xt, Xs, medians


def zero_counts_report(df: pd.DataFrame, columns=None) -> None:
    cols = COLUMNS_ZERO_TO_NA if columns is None else columns
    for col in cols:
        zero_count = int((df[col] == 0).sum())
        pct = 100 * zero_count / len(df)
        print(f"{col}: {zero_count} : %{pct}")
