"""Load the Pima Indians Diabetes CSV used in the original notebook."""

from __future__ import annotations

import pandas as pd

from .config import DATA_CSV


def load_diabetes_dataframe(csv_path=None) -> pd.DataFrame:
    path = DATA_CSV if csv_path is None else csv_path
    return pd.read_csv(path)
