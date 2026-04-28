"""Paths and shared constants for the diabetes / AdaBoost project."""

from pathlib import Path

# diabetes_adaboost/config.py -> parents[2] == project root (contains data/, src/, flutter_app/, …)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "diabetes.csv"

RANDOM_STATE = 15
TEST_SIZE = 0.2

COLUMNS_ZERO_TO_NA = ["Insulin", "BloodPressure", "SkinThickness", "Glucose", "BMI"]
COLUMNS_IMPUTE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
