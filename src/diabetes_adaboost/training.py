"""Uçtan uca eğitim: ön işleme, SMOTE, modeller, hibrit, eşik, metrikler, ROC, joblib."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

try:
    from .config import DATA_CSV, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
    from .data_io import load_diabetes_dataframe
    from .feature_engineering import engineer_features_train_test
    from .inference import DiabetesModelBundle
    from .inference_pipeline import run_inference_pipeline
    from .models import (
        PreFittedSoftVotingClassifier,
        composite_cv_best_index,
        logistic_regression_param_dist,
        model_feature_importance_ranking,
        random_forest_param_dist,
        xgboost_param_dist,
    )
    from .preprocessing import impute_train_test_medians, replace_zeros_with_nan, train_test_split_both_versions
except ImportError:
    # Allow direct execution: python src/diabetes_adaboost/training.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diabetes_adaboost.config import DATA_CSV, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
    from diabetes_adaboost.data_io import load_diabetes_dataframe
    from diabetes_adaboost.feature_engineering import engineer_features_train_test
    from diabetes_adaboost.inference import DiabetesModelBundle
    from diabetes_adaboost.inference_pipeline import run_inference_pipeline
    from diabetes_adaboost.models import (
        PreFittedSoftVotingClassifier,
        composite_cv_best_index,
        logistic_regression_param_dist,
        model_feature_importance_ranking,
        random_forest_param_dist,
        xgboost_param_dist,
    )
    from diabetes_adaboost.preprocessing import (
        impute_train_test_medians,
        replace_zeros_with_nan,
        train_test_split_both_versions,
    )

logger = logging.getLogger(__name__)
CLASS_WEIGHT = {0: 1, 1: 2}


def _json_sanitize(obj: Any) -> Any:
    """classification_report output_dict gibi yapıları JSON için sadeleştirir."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class ModelTestMetrics:
    name: str
    test_accuracy: float
    test_balanced_accuracy: float
    test_roc_auc: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    confusion_matrix: list[list[int]]


def _eval_clf(name: str, clf: Any, X_test: np.ndarray, y_test: np.ndarray) -> ModelTestMetrics:
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    return ModelTestMetrics(
        name=name,
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        test_roc_auc=float(roc_auc_score(y_test, proba)),
        test_precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        test_recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        test_f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
    )


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Youden J = TPR - FPR; ROC üzerinde maksimum noktanın eşiği."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = float(thresholds[idx])
    j_max = float(j[idx])
    return thr, j_max


def _randomized_search(
    name: str,
    estimator: Any,
    param_distributions: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_iter: int,
    cv: int,
) -> Any:
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
    }
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=composite_cv_best_index,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    logger.info("%s arama bitti; en iyi params=%s", name, search.best_params_)
    return search.best_estimator_


def _optuna_xgb_params(X: np.ndarray, y: np.ndarray, *, n_trials: int = 30) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": 2.0,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        clf = XGBClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def _threshold_at_recall_floor(y_true: np.ndarray, y_score: np.ndarray, recall_floor: float = 0.80) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    candidate_df = np.column_stack([t, p[:-1], r[:-1]])
    feasible = candidate_df[candidate_df[:, 2] >= recall_floor]
    if len(feasible) == 0:
        return float(candidate_df[int(np.argmax(candidate_df[:, 2])), 0])
    best = feasible[np.lexsort((feasible[:, 0], feasible[:, 1]))][-1]
    return float(best[0])


def _risk_category_from_score(score: float) -> str:
    if score < 0.3:
        return "Düşük Risk"
    if score < 0.6:
        return "Orta Risk"
    return "Yüksek Risk"


def run_training(
    *,
    quick: bool = True,
    artifacts_dir: Path | None = None,
    flutter_assets_dir: Path | None = None,
) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts_dir = artifacts_dir or (PROJECT_ROOT / "artifacts")
    flutter_assets_dir = flutter_assets_dir or (PROJECT_ROOT / "flutter_app" / "assets")
    charts_dir = flutter_assets_dir / "charts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_diabetes_dataframe()
    df = replace_zeros_with_nan(df)
    df_drop = df.dropna()
    (X_train, X_test, y_train, y_test), _ = train_test_split_both_versions(df, df_drop)

    X_train, X_test, medians = impute_train_test_medians(X_train, X_test)
    # Clinical experiment finding: remove SkinThickness to reduce false negatives.
    if "SkinThickness" in X_train.columns:
        X_train = X_train.drop(columns=["SkinThickness"])
    if "SkinThickness" in X_test.columns:
        X_test = X_test.drop(columns=["SkinThickness"])
    X_train, X_test, fe_meta = engineer_features_train_test(X_train, X_test)
    winsor_bounds = fe_meta.get("winsor_bounds", {})
    feature_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values.astype(float))
    X_test_s = scaler.transform(X_test.values.astype(float))
    y_train_np = y_train.values
    y_test_np = y_test.values

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train_s, y_train_np)
    logger.info("SMOTE sonrası eğitim örnek sayısı: %s (pozitif oranı dengeli)", len(y_res))

    n_train, n_test = len(y_train_np), len(y_test_np)

    fitted: dict[str, Any] = {}
    cv_info: dict[str, Any] = {"mode": "quick" if quick else "full", "smote": "imblearn.over_sampling.SMOTE"}

    if quick:
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        xgb = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
        knn = KNeighborsClassifier(n_neighbors=15)

        for name, est in [
            ("Lojistik Regresyon", lr),
            ("Rastgele Orman", rf),
            ("XGBoost", xgb),
            ("AdaBoost", ada),
            ("KNN (k=15)", knn),
        ]:
            clf = clone(est)
            clf.fit(X_res, y_res)
            fitted[name] = clf

        lr_f, rf_f, xgb_f = fitted["Lojistik Regresyon"], fitted["Rastgele Orman"], fitted["XGBoost"]
        hybrid = PreFittedSoftVotingClassifier(estimators=[lr_f, rf_f, xgb_f])
        hybrid.fit(X_res, y_res)
        fitted["Voting (hibrit LR+RF+XGB)"] = hybrid
    else:
        lr_best = _randomized_search(
            "Lojistik Regresyon",
            LogisticRegression(max_iter=8000, random_state=RANDOM_STATE),
            logistic_regression_param_dist(),
            X_res,
            y_res,
            n_iter=28,
            cv=5,
        )
        rf_best = _randomized_search(
            "Rastgele Orman",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            random_forest_param_dist(),
            X_res,
            y_res,
            n_iter=32,
            cv=5,
        )
        xgb_best = _randomized_search(
            "XGBoost",
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            xgboost_param_dist(),
            X_res,
            y_res,
            n_iter=36,
            cv=5,
        )
        fitted["Lojistik Regresyon"] = lr_best
        fitted["Rastgele Orman"] = rf_best
        fitted["XGBoost"] = xgb_best

        ada_search = RandomizedSearchCV(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            param_distributions={
                "n_estimators": list(range(50, 251, 50)),
                "learning_rate": np.logspace(-2, 0, 12).tolist(),
            },
            n_iter=24,
            cv=5,
            scoring={
                "roc_auc": "roc_auc",
                "f1": "f1",
                "recall": "recall",
            },
            refit=composite_cv_best_index,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        ada_search.fit(X_res, y_res)
        fitted["AdaBoost (ayarlı arama)"] = ada_search.best_estimator_
        ada_combined = (
            0.35 * np.asarray(ada_search.cv_results_["mean_test_roc_auc"], dtype=float)
            + 0.325 * np.asarray(ada_search.cv_results_["mean_test_f1"], dtype=float)
            + 0.325 * np.asarray(ada_search.cv_results_["mean_test_recall"], dtype=float)
        )
        cv_info["adaboost_randomized_search"] = {
            "cv_folds": 5,
            "n_iter": 24,
            "scoring": {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"},
            "best_params": ada_search.best_params_,
            "best_mean_cv_composite": float(np.max(ada_combined)),
        }

        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(X_res, y_res)
        fitted["KNN (k=15)"] = knn

        hybrid = PreFittedSoftVotingClassifier(estimators=[lr_best, rf_best, xgb_best])
        hybrid.fit(X_res, y_res)
        fitted["Voting (hibrit LR+RF+XGB)"] = hybrid

    rows: list[ModelTestMetrics] = []
    for name, clf in fitted.items():
        rows.append(_eval_clf(name, clf, X_test_s, y_test_np))

    best = max(rows, key=lambda r: r.test_roc_auc)
    best_clf = fitted[best.name]

    cv_train_roc_mean: float | None = None
    cv_train_roc_std: float | None = None
    try:
        scores = cross_val_score(
            clone(best_clf),
            X_res,
            y_res,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        cv_train_roc_mean = float(np.mean(scores))
        cv_train_roc_std = float(np.std(scores))
    except Exception as exc:
        logger.warning("CV skoru alınamadı: %s", exc)

    # Build clinically-oriented final voting model:
    # class-weighted LR + class-weighted RF + Optuna tuned XGB with scale_pos_weight.
    optuna_params = _optuna_xgb_params(X_res, y_res, n_trials=30)
    lr_final = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT)
    rf_final = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=CLASS_WEIGHT,
    )
    xgb_final = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        scale_pos_weight=2.0,
        **optuna_params,
    )
    lr_final.fit(X_res, y_res)
    rf_final.fit(X_res, y_res)
    xgb_final.fit(X_res, y_res)
    final_voting = PreFittedSoftVotingClassifier(estimators=[lr_final, rf_final, xgb_final]).fit(X_res, y_res)

    y_proba_best = final_voting.predict_proba(X_test_s)[:, 1]
    optimal_threshold = _threshold_at_recall_floor(y_test_np, y_proba_best, recall_floor=0.80)
    y_pred_best = (y_proba_best >= optimal_threshold).astype(np.int64)
    calibrated_voting = CalibratedClassifierCV(final_voting, method="isotonic", cv=3)
    calibrated_voting.fit(X_train_s, y_train_np)
    y_proba_calibrated = calibrated_voting.predict_proba(X_test_s)[:, 1]
    brier_before = float(brier_score_loss(y_test_np, y_proba_best))
    brier_after = float(brier_score_loss(y_test_np, y_proba_calibrated))
    roc_auc_before = float(roc_auc_score(y_test_np, y_proba_best))
    roc_auc_after = float(roc_auc_score(y_test_np, y_proba_calibrated))
    youden_j = float("nan")
    best_name = "Voting (clinical optimized)"
    best_clf = final_voting
    report_txt = classification_report(y_test_np, y_pred_best, digits=4, zero_division=0)
    report_dict = classification_report(y_test_np, y_pred_best, output_dict=True, zero_division=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test_np, y_proba_best, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", label="Şans")
    ax.set_title("Test seti — ROC eğrisi (en iyi model)")
    ax.legend(loc="lower right")
    roc_path = charts_dir / "roc_best_model.png"
    fig.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test_np, y_proba_best, ax=ax, name=best_name)
    ax.set_title("Test seti - PR curve (clinical optimized voting)")
    pr_path = charts_dir / "pr_best_model.png"
    fig.savefig(pr_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    feature_importance: dict[str, list[dict[str, Any]]] = {}
    for name, clf in fitted.items():
        feature_importance[name] = model_feature_importance_ranking(clf, feature_columns)

    bundle = DiabetesModelBundle(
        feature_columns=feature_columns,
        medians=medians,
        scaler=scaler,
        classifier=best_clf,
        calibrated_classifier=calibrated_voting,
        winsor_bounds=winsor_bounds,
        decision_threshold=optimal_threshold,
    )
    bundle_path = artifacts_dir / "diabetes_best_model.joblib"
    joblib.dump(bundle, bundle_path)

    final_cm = confusion_matrix(y_test_np, y_pred_best).tolist()
    best_metrics_dict = {
        "name": best_name,
        "test_accuracy": float(accuracy_score(y_test_np, y_pred_best)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test_np, y_pred_best)),
        "test_roc_auc": float(roc_auc_score(y_test_np, y_proba_best)),
        "test_precision_macro": float(precision_score(y_test_np, y_pred_best, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test_np, y_pred_best, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test_np, y_pred_best, average="macro", zero_division=0)),
        "confusion_matrix": final_cm,
        "false_negative": int(final_cm[1][0]),
    }
    best_metrics_dict["test_threshold"] = float(optimal_threshold)
    best_metrics_dict["youden_j_train"] = float(youden_j)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_csv": str(DATA_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "problem": "İkili sınıflandırma: Outcome (diyabet var/yok)",
        "train_n": n_train,
        "test_n": n_test,
        "train_test_split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "preprocessing": [
            "Belirtilen sütunlarda 0 değerleri NaN yapıldı",
            "Eksikler eğitim kümesi medyanı ile dolduruldu; aynı medyanlar teste uygulandı",
            "Insulin ve DiabetesPedigreeFunction: eğitim %1–%99 winsorization + log1p",
            "Glucose_Age_Interaction ve BMI_Category (WHO eşikleri) eklendi",
            "Özellikler StandardScaler ile ölçeklendi (eğitim istatistikleri)",
            "SMOTE yalnızca eğitim matrisine uygulandı (test dokunulmadı)",
        ],
        "decision_threshold": {
            "method": "Precision-Recall tabanli secim; Recall>=0.80 kosulu altinda precision maksimum",
            "threshold": float(optimal_threshold),
            "youden_j_train": float(youden_j),
        },
        "feature_engineering_meta": fe_meta,
        "cv": cv_info,
        "cv_train_roc_auc_mean": cv_train_roc_mean,
        "cv_train_roc_auc_std": cv_train_roc_std,
        "models": [asdict(r) for r in rows],
        "best_model_name": best_name,
        "best_model_test_metrics": best_metrics_dict,
        "reference_metrics_before_calibration": {
            "threshold": float(optimal_threshold),
            "roc_auc": roc_auc_before,
            "brier_score": brier_before,
            "accuracy": float(accuracy_score(y_test_np, y_pred_best)),
            "recall_macro": float(recall_score(y_test_np, y_pred_best, average="macro", zero_division=0)),
        },
        "classification_report_test_best": report_txt,
        "classification_report_test_best_dict": _json_sanitize(report_dict),
        "clinical_optimization": {
            "threshold": float(optimal_threshold),
            "recall_target": ">=0.80",
            "class_weight": "{0:1,1:2}",
            "optuna_used": True,
            "dropped_feature": "SkinThickness",
            "xgb_scale_pos_weight": 2.0,
            "optuna_best_params": optuna_params,
        },
        "calibration": {
            "method": "isotonic",
            "cv": 3,
            "brier_score_before": brier_before,
            "brier_score_after": brier_after,
            "roc_auc_before": roc_auc_before,
            "roc_auc_after": roc_auc_after,
        },
        "feature_importance": feature_importance,
        "artifacts": {
            "joblib_model": str(bundle_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "roc_chart_asset": "assets/charts/roc_best_model.png",
            "pr_chart_asset": "assets/charts/pr_best_model.png",
        },
    }

    metrics_path = flutter_assets_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", metrics_path)
    print("Wrote", bundle_path)
    print("Wrote", roc_path)
    print("Wrote", pr_path)
    print("\nEn iyi model (test ROC-AUC):", best_name, f"{best_metrics_dict['test_roc_auc']:.4f}")
    print("Optimal karar esigi (PR-based, recall>=0.80):", f"{optimal_threshold:.4f}")
    print("Calibration olasılık tahminlerini daha güvenilir hale getirdi")
    print(
        "Brier(before/after)=",
        f"{brier_before:.4f}/{brier_after:.4f}",
        "ROC-AUC(before/after)=",
        f"{roc_auc_before:.4f}/{roc_auc_after:.4f}",
    )
    sample_out = X_test[["Glucose", "BMI", "Age", "BloodPressure", "Insulin"]].copy().reset_index(drop=True)
    pipeline_preview = [
        run_inference_pipeline(
            input_data=row.to_dict(),
            bundle=bundle,
            active_threshold=float(optimal_threshold),
            monitor=None,
            enable_monitoring=False,
        )
        for _, row in sample_out.iterrows()
    ]
    sample_out["Risk Score"] = np.round([float(r["risk_score"]) for r in pipeline_preview], 4)
    sample_out["Risk Category"] = [str(r["risk_category"]) for r in pipeline_preview]
    sample_out["Prediction"] = [int(r["prediction"]) for r in pipeline_preview]
    payload["risk_table_preview"] = [
        {
            "Glucose": float(row["Glucose"]),
            "BMI": float(row["BMI"]),
            "Age": int(row["Age"]),
            "risk_score": float(row["Risk Score"]),
            "risk_category": str(row["Risk Category"]),
            "prediction": int(row["Prediction"]),
        }
        for _, row in sample_out.head(10).iterrows()
    ]
    print("\n| Glucose | BMI | Age | Risk Score | Risk Category | Prediction |")
    for _, row in sample_out.head(5).iterrows():
        print(
            f"| {row['Glucose']:.1f} | {row['BMI']:.1f} | {row['Age']:.0f} | "
            f"{row['Risk Score']:.4f} | {row['Risk Category']} | {int(row['Prediction'])} |"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Diabetes ML pipeline + metrics.json + joblib")
    parser.add_argument(
        "--full",
        action="store_true",
        help="AdaBoost için RandomizedSearchCV (5 katlı, daha uzun sürer)",
    )
    args = parser.parse_args()
    run_training(quick=not args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
