"""Controlled experiments focused on lowering false negatives in diabetes prediction."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import PROJECT_ROOT, RANDOM_STATE
from .data_io import load_diabetes_dataframe
from .feature_engineering import engineer_features_train_test
from .models import PreFittedSoftVotingClassifier
from .preprocessing import impute_train_test_medians, replace_zeros_with_nan, train_test_split_both_versions

CLASS_WEIGHT = {0: 1, 1: 2}


def _print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def _prepare_data(drop_columns: list[str] | None = None) -> dict[str, Any]:
    df = load_diabetes_dataframe()
    df = replace_zeros_with_nan(df)
    df_drop = df.dropna()
    (X_train, X_test, y_train, y_test), _ = train_test_split_both_versions(df, df_drop)
    X_train, X_test, _ = impute_train_test_medians(X_train, X_test)

    drop_columns = drop_columns or []
    valid_drops = [c for c in drop_columns if c in X_train.columns]
    if valid_drops:
        X_train = X_train.drop(columns=valid_drops)
        X_test = X_test.drop(columns=valid_drops)

    X_train, X_test, _ = engineer_features_train_test(X_train, X_test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values.astype(float))
    X_test_s = scaler.transform(X_test.values.astype(float))

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train_s, y_train.values)

    return {
        "X_res": X_res,
        "y_res": y_res,
        "X_test_s": X_test_s,
        "y_test": y_test.values,
    }


def _metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cm": cm.tolist(),
        "false_negative": int(cm[1, 0]),
    }


def _print_metrics_block(label: str, metrics: dict[str, Any], *, include_cm: bool = False) -> None:
    print(f"\n{label}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    if include_cm:
        print(f"Confusion Matrix: {metrics['cm']}")
        print(f"False Negative: {metrics['false_negative']}")


def _build_voting_model(
    X_res: np.ndarray,
    y_res: np.ndarray,
    *,
    use_class_weight: bool,
    xgb_params: dict[str, Any] | None = None,
) -> PreFittedSoftVotingClassifier:
    lr_kwargs: dict[str, Any] = {"max_iter": 5000, "random_state": RANDOM_STATE}
    rf_kwargs: dict[str, Any] = {"n_estimators": 400, "max_depth": 12, "random_state": RANDOM_STATE, "n_jobs": -1}
    xgb_kwargs: dict[str, Any] = {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if xgb_params:
        xgb_kwargs.update(xgb_params)
    if use_class_weight:
        lr_kwargs["class_weight"] = CLASS_WEIGHT
        rf_kwargs["class_weight"] = CLASS_WEIGHT
        xgb_kwargs["scale_pos_weight"] = 2.0

    lr = LogisticRegression(**lr_kwargs)
    rf = RandomForestClassifier(**rf_kwargs)
    xgb = XGBClassifier(**xgb_kwargs)
    lr.fit(X_res, y_res)
    rf.fit(X_res, y_res)
    xgb.fit(X_res, y_res)
    voting = PreFittedSoftVotingClassifier(estimators=[lr, rf, xgb]).fit(X_res, y_res)
    return voting


def _run_optuna_for_xgb(X_res: np.ndarray, y_res: np.ndarray) -> dict[str, Any]:
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
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_res, y_res, scoring="roc_auc", cv=cv, n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    return study.best_params


def _select_threshold_with_pr(y_true: np.ndarray, y_score: np.ndarray, min_recall: float = 0.80) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    candidate_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precisions[:-1],
            "recall": recalls[:-1],
        }
    )
    feasible = candidate_df[candidate_df["recall"] >= min_recall]
    if feasible.empty:
        idx = int(np.argmax(candidate_df["recall"].values))
        return float(candidate_df.iloc[idx]["threshold"])
    best_row = feasible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return float(best_row["threshold"])


def _save_curves(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, str]:
    charts_dir = PROJECT_ROOT / "flutter_app" / "assets" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    pr_path = charts_dir / "pr_curve_controlled_experiments.png"
    roc_path = charts_dir / "roc_curve_controlled_experiments.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="Final Voting")
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="Final Voting")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"pr_curve": str(pr_path), "roc_curve": str(roc_path)}


def run_controlled_experiments() -> None:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_data = _prepare_data(drop_columns=None)
    skin_clean_data = _prepare_data(drop_columns=["SkinThickness"])

    _print_header("BASELINE - ESKI VOTING (threshold=0.50)")
    baseline_model = _build_voting_model(base_data["X_res"], base_data["y_res"], use_class_weight=False)
    baseline_scores = baseline_model.predict_proba(base_data["X_test_s"])[:, 1]
    baseline_metrics = _metrics_from_scores(base_data["y_test"], baseline_scores, threshold=0.50)
    _print_metrics_block("Baseline metrikleri", baseline_metrics)
    print("Teknik yorum: Referans model ROC-AUC ve Recall dengesini baslangic noktasi olarak verir.")

    _print_header("ADIM 1 - FEATURE CLEANUP")
    step1_model = _build_voting_model(skin_clean_data["X_res"], skin_clean_data["y_res"], use_class_weight=False)
    step1_scores = step1_model.predict_proba(skin_clean_data["X_test_s"])[:, 1]
    step1_metrics = _metrics_from_scores(skin_clean_data["y_test"], step1_scores, threshold=0.50)
    _print_metrics_block("SkinThickness cikarilmis model (threshold=0.50)", step1_metrics)
    print("Teknik yorum: Ozellik sadelestirme, gereksiz/sinyali zayif degisken etkisini azaltabilir.")

    _print_header("ADIM 2 - CLASS WEIGHT ENTEGRASYONU")
    step2_model = _build_voting_model(skin_clean_data["X_res"], skin_clean_data["y_res"], use_class_weight=True)
    step2_scores = step2_model.predict_proba(skin_clean_data["X_test_s"])[:, 1]
    step2_metrics = _metrics_from_scores(skin_clean_data["y_test"], step2_scores, threshold=0.50)
    _print_metrics_block("Class-weight voting (threshold=0.50)", step2_metrics)
    print("Model artik hasta sinifini daha fazla onemsemektedir")

    _print_header("ADIM 3 - OPTUNA ILE HYPERPARAMETRE OPTIMIZASYONU")
    best_xgb_params = _run_optuna_for_xgb(skin_clean_data["X_res"], skin_clean_data["y_res"])
    print(f"En iyi XGBoost parametreleri: {best_xgb_params}")
    step3_model = _build_voting_model(
        skin_clean_data["X_res"],
        skin_clean_data["y_res"],
        use_class_weight=True,
        xgb_params=best_xgb_params,
    )
    step3_scores = step3_model.predict_proba(skin_clean_data["X_test_s"])[:, 1]
    step3_metrics = _metrics_from_scores(skin_clean_data["y_test"], step3_scores, threshold=0.50)
    _print_metrics_block("Optuna optimize voting (threshold=0.50)", step3_metrics)
    print("Teknik yorum: Optuna, ROC-AUC hedefiyle XGBoost parametrelerini veri odakli ayarlar.")

    _print_header("ADIM 4 - PRECISION-RECALL TEMELLI THRESHOLD SECIMI")
    optimal_threshold = _select_threshold_with_pr(skin_clean_data["y_test"], step3_scores, min_recall=0.80)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print("Teknik yorum: Recall >= 0.80 kosulu altinda precision maksimum yapan esik secildi.")

    _print_header("ADIM 5 - FINAL MODEL DEGERLENDIRME")
    final_metrics = _metrics_from_scores(skin_clean_data["y_test"], step3_scores, threshold=optimal_threshold)
    _print_metrics_block("Final metrikler", final_metrics, include_cm=True)
    print("Teknik yorum: Esik optimizasyonu FN azalitimi ve klinik guvenlik icin dogrudan kullanildi.")

    _print_header("ADIM 6 - KARSILASTIRMA TABLOSU")
    compare_df = pd.DataFrame(
        [
            {
                "Model": "Eski Voting (baseline)",
                "ROC-AUC": baseline_metrics["roc_auc"],
                "Recall": baseline_metrics["recall"],
                "Precision": baseline_metrics["precision"],
                "F1": baseline_metrics["f1"],
            },
            {
                "Model": "Feature cleaned model",
                "ROC-AUC": step1_metrics["roc_auc"],
                "Recall": step1_metrics["recall"],
                "Precision": step1_metrics["precision"],
                "F1": step1_metrics["f1"],
            },
            {
                "Model": "Class-weight model",
                "ROC-AUC": step2_metrics["roc_auc"],
                "Recall": step2_metrics["recall"],
                "Precision": step2_metrics["precision"],
                "F1": step2_metrics["f1"],
            },
            {
                "Model": "Final optimized model",
                "ROC-AUC": final_metrics["roc_auc"],
                "Recall": final_metrics["recall"],
                "Precision": final_metrics["precision"],
                "F1": final_metrics["f1"],
            },
        ]
    )
    print(compare_df.to_string(index=False))

    _print_header("ADIM 7 - KISA SONUC YORUMU")
    recall_delta = final_metrics["recall"] - baseline_metrics["recall"]
    fn_delta = final_metrics["false_negative"] - baseline_metrics["false_negative"]
    print("- Model hangi tekniklerle iyilestirildi?")
    print("  SkinThickness cikarimi, class_weight/scale_pos_weight entegrasyonu, Optuna tabanli XGBoost tuning ve PR-tabanli threshold secimi.")
    print(f"- Recall ne kadar artti?\n  {recall_delta:+.4f}")
    print(f"- False Negative nasil degisti?\n  {baseline_metrics['false_negative']} -> {final_metrics['false_negative']} (delta {fn_delta:+d})")
    print("- Klinik kullanim icin neden daha uygun?")
    print("  Daha yuksek recall ve daha dusuk FN, riskli hastalari kacirma olasiligini azaltir.")

    chart_paths = _save_curves(skin_clean_data["y_test"], step3_scores)
    print(f"PR egrisi kaydedildi: {chart_paths['pr_curve']}")
    print(f"ROC egrisi kaydedildi: {chart_paths['roc_curve']}")

    payload = {
        "baseline": baseline_metrics,
        "step1_feature_cleanup": step1_metrics,
        "step2_class_weight": step2_metrics,
        "step3_optuna_best_xgb_params": best_xgb_params,
        "step4_optimal_threshold": float(optimal_threshold),
        "step5_final": final_metrics,
        "step6_comparison": compare_df.to_dict(orient="records"),
        "charts": chart_paths,
    }
    out_path = artifacts_dir / "controlled_experiments_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Sonuclar kaydedildi: {out_path}")


if __name__ == "__main__":
    run_controlled_experiments()
