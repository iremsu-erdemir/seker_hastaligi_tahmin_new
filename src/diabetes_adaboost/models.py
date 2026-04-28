"""AdaBoost, arama ızgaraları, taban sınıflandırıcılar ve hibrit (soft voting) yardımcıları."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class PreFittedSoftVotingClassifier(ClassifierMixin, BaseEstimator):
    """
    Zaten eğitilmiş sınıflandırıcıların olasılık ortalaması.
    İkinci kez fit maliyetinden kaçınmak için eğitim sonunda kullanılır.
    """

    def __init__(self, estimators: list[Any] | None = None):
        self.estimators = estimators

    def fit(self, X, y):
        if not self.estimators:
            raise ValueError("estimators boş olamaz")
        fitted: list[Any] = []
        for est in self.estimators:
            if hasattr(est, "classes_"):
                fitted.append(est)
            else:
                fitted.append(clone(est).fit(X, y))
        self.estimators_ = fitted
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators_]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def composite_cv_best_index(cv_results: dict[str, Any]) -> int:
    """RandomizedSearchCV refit: ROC-AUC, F1 ve Recall ortalaması (0–1 ölçeği)."""
    roc = np.asarray(cv_results["mean_test_roc_auc"], dtype=float)
    f1 = np.asarray(cv_results["mean_test_f1"], dtype=float)
    rec = np.asarray(cv_results["mean_test_recall"], dtype=float)
    combined = 0.35 * roc + 0.325 * f1 + 0.325 * rec
    idx = int(np.nanargmax(combined))
    return idx


def logistic_regression_param_dist() -> dict[str, Any]:
    return {
        "C": np.logspace(-3, 2, 18).tolist(),
        "solver": ["lbfgs", "saga"],
    }


def random_forest_param_dist() -> dict[str, Any]:
    return {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [6, 8, 10, 12, 16, None],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


def xgboost_param_dist() -> dict[str, Any]:
    return {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }


def _normalize_importances(values: np.ndarray) -> np.ndarray:
    s = float(np.sum(np.abs(values)))
    if s <= 0:
        return np.ones_like(values) / max(len(values), 1)
    return np.abs(values) / s


def model_feature_importance_ranking(
    clf: Any,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Sınıflandırıcıdan özellik önem sıralaması (JSON için liste sözlük)."""
    names = list(feature_names)
    if isinstance(clf, PreFittedSoftVotingClassifier):
        per_model: list[np.ndarray] = []
        for sub in clf.estimators_:
            sub_rank = model_feature_importance_ranking(sub, names)
            vec = np.zeros(len(names), dtype=float)
            for item in sub_rank:
                idx = names.index(item["feature"]) if item["feature"] in names else -1
                if idx >= 0:
                    vec[idx] = float(item["importance"])
            per_model.append(_normalize_importances(vec))
        if not per_model:
            return []
        stacked = np.mean(np.vstack(per_model), axis=0)
        order = np.argsort(-stacked)
        return [{"feature": names[i], "importance": float(stacked[i])} for i in order]

    if hasattr(clf, "coef_"):
        coef = np.ravel(np.asarray(clf.coef_))
        if coef.shape[0] != len(names):
            return []
        imp = _normalize_importances(coef)
        order = np.argsort(-imp)
        return [{"feature": names[i], "importance": float(imp[i])} for i in order]

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        if imp.shape[0] != len(names):
            return []
        imp = _normalize_importances(imp)
        order = np.argsort(-imp)
        return [{"feature": names[i], "importance": float(imp[i])} for i in order]

    return []


def calculate_model_metrics(true, predicted):
    cmx = confusion_matrix(true, predicted)
    c_report = classification_report(true, predicted)
    accuracy = accuracy_score(true, predicted)
    return cmx, c_report, accuracy


def print_evaluation_block(name: str, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def train_default_adaboosts():
    base = DecisionTreeClassifier(max_depth=4)
    ada_clf = AdaBoostClassifier(estimator=base)
    ada_drop = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4))
    return ada_clf, ada_drop


def adaboost_param_grid():
    return {
        "n_estimators": np.arange(50, 250, 50),
        "learning_rate": np.logspace(-2, 0, 10),
    }


def run_randomized_search(ada_clf, ada_drop, X_train, y_train, X_train_drop, y_train_drop, n_iter=10, cv=10):
    params = adaboost_param_grid()
    random_cv = RandomizedSearchCV(
        estimator=ada_clf,
        param_distributions=params,
        n_iter=n_iter,
        n_jobs=-1,
        scoring="accuracy",
        cv=cv,
    )
    random_cv_drop = RandomizedSearchCV(
        estimator=ada_drop,
        param_distributions=params,
        n_iter=n_iter,
        n_jobs=-1,
        scoring="accuracy",
        cv=cv,
    )
    random_cv.fit(X_train, y_train)
    random_cv_drop.fit(X_train_drop, y_train_drop)
    return random_cv, random_cv_drop


def print_best_search_results(random_cv, random_cv_drop):
    print(f"Best parameters: {random_cv.best_params_}")
    print(f"Best accuracy: {random_cv.best_score_:.2f}")
    print("--------------")
    print(f"Best parameters: {random_cv_drop.best_params_}")
    print(f"Best accuracy: {random_cv_drop.best_score_:.2f}")


def build_tuned_ada_drop():
    """Hand-picked params from the original notebook after search."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=200,
        learning_rate=0.1291549665014884,
    )


def baseline_models_dict():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "K-Neighbors Classifier": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
    }


def evaluate_models_dict(models: dict, X_train, y_train, X_test, y_test):
    for model in models.values():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        model_train_cfx, model_train_c_report, model_train_accuracy = calculate_model_metrics(
            y_train, y_train_pred
        )
        model_test_cfx, model_test_c_report, model_test_accuracy = calculate_model_metrics(
            y_test, y_test_pred
        )

        print(model)
        print("Evaluation for Training Set")
        print("cfx :", model_train_cfx)
        print("c_report :", model_train_c_report)
        print("accuracy :", model_train_accuracy)
        print("-----------------------------")
        print("Evaluation for Test Set")
        print("cfx :", model_test_cfx)
        print("c_report:", model_test_c_report)
        print("accuracy Score :", model_test_accuracy)
        print("-----------------------------\n")


def knn_and_rf_param_grids():
    knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}
    rf_params = {
        "max_depth": [5, 8, 10, 15, None],
        "max_features": ["sqrt", "log2", 5, 7, 10],
        "min_samples_split": [2, 8, 12, 20],
        "n_estimators": [100, 200, 500, 1000],
    }
    return knn_params, rf_params


def run_knn_rf_random_search(X_train, y_train, n_iter=20, cv=3):
    knn_params, rf_params = knn_and_rf_param_grids()
    randomcv_models = [
        ("KNN", KNeighborsClassifier(), knn_params),
        ("RF", RandomForestClassifier(), rf_params),
    ]
    results = {}
    for name, model, params in randomcv_models:
        randomcv = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
        )
        randomcv.fit(X_train, y_train)
        results[name] = randomcv.best_params_
        print("best params for :", name, randomcv.best_params_)
    return results


def tuned_knn_rf_models():
    return {
        "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=50),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            max_features=7,
            max_depth=15,
        ),
    }
