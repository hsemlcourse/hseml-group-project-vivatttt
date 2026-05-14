from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def make_pipeline(estimator, **preprocessor_kwargs):
    return Pipeline([
        ("preprocessor", build_preprocessor(**preprocessor_kwargs)),
        ("estimator", estimator),
    ])


def make_pca_pipeline(estimator, n_components=0.95, **preprocessor_kwargs):
    return Pipeline([
        ("preprocessor", build_preprocessor(**preprocessor_kwargs)),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("estimator", estimator),
    ])


def get_baseline_model(random_state=42):
    return make_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))


def get_candidate_models(random_state=42):
    return {
        "logreg": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=15)),
        "svm": make_pipeline(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        ),
        "hist_gb": make_pipeline(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=random_state)
        ),
        "xgb": make_pipeline(
            XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                tree_method="hist", eval_metric="mlogloss",
                random_state=random_state, n_jobs=-1,
            )
        ),
    }


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "f1_weighted": float(f1_score(y, preds, average="weighted")),
    }


def detailed_report(model, X, y):
    return classification_report(y, model.predict(X), digits=4)


def confusion(model, X, y, labels):
    return confusion_matrix(y, model.predict(X), labels=labels)


def train_and_score(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    tr = evaluate(model, X_train, y_train)
    val = evaluate(model, X_val, y_val)
    return {
        "train_accuracy": tr["accuracy"],
        "train_f1_macro": tr["f1_macro"],
        "val_accuracy": val["accuracy"],
        "val_f1_macro": val["f1_macro"],
        "val_f1_weighted": val["f1_weighted"],
    }


def benchmark_models(X_train, y_train, X_val, y_val, random_state=42):
    rows = []
    for name, model in get_candidate_models(random_state).items():
        if name == "xgb":
            continue
        scores = train_and_score(model, X_train, y_train, X_val, y_val)
        scores["model"] = name
        rows.append(scores)
    return pd.DataFrame(rows).set_index("model").sort_values("val_f1_macro", ascending=False)


def cross_validate_model(model, X, y, cv=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    res = cross_validate(model, X, y, cv=skf, scoring=["f1_macro", "accuracy"], n_jobs=-1)
    return {
        "mean_f1_macro": float(res["test_f1_macro"].mean()),
        "std_f1_macro": float(res["test_f1_macro"].std()),
        "mean_accuracy": float(res["test_accuracy"].mean()),
        "std_accuracy": float(res["test_accuracy"].std()),
    }


def tune_model(model, param_grid, X, y, cv=5, search="grid", n_iter=25, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if search == "random":
        s = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, scoring="f1_macro",
            cv=skf, n_jobs=-1, random_state=random_state, refit=True,
        )
    else:
        s = GridSearchCV(model, param_grid, scoring="f1_macro", cv=skf, n_jobs=-1, refit=True)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_, float(s.best_score_)


def permutation_feature_importance(model, X, y, n_repeats=10, random_state=42):
    res = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="f1_macro", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_model(model, name="baseline.joblib"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / name
    joblib.dump(model, out)
    return out


def load_model(name="baseline.joblib"):
    return joblib.load(MODELS_DIR / name)
