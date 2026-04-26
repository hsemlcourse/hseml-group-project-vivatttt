from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("estimator", estimator),
        ]
    )


def get_baseline_model(random_state: int = 42) -> Pipeline:
    return make_pipeline(
        LogisticRegression(max_iter=1000, random_state=random_state)
    )


def get_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "logreg": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=15)),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }


def evaluate(model: Pipeline, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "f1_weighted": float(f1_score(y, preds, average="weighted")),
    }


def detailed_report(model: Pipeline, X, y) -> str:
    preds = model.predict(X)
    return classification_report(y, preds, digits=4)


def confusion(model: Pipeline, X, y, labels: list[str]) -> np.ndarray:
    preds = model.predict(X)
    return confusion_matrix(y, preds, labels=labels)


def train_and_score(
    model: Pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_f1_macro": train_metrics["f1_macro"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
    }


def benchmark_models(
    X_train, y_train, X_val, y_val, random_state: int = 42
) -> pd.DataFrame:
    rows = []
    for name, model in get_candidate_models(random_state).items():
        scores = train_and_score(model, X_train, y_train, X_val, y_val)
        scores["model"] = name
        rows.append(scores)
    return pd.DataFrame(rows).set_index("model").sort_values("val_f1_macro", ascending=False)


def save_model(model: Pipeline, name: str = "baseline.joblib") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / name
    joblib.dump(model, out)
    return out


def load_model(name: str = "baseline.joblib") -> Pipeline:
    return joblib.load(MODELS_DIR / name)
