import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    CLASS_LABELS,
    NUMERIC_FEATURES,
    RAW_PATH,
    TARGET_COL,
    assign_planet_class,
    clean_data,
    iqr_clip,
    load_raw_data,
    prepare_data,
)
from src.modeling import (  # noqa: E402
    cross_validate_model,
    evaluate,
    get_baseline_model,
    make_pca_pipeline,
    make_pipeline,
    permutation_feature_importance,
    tune_model,
)


needs_raw = pytest.mark.skipif(
    not RAW_PATH.exists(), reason=f"raw dataset missing at {RAW_PATH}"
)


def test_assign_planet_class_by_mass():
    assert assign_planet_class(0.001, None) == "terrestrial"
    assert assign_planet_class(0.05, None) == "neptunian"
    assert assign_planet_class(1.0, None) == "gas_giant"


def test_assign_planet_class_by_radius_fallback():
    assert assign_planet_class(float("nan"), 0.1) == "terrestrial"
    assert assign_planet_class(float("nan"), 0.3) == "neptunian"
    assert assign_planet_class(float("nan"), 1.2) == "gas_giant"


def test_assign_planet_class_returns_none_when_no_data():
    assert assign_planet_class(float("nan"), float("nan")) is None


def test_iqr_clip_reduces_max():
    df = pd.DataFrame({"x": list(range(20)) + [10_000]})
    out = iqr_clip(df, cols=["x"], k=1.5)
    assert out["x"].max() < 10_000
    assert out["x"].min() == df["x"].min()


@needs_raw
def test_clean_data_has_target_and_no_leakage():
    df = clean_data(load_raw_data())
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].notna().all()
    for col in ["PlanetaryMassJpt", "RadiusJpt", "SurfaceTempK", "TypeFlag"]:
        assert col not in df.columns
    assert set(df[TARGET_COL].unique()) <= set(CLASS_LABELS)


@needs_raw
def test_prepare_data_split_sizes_and_features():
    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_data(random_state=0)
    total = len(X_tr) + len(X_val) + len(X_te)
    assert total > 3000
    assert abs(len(X_val) / total - 0.15) < 0.02
    assert abs(len(X_te) / total - 0.15) < 0.02
    for col in NUMERIC_FEATURES:
        assert col in X_tr.columns


@needs_raw
def test_baseline_beats_random():
    X_tr, X_val, _, y_tr, y_val, _ = prepare_data(random_state=0)
    model = get_baseline_model()
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_val, y_val)
    assert metrics["f1_macro"] > 0.5


@needs_raw
def test_cross_validate_model_returns_metrics():
    X_tr, _, _, y_tr, _, _ = prepare_data(random_state=0)
    res = cross_validate_model(get_baseline_model(), X_tr, y_tr, cv=3, random_state=0)
    assert {"mean_f1_macro", "std_f1_macro", "mean_accuracy"} <= res.keys()
    assert res["mean_f1_macro"] > 0.5


@needs_raw
def test_make_pca_pipeline_fits_and_predicts():
    X_tr, X_val, _, y_tr, y_val, _ = prepare_data(random_state=0)
    model = make_pca_pipeline(LogisticRegression(max_iter=500, random_state=0), n_components=4)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert set(np.unique(preds)) <= set(CLASS_LABELS)


@needs_raw
def test_tune_model_returns_best_estimator():
    X_tr, _, _, y_tr, _, _ = prepare_data(random_state=0)
    pipe = make_pipeline(LogisticRegression(max_iter=500, random_state=0))
    best, params, score = tune_model(
        pipe, {"estimator__C": [0.1, 1.0]}, X_tr, y_tr, cv=3, search="grid", random_state=0
    )
    assert "estimator__C" in params
    assert score > 0.5
    assert hasattr(best, "predict")


@needs_raw
def test_permutation_importance_shape():
    X_tr, X_val, _, y_tr, y_val, _ = prepare_data(random_state=0)
    model = get_baseline_model()
    model.fit(X_tr, y_tr)
    df = permutation_feature_importance(model, X_val, y_val, n_repeats=3, random_state=0)
    assert {"feature", "importance_mean"} <= set(df.columns)
    assert len(df) == X_val.shape[1]
