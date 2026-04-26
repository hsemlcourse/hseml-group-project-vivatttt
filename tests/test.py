import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    CLASS_LABELS,
    NUMERIC_FEATURES,
    TARGET_COL,
    assign_planet_class,
    clean_data,
    load_raw_data,
    prepare_data,
)
from src.modeling import (  # noqa: E402
    evaluate,
    get_baseline_model,
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


def test_clean_data_has_target_and_no_leakage():
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].notna().all()
    for col in ["PlanetaryMassJpt", "RadiusJpt", "SurfaceTempK", "TypeFlag"]:
        assert col not in df.columns
    assert set(df[TARGET_COL].unique()) <= set(CLASS_LABELS)


def test_prepare_data_split_sizes_and_features():
    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_data(random_state=0)
    total = len(X_tr) + len(X_val) + len(X_te)
    assert total > 3000
    assert len(y_tr) == len(X_tr)
    assert len(y_val) == len(X_val)
    assert len(y_te) == len(X_te)
    assert abs(len(X_val) / total - 0.15) < 0.02
    assert abs(len(X_te) / total - 0.15) < 0.02
    for col in NUMERIC_FEATURES:
        assert col in X_tr.columns


def test_baseline_beats_random():
    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_data(random_state=0)
    model = get_baseline_model()
    model.fit(X_tr, y_tr)
    metrics = evaluate(model, X_val, y_val)
    assert metrics["accuracy"] > 0.5
    assert metrics["f1_macro"] > 0.5
