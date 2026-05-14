from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "oec.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET_COL = "PlanetClass"

NUMERIC_FEATURES = [
    "PeriodDays",
    "SemiMajorAxisAU",
    "Eccentricity",
    "InclinationDeg",
    "HostStarMassSlrMass",
    "HostStarRadiusSlrRad",
    "HostStarMetallicity",
    "HostStarTempK",
    "DistFromSunParsec",
    "DiscoveryYear",
]

CATEGORICAL_FEATURES = ["DiscoveryMethod"]

ENGINEERED_FEATURES = [
    "LogPeriodDays",
    "LogSemiMajorAxisAU",
    "LogDistFromSunParsec",
    "EquilibriumProxy",
]

LEAKAGE_COLS = [
    "PlanetaryMassJpt",
    "RadiusJpt",
    "SurfaceTempK",
    "TypeFlag",
    "ListsPlanetIsOn",
]

DROP_COLS = [
    "PlanetIdentifier",
    "LastUpdated",
    "RightAscension",
    "Declination",
    "PeriastronDeg",
    "LongitudeDeg",
    "AscendingNodeDeg",
    "AgeGyr",
    "HostStarAgeGyr",
]

MASS_LOW = 0.03
MASS_HIGH = 0.15
RADIUS_LOW = 0.18
RADIUS_HIGH = 0.50

CLASS_LABELS = ["terrestrial", "neptunian", "gas_giant"]


def load_raw_data(path: Path | str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def assign_planet_class(mass: float, radius: float) -> str | None:
    if pd.notna(mass):
        if mass < MASS_LOW:
            return "terrestrial"
        if mass < MASS_HIGH:
            return "neptunian"
        return "gas_giant"
    if pd.notna(radius):
        if radius < RADIUS_LOW:
            return "terrestrial"
        if radius < RADIUS_HIGH:
            return "neptunian"
        return "gas_giant"
    return None


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = [
        assign_planet_class(m, r)
        for m, r in zip(df["PlanetaryMassJpt"], df["RadiusJpt"])
    ]
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LogPeriodDays"] = np.log1p(df["PeriodDays"])
    df["LogSemiMajorAxisAU"] = np.log1p(df["SemiMajorAxisAU"])
    df["LogDistFromSunParsec"] = np.log1p(df["DistFromSunParsec"])
    df["EquilibriumProxy"] = df["HostStarTempK"] * np.sqrt(
        df["HostStarRadiusSlrRad"] / df["SemiMajorAxisAU"]
    )
    return df


def remove_outliers(df: pd.DataFrame, cols: list[str], q: float = 0.995) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            cap = df[c].quantile(q)
            df[c] = df[c].clip(upper=cap)
    return df


def iqr_clip(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def clean_data(df: pd.DataFrame, outlier_strategy: str = "quantile") -> pd.DataFrame:
    df = df.drop_duplicates(subset=["PlanetIdentifier"]).reset_index(drop=True)
    df = add_target(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df = df.drop(columns=LEAKAGE_COLS + DROP_COLS, errors="ignore")
    df = add_engineered_features(df)
    cols = ["PeriodDays", "SemiMajorAxisAU", "DistFromSunParsec", "EquilibriumProxy"]
    if outlier_strategy == "iqr":
        df = iqr_clip(df, cols=cols, k=3.0)
    else:
        df = remove_outliers(df, cols=cols)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=rel_val,
        stratify=y_trainval,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(numeric_imputer: str = "median", knn_neighbors: int = 5) -> ColumnTransformer:
    if numeric_imputer == "knn":
        num_imp = KNNImputer(n_neighbors=knn_neighbors)
    else:
        num_imp = SimpleImputer(strategy="median")
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", num_imp),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def prepare_data(
    raw_path: Path | str = RAW_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    outlier_strategy: str = "quantile",
):
    df = load_raw_data(raw_path)
    df_clean = clean_data(df, outlier_strategy=outlier_strategy)
    X, y = split_features_target(df_clean)
    return stratified_split(X, y, test_size, val_size, random_state)


def save_processed(df: pd.DataFrame, name: str = "oec_clean.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out
