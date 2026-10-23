"""
Unit and Integration Tests for AML Water Quality ML Pipeline.
Tests data leakage detection, split-aware imputation, and model pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.engineering import (
    compute_do_saturation,
    audit_data_leakage,
    SplitAwareImputer,
    ChemicalDomainFeatureExtractor
)
from src.data.clean import audit_physical_bounds, flag_and_clean_sensor_anomalies
from src.models.train_baseline import build_pipeline


@pytest.fixture
def sample_water_data():
    """Generate synthetic water quality sensor dataset."""
    np.random.seed(42)
    n = 100
    temp = np.random.uniform(10.0, 30.0, n)
    do = np.random.uniform(4.0, 10.0, n)
    ph = np.random.uniform(6.0, 8.5, n)
    turbidity = np.random.uniform(1.0, 25.0, n)
    conductivity = np.random.uniform(100.0, 800.0, n)
    solids = conductivity * np.random.uniform(0.55, 0.70, n)

    # Some missing values
    ph[10:15] = np.nan
    do[25:28] = np.nan

    # Target: binary potability or continuous index
    target_class = (ph > 6.5) & (ph < 8.5) & (do > 5.0) & (turbidity < 15.0)
    target_class = target_class.astype(int)

    df = pd.DataFrame({
        "temperature": temp,
        "dissolved_oxygen": do,
        "ph": ph,
        "turbidity": turbidity,
        "conductivity": conductivity,
        "total_dissolved_solids": solids,
        "potability": target_class
    })
    return df


def test_do_saturation_physics():
    """DO saturation must decrease with higher water temperatures."""
    sat_cold = compute_do_saturation(np.array([5.0]))
    sat_warm = compute_do_saturation(np.array([25.0]))
    assert sat_cold > sat_warm, "Cold water should hold more dissolved oxygen than warm water."


def test_physical_bounds_audit(sample_water_data):
    """Test physical bounds reporting."""
    # Inject impossible pH
    sample_water_data.loc[0, "ph"] = 25.0
    report = audit_physical_bounds(sample_water_data)
    assert not report.empty
    row_ph = report[report["column"] == "ph"].iloc[0]
    assert row_ph["violations_above"] >= 1


def test_leakage_audit_detects_leak(sample_water_data):
    """Test that audit_data_leakage flags near-identical features."""
    sample_water_data["leaked_target_copy"] = sample_water_data["potability"] + np.random.normal(0, 0.01, len(sample_water_data))
    report = audit_data_leakage(sample_water_data, target_col="potability", threshold=0.95)
    flagged_names = [f["feature"] for f in report["flagged_leakages"]]
    assert "leaked_target_copy" in flagged_names


def test_split_aware_imputer_isolation(sample_water_data):
    """Verify that imputer fits only on training set without raising or leaking."""
    X = sample_water_data.drop(columns=["potability"])
    X_train = X.iloc[:70].copy()
    X_test = X.iloc[70:].copy()

    imputer = SplitAwareImputer(strategy="median", add_indicator=True)
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    assert not X_train_imp.isna().any().any()
    assert not X_test_imp.isna().any().any()
    assert "ph_missing" in X_train_imp.columns


def test_model_pipeline_end_to_end(sample_water_data):
    """Test building and training baseline pipelines."""
    X = sample_water_data.drop(columns=["potability"])
    y = sample_water_data["potability"]

    pipe = build_pipeline(model_type="random_forest", task="classification", random_state=42)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})
