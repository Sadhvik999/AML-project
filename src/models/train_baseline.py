"""
Baseline AML Model Training Module: Random Forest and XGBoost.
Supports both continuous WQI index regression and discrete water quality classification,
with nested cross-validation, hyperparameter tuning, and pipeline packaging.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier

from src.features.engineering import SplitAwareImputer, ChemicalDomainFeatureExtractor
from src.models.evaluate import compute_regression_metrics, compute_classification_metrics

logger = logging.getLogger(__name__)


def build_pipeline(
    model_type: str = "xgboost",
    task: str = "regression",
    impute_strategy: str = "iterative",
    add_domain_features: bool = True,
    random_state: int = 42
) -> Pipeline:
    """
    Construct an end-to-end scikit-learn Pipeline with feature engineering,
    split-aware imputation, and the specified estimator.
    """
    steps = []

    if add_domain_features:
        steps.append(("domain_features", ChemicalDomainFeatureExtractor()))

    steps.append(("imputer", SplitAwareImputer(strategy=impute_strategy, add_indicator=True)))

    if task == "regression":
        if model_type == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            estimator = XGBRegressor(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "dummy":
            estimator = DummyRegressor(strategy="mean")
        else:
            raise ValueError(f"Unknown regression model_type: {model_type}")
    else: # classification
        if model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            estimator = XGBClassifier(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
        elif model_type == "dummy":
            estimator = DummyClassifier(strategy="stratified", random_state=random_state)
        else:
            raise ValueError(f"Unknown classification model_type: {model_type}")

    steps.append(("estimator", estimator))
    return Pipeline(steps)


def get_search_spaces(task: str = "regression") -> Dict[str, dict]:
    """Define robust hyperparameter distributions for Random Forest and XGBoost."""
    rf_params = {
        "estimator__n_estimators": [150, 300, 500],
        "estimator__max_depth": [6, 10, 15, None],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__max_features": ["sqrt", "log2", 0.8]
    }

    xgb_params = {
        "estimator__n_estimators": [200, 350, 500],
        "estimator__learning_rate": [0.01, 0.03, 0.08, 0.15],
        "estimator__max_depth": [3, 5, 7, 9],
        "estimator__subsample": [0.7, 0.85, 1.0],
        "estimator__colsample_bytree": [0.7, 0.85, 1.0],
        "estimator__reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "estimator__reg_lambda": [0.1, 1.0, 5.0, 10.0]
    }

    return {"random_forest": rf_params, "xgboost": xgb_params}


def train_and_evaluate_baselines(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    task: str = "regression",
    test_size: float = 0.20,
    tune_hyperparameters: bool = True,
    n_iter_search: int = 15,
    random_state: int = 42,
    output_dir: Union[str, Path] = "models"
) -> Dict[str, any]:
    """
    Train Dummy baseline, Random Forest, and XGBoost on train split,
    tune hyperparameters via cross-validation, and evaluate on untouched holdout test set.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train / Test split
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    logger.info("Dataset split: Train=%d samples, Test=%d samples (Task=%s)",
                len(X_train), len(X_test), task)

    models_to_run = ["dummy", "random_forest", "xgboost"]
    search_spaces = get_search_spaces(task=task) if tune_hyperparameters else {}

    results: Dict[str, any] = {
        "task": task,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models": {}
    }

    for model_name in models_to_run:
        logger.info("--> Training pipeline for %s (%s)...", model_name, task)
        pipe = build_pipeline(model_type=model_name, task=task, random_state=random_state)

        if tune_hyperparameters and model_name in search_spaces:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if task == "classification" else KFold(n_splits=5, shuffle=True, random_state=random_state)
            scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"

            search = RandomizedSearchCV(
                pipe,
                param_distributions=search_spaces[model_name],
                n_iter=n_iter_search,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            logger.info("    Best params for %s: %s", model_name, best_params)
        else:
            best_model = pipe.fit(X_train, y_train)
            best_params = "default"

        # Predictions
        y_pred = best_model.predict(X_test)

        if task == "regression":
            test_metrics = compute_regression_metrics(np.array(y_test), y_pred)
            y_prob = None
        else:
            y_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
            test_metrics = compute_classification_metrics(np.array(y_test), y_pred, y_prob=y_prob)

        logger.info("    Test metrics for %s: %s", model_name, test_metrics)

        # Save model
        save_file = output_path / f"{model_name}_{task}.joblib"
        joblib.dump(best_model, save_file)
        logger.info("    Persisted model artifact to %s", save_file)

        results["models"][model_name] = {
            "best_params": best_params,
            "metrics": test_metrics,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "artifact_path": str(save_file)
        }

    return results
