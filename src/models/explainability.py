"""
Explainability and Interpretability Module.
Computes TreeSHAP explanations, permutation feature importances, and
dependence plots for chemical sensor interactions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import matplotlib
import os

# Set local cache dir for matplotlib
mpl_cache = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)
import matplotlib.pyplot as plt

import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def extract_preprocessed_features(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform features through all pipeline steps prior to final estimator."""
    X_trans = X.copy()
    for name, step in pipeline.steps[:-1]:
        X_trans = step.transform(X_trans)

    if isinstance(X_trans, pd.DataFrame):
        feature_names = list(X_trans.columns)
        X_array = X_trans.to_numpy()
    else:
        feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
        X_array = X_trans

    return X_array, feature_names


def explain_model_with_shap(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    model_name: str = "xgboost",
    output_dir: Union[str, Path] = "reports/figures",
    max_display: int = 12
) -> Dict[str, any]:
    """
    Compute SHAP values using TreeExplainer and save beeswarm summary & bar plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    estimator = pipeline.named_steps["estimator"]
    X_trans_arr, feature_names = extract_preprocessed_features(pipeline, X_sample)
    df_features = pd.DataFrame(X_trans_arr, columns=feature_names)

    logger.info("Computing TreeSHAP for %s on %d samples with %d features...",
                model_name, len(df_features), len(feature_names))

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(df_features)

        # Handle binary classification case where shap_values has 3 dims (samples, features, classes)
        if len(shap_values.shape) == 3:
            shap_vals_to_plot = shap_values[:, :, 1]
        else:
            shap_vals_to_plot = shap_values

        # 1. Beeswarm Plot
        plt.figure(figsize=(9, 6), dpi=300)
        shap.plots.beeswarm(shap_vals_to_plot, max_display=max_display, show=False)
        plt.title(f"SHAP Summary (Beeswarm) - {model_name.upper()}", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = out_path / f"shap_beeswarm_{model_name}.png"
        plt.savefig(beeswarm_path, bbox_inches="tight")
        plt.close()

        # 2. Bar Plot (Global Importance)
        plt.figure(figsize=(8, 5), dpi=300)
        shap.plots.bar(shap_vals_to_plot, max_display=max_display, show=False)
        plt.title(f"Mean |SHAP| Global Feature Importance - {model_name.upper()}", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        bar_path = out_path / f"shap_bar_{model_name}.png"
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved SHAP plots to %s", out_path)
        return {
            "explainer": explainer,
            "shap_values": shap_vals_to_plot,
            "feature_names": feature_names,
            "plots": {
                "beeswarm": str(beeswarm_path),
                "bar": str(bar_path)
            }
        }
    except Exception as e:
        logger.error("Error computing SHAP values: %s", e)
        return {}


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    scoring: str = "neg_root_mean_squared_error",
    n_repeats: int = 10,
    random_state: int = 42,
    output_dir: Union[str, Path] = "reports/figures",
    model_name: str = "model"
) -> pd.DataFrame:
    """
    Compute permutation importance on holdout test set to avoid Gini impurity bias.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    df_imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    # Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.barh(df_imp["feature"][:12][::-1], df_imp["importance_mean"][:12][::-1],
             xerr=df_imp["importance_std"][:12][::-1], color="#2b5c8f", alpha=0.85, capsize=3)
    plt.xlabel(f"Permutation Importance ({scoring})", fontsize=10)
    plt.title(f"Permutation Feature Importance - {model_name.upper()}", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plot_path = out_path / f"permutation_importance_{model_name}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved permutation importance plot to %s", plot_path)
    return df_imp
