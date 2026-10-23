"""
Model Evaluation and Calibration Module.
Provides comprehensive metrics for both continuous WQI index targets and
categorical water quality / potability classifications, including Expected
Calibration Error (ECE), Brier score, and residual diagnostics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import matplotlib
import os

# Set local cache dir for matplotlib within workspace
mpl_cache = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, and MAPE for continuous targets."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    try:
        # Avoid zero-division in MAPE if target has 0s
        non_zero = y_true != 0
        if non_zero.sum() > 0:
            mape = float(mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero]))
        else:
            mape = float("nan")
    except Exception:
        mape = float("nan")

    tag = f"{prefix}_" if prefix else ""
    return {
        f"{tag}mae": round(mae, 4),
        f"{tag}rmse": round(rmse, 4),
        f"{tag}r2": round(r2, 4),
        f"{tag}mape": round(mape, 4) if not np.isnan(mape) else None
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE):
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        n_in_bin = np.sum(mask)

        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_in_bin / n_total) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """Compute Accuracy, Balanced Accuracy, Macro F1, Precision, Recall, ROC-AUC, Brier score, and ECE."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    tag = f"{prefix}_" if prefix else ""
    metrics = {
        f"{tag}accuracy": round(acc, 4),
        f"{tag}balanced_accuracy": round(bal_acc, 4),
        f"{tag}macro_f1": round(macro_f1, 4),
        f"{tag}weighted_f1": round(weighted_f1, 4),
        f"{tag}macro_precision": round(macro_prec, 4),
        f"{tag}macro_recall": round(macro_rec, 4)
    }

    if y_prob is not None:
        try:
            # If binary probability (1D or 2D with 2 columns)
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
            roc_auc = float(roc_auc_score(y_true, prob_pos))
            brier = float(brier_score_loss(y_true, prob_pos))
            ece = compute_expected_calibration_error(y_true, prob_pos)
            metrics[f"{tag}roc_auc"] = round(roc_auc, 4)
            metrics[f"{tag}brier_score"] = round(brier, 4)
            metrics[f"{tag}ece"] = round(ece, 4)
        except Exception as e:
            logger.warning("Could not compute probabilistic calibration metrics: %s", e)

    return metrics


def plot_calibration_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot comparative reliability calibration curves for multiple models.
    results_dict: {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated (Ideal)")

    for name, (y_true, y_prob) in results_dict.items():
        prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
        prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=10)
        ece = compute_expected_calibration_error(y_true, prob_pos)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{name} (ECE: {ece:.3f})")

    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical Accuracy)", fontsize=11)
    ax.set_title("Reliability Diagram / Calibration Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved calibration curve plot to %s", save_path)
    plt.close()
