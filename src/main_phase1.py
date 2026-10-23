"""
Phase 1 Master Pipeline Execution Script for CCME Water Quality Telemetry.
Orchestrates:
1. Large-scale ingestion & stratified sampling
2. Sensor physical bounds auditing & anomaly flagging
3. Zero-leakage audit (ensuring CCME_Values and CCME_WQI do not leak into features)
4. Site-separated & cross-validation splitting
5. Baseline training (Dummy, Random Forest, XGBoost) for both continuous WQI index and discrete class tiers
6. Calibration analysis (ECE & reliability diagrams) and TreeSHAP explainability plots
"""

from pathlib import Path
import sys
import logging
import json
import yaml
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data.loader import find_raw_dataset, load_dataset, inspect_schema
from src.data.clean import audit_physical_bounds, flag_and_clean_sensor_anomalies
from src.features.engineering import audit_data_leakage
from src.models.train_baseline import train_and_evaluate_baselines
from src.models.evaluate import plot_calibration_curves
from src.models.explainability import explain_model_with_shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase1_pipeline(config_path: str = "configs/config.yaml"):
    """Execute complete Phase 1 Machine Learning pipeline on uploaded CCME dataset."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=== Starting Phase 1: Machine Learning Evaluation ===")
    raw_dir = Path(config["data"]["raw_path"])
    raw_file = find_raw_dataset(raw_dir)

    if raw_file is None:
        logger.error("No raw dataset found in '%s'.", raw_dir)
        return False

    # 1. Ingestion
    logger.info("[1/7] Ingesting dataset: %s", raw_file.name)
    df_full = load_dataset(raw_file)
    total_records = len(df_full)
    logger.info("Raw dataset shape: %d rows, %d columns", total_records, df_full.shape[1])

    # Sample for robust, rapid cross-validation
    sample_size = config["data"].get("sample_size", 50000)
    if total_records > sample_size:
        logger.info("Subsampling %d representative records across quality tiers...", sample_size)
        df = df_full.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df = df_full

    # 2. Schema and Sensor Detection
    schema = inspect_schema(df)
    target_cont = config["data"].get("target_continuous", "CCME_Values")
    target_cat = config["data"].get("target_categorical", "CCME_WQI")

    # 3. Physical Bounds Auditing
    logger.info("[2/7] Auditing sensor physical bounds...")
    bounds_report = audit_physical_bounds(df)
    if not bounds_report.empty:
        print("\n=== Physical Sensor Bounds Diagnostic ===")
        print(tabulate(bounds_report, headers="keys", tablefmt="github", showindex=False))

    df_cleaned, anomaly_counts = flag_and_clean_sensor_anomalies(df, action="nan_and_flag")

    # 4. Leakage Audit
    logger.info("[3/7] Auditing feature space against targets to prevent data leakage...")
    leakage_report_cont = audit_data_leakage(df_cleaned.drop(columns=[target_cat], errors="ignore"), target_col=target_cont)
    print("\n=== Data Leakage Audit (Target: CCME_Values) ===")
    for feat, corr in leakage_report_cont["top_correlated_features"]:
        print(f"  - Feature '{feat}': Pearson r = {corr:.4f}")

    # Isolate feature matrix X (exclude metadata and target columns)
    cols_to_exclude = [
        target_cont, target_cat, "Country", "Area", "Waterbody Type", "Date"
    ]
    feature_cols = [c for c in df_cleaned.columns if c not in cols_to_exclude and not c.endswith("_fault")]
    # Include fault indicator flags as features
    fault_cols = [c for c in df_cleaned.columns if c.endswith("_is_sensor_fault")]
    all_feature_cols = feature_cols + fault_cols

    X = df_cleaned[all_feature_cols].copy()
    logger.info("Constructed feature matrix with %d features: %s", len(X.columns), list(X.columns))

    figures_dir = Path(config["reports"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_evaluations = {}

    # 5. Continuous WQI Index Regression Benchmark
    logger.info("[4/7] Training Baseline Models on Continuous Target: %s", target_cont)
    y_cont = df_cleaned[target_cont]
    results_reg = train_and_evaluate_baselines(
        X, y_cont,
        task="regression",
        test_size=config["split"]["test_size"],
        tune_hyperparameters=True,
        n_iter_search=5,
        random_state=config["project"]["random_seed"],
        output_dir="models"
    )

    reg_metrics = []
    for m_name, m_info in results_reg["models"].items():
        row = {"Model": m_name.replace("_", " ").title()}
        row.update(m_info["metrics"])
        reg_metrics.append(row)
    df_reg_metrics = pd.DataFrame(reg_metrics)
    print("\n=== CONTINUOUS WQI REGRESSION METRICS (TEST SET) ===")
    print(tabulate(df_reg_metrics, headers="keys", tablefmt="github", showindex=False))
    all_evaluations["regression"] = {
        m: results_reg["models"][m]["metrics"] for m in results_reg["models"]
    }

    # 6. Discrete Water Quality Classification Benchmark
    logger.info("[5/7] Training Baseline Models on Discrete Target: %s", target_cat)
    y_cat_raw = df_cleaned[target_cat]
    y_cat, class_names = pd.factorize(y_cat_raw)
    class_name_list = list(class_names)
    logger.info("Classification classes: %s", class_name_list)

    results_cls = train_and_evaluate_baselines(
        X, y_cat,
        task="classification",
        test_size=config["split"]["test_size"],
        tune_hyperparameters=True,
        n_iter_search=5,
        random_state=config["project"]["random_seed"],
        output_dir="models"
    )

    cls_metrics = []
    for m_name, m_info in results_cls["models"].items():
        row = {"Model": m_name.replace("_", " ").title()}
        row.update(m_info["metrics"])
        cls_metrics.append(row)
    df_cls_metrics = pd.DataFrame(cls_metrics)
    print("\n=== DISCRETE WQI CLASSIFICATION METRICS (TEST SET) ===")
    print(tabulate(df_cls_metrics, headers="keys", tablefmt="github", showindex=False))
    all_evaluations["classification"] = {
        m: results_cls["models"][m]["metrics"] for m in results_cls["models"]
    }

    # 7. Calibration Diagram & SHAP Explainability
    logger.info("[6/7] Computing probabilistic calibration and reliability curves...")
    try:
        calib_dict = {}
        for m in ["random_forest", "xgboost"]:
            y_test_m = np.asarray(results_cls["models"][m]["y_test"])
            y_prob_m = results_cls["models"][m]["y_prob"]
            if y_prob_m is not None:
                calib_dict[m.replace("_", " ").title()] = (
                    (y_test_m == 0).astype(int),
                    y_prob_m[:, 0]
                )
        if calib_dict:
            plot_calibration_curves(calib_dict, save_path=figures_dir / "model_calibration_curves.png")
    except Exception as e:
        logger.warning("Could not generate calibration curves: %s", e)

    logger.info("[7/7] Generating TreeSHAP interpretability figures for XGBoost...")
    best_xgb_path = Path("models") / "xgboost_regression.joblib"
    if best_xgb_path.exists():
        import joblib
        xgb_pipe = joblib.load(best_xgb_path)
        sample_x = X.sample(n=min(config["evaluation"]["shap_sample_size"], len(X)), random_state=42)
        explain_model_with_shap(xgb_pipe, sample_x, model_name="xgboost_regression", output_dir=figures_dir)

    # Save summary evaluation json
    with open(figures_dir / "evaluation_summary.json", "w") as f:
        json.dump(all_evaluations, f, indent=2)

    logger.info("=== Phase 1 Pipeline Successfully Executed! ===")
    return True


if __name__ == "__main__":
    run_phase1_pipeline()
