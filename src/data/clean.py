"""
Data Cleaning & Sanity Checking Module.
Provides:
1. Dataset shape, memory footprint, and missingness audit (number of rows & columns)
2. Duplicate row detection and removal
3. Validation of physical/chemical bounds for water telemetry
4. Non-destructive sensor anomaly flagging (converting invalid readings to NaN + fault indicators)
5. Comprehensive missing value imputation / filling (median, mean, or grouped)
6. End-to-end cleaning pipeline reporting before-and-after diagnostics
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard natural/environmental physical bounds
DEFAULT_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ph": (0.0, 14.0),
    "dissolved_oxygen": (0.0, 30.0),       # mg/L
    "turbidity": (0.0, 5000.0),            # NTU
    "temperature": (-5.0, 50.0),           # Celsius (surface fresh/marine water)
    "conductivity": (0.0, 60000.0),        # µS/cm
    "total_dissolved_solids": (0.0, 60000.0), # mg/L
    "biochemical_oxygen_demand": (0.0, 1000.0), # mg/L
    "ammonia": (0.0, 500.0),               # mg/L
    "orthophosphate": (0.0, 200.0),        # mg/L
    "nitrogen": (0.0, 500.0),              # mg/L
    "nitrate": (0.0, 500.0),               # mg/L
    "hardness": (0.0, 2000.0),             # mg/L CaCO3
    "chloramines": (0.0, 30.0),            # mg/L
    "sulfate": (0.0, 3000.0),              # mg/L
}


def inspect_dataset_shape_and_nulls(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate and report dataset dimensions, memory usage, duplicate rows,
    and missing values per column.
    """
    n_rows, n_cols = df.shape
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    n_duplicates = int(df.duplicated().sum())
    
    missing_counts = df.isna().sum().to_dict()
    missing_pcts = (df.isna().mean() * 100).round(3).to_dict()
    total_missing = int(df.isna().sum().sum())

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": mem_mb,
        "n_duplicates": n_duplicates,
        "total_missing_cells": total_missing,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "columns": list(df.columns)
    }
    
    logger.info("Dataset shape: %d rows x %d columns | Memory: %.2f MB | Duplicates: %d | Missing cells: %d",
                n_rows, n_cols, mem_mb, n_duplicates, total_missing)
    return summary


def remove_duplicate_records(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """
    Identify and remove duplicate observations.
    Returns: (cleaned_df, number_of_duplicates_dropped)
    """
    initial_rows = len(df)
    df_dedup = df.drop_duplicates(subset=subset).reset_index(drop=True)
    n_dropped = initial_rows - len(df_dedup)
    logger.info("Duplicate check: dropped %d duplicate records (%d rows remaining).", n_dropped, len(df_dedup))
    return df_dedup, n_dropped


def audit_physical_bounds(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Audit dataframe against physical chemical constraints.
    Returns a diagnostic report dataframe with out-of-bounds counts and percentages.
    """
    bounds = bounds or DEFAULT_PHYSICAL_BOUNDS
    records = []

    for col in df.select_dtypes(include=[np.number]).columns:
        norm_col = col.strip().lower().replace(" ", "_").replace("-", "_")
        target_bound = None
        for key, bound in bounds.items():
            if key == norm_col or key in norm_col:
                target_bound = bound
                break

        if target_bound is not None:
            low, high = target_bound
            series = df[col].dropna()
            total_valid = len(series)
            below = (series < low).sum()
            above = (series > high).sum()
            invalid_total = below + above
            pct_invalid = (invalid_total / total_valid * 100) if total_valid > 0 else 0.0

            records.append({
                "column": col,
                "expected_min": low,
                "expected_max": high,
                "actual_min": series.min() if total_valid > 0 else np.nan,
                "actual_max": series.max() if total_valid > 0 else np.nan,
                "violations_below": int(below),
                "violations_above": int(above),
                "total_violations": int(invalid_total),
                "violation_pct": round(pct_invalid, 3)
            })

    report_df = pd.DataFrame(records)
    logger.info("Physical bounds audit complete. Audited %d matched sensor columns.", len(report_df))
    return report_df


def flag_and_clean_sensor_anomalies(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    action: str = "nan_and_flag" # 'nan_and_flag', 'clip', or 'drop'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Flag or correct impossible physical values.
    'nan_and_flag' sets physically impossible sensor values to NaN and adds a binary boolean flag column.
    This preserves records while signaling sensor failure to downstream imputers/models.
    """
    bounds = bounds or DEFAULT_PHYSICAL_BOUNDS
    df_clean = df.copy()
    violation_counts: Dict[str, int] = {}

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        norm_col = col.strip().lower().replace(" ", "_").replace("-", "_")
        target_bound = None
        for key, bound in bounds.items():
            if key == norm_col or key in norm_col:
                target_bound = bound
                break

        if target_bound is not None:
            low, high = target_bound
            mask_invalid = (df_clean[col] < low) | (df_clean[col] > high)
            num_invalid = int(mask_invalid.sum())
            violation_counts[col] = num_invalid

            if num_invalid > 0:
                logger.warning("Column '%s': found %d values outside physical range [%.1f, %.1f]",
                               col, num_invalid, low, high)
                if action == "nan_and_flag":
                    df_clean[f"{col}_is_sensor_fault"] = mask_invalid.astype(int)
                    df_clean.loc[mask_invalid, col] = np.nan
                elif action == "clip":
                    df_clean[col] = df_clean[col].clip(lower=low, upper=high)

    return df_clean, violation_counts


def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Impute / fill all missing/empty values (NaNs) in the dataframe.
    - If group_col is provided (e.g. 'Waterbody Type'), imputes using grouped median.
    - Falls back to global median/mean for any remaining NaNs.
    - Categorical columns are imputed with mode.
    Returns: (imputed_df, filled_counts_per_column)
    """
    df_filled = df.copy()
    filled_counts: Dict[str, int] = {}
    
    # Numeric columns
    num_cols = df_filled.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_missing = int(df_filled[col].isna().sum())
        if n_missing > 0:
            filled_counts[col] = n_missing
            if group_col and group_col in df_filled.columns:
                # Grouped median
                group_median = df_filled.groupby(group_col)[col].transform("median")
                df_filled[col] = df_filled[col].fillna(group_median)
            
            # Fill remaining with global median
            val_fill = df_filled[col].median() if numeric_strategy == "median" else df_filled[col].mean()
            df_filled[col] = df_filled[col].fillna(val_fill)
            logger.info("Filled %d missing values in '%s' using %s (fill value: %.4f)",
                        n_missing, col, numeric_strategy, float(val_fill))

    # Categorical columns
    cat_cols = df_filled.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        n_missing = int(df_filled[col].isna().sum())
        if n_missing > 0:
            filled_counts[col] = n_missing
            mode_val = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else "Unknown"
            df_filled[col] = df_filled[col].fillna(mode_val)
            logger.info("Filled %d missing values in categorical '%s' with mode '%s'",
                        n_missing, col, mode_val)

    remaining_nulls = int(df_filled.isna().sum().sum())
    logger.info("Imputation complete. Total remaining missing cells: %d", remaining_nulls)
    return df_filled, filled_counts


def clean_water_quality_dataset(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    impute_strategy: str = "median",
    group_col: Optional[str] = "Waterbody Type"
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Master end-to-end data cleaning routine:
    1. Inspects raw shape, columns, and missingness
    2. Strips whitespace from strings
    3. Detects and removes duplicate observations
    4. Enforces physical bounds, setting invalid sensor values to NaN + fault flag
    5. Imputes all missing values using grouped/global median
    6. Returns clean dataframe and audit diagnostic summary
    """
    logger.info("=== Starting Master Data Cleaning Routine ===")
    initial_stats = inspect_dataset_shape_and_nulls(df)
    
    # 1. Clean string whitespace
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Remove duplicates
    df_clean, n_dups_removed = remove_duplicate_records(df_clean)

    # 3. Flag physical bounds & anomalies
    df_clean, anomaly_counts = flag_and_clean_sensor_anomalies(df_clean, bounds=bounds, action="nan_and_flag")

    # 4. Fill missing values
    df_clean, filled_counts = fill_missing_values(df_clean, numeric_strategy=impute_strategy, group_col=group_col)

    final_stats = inspect_dataset_shape_and_nulls(df_clean)

    diagnostics = {
        "initial": initial_stats,
        "duplicates_removed": n_dups_removed,
        "anomalies_flagged": anomaly_counts,
        "values_filled": filled_counts,
        "final": final_stats
    }
    
    logger.info("=== Master Data Cleaning Complete ===")
    logger.info("Initial: %d rows x %d cols -> Final: %d rows x %d cols (0 nulls remaining)",
                initial_stats["n_rows"], initial_stats["n_cols"], final_stats["n_rows"], final_stats["n_cols"])
    return df_clean, diagnostics
