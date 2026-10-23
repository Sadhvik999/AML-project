"""
Data Cleaning & Sanity Checking Module.
Validates physical/chemical bounds for water sensors, flags impossible readings,
and handles detection limits without data leakage.
"""

from typing import Dict, List, Optional, Tuple
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
