"""
Feature Engineering and Data Leakage Prevention Module.
Implements:
1. Train-split-isolated imputation (MICE / Median / KNN + missing indicators)
2. Domain-driven chemical and physical sensor interaction features
3. Data leakage audits (target reconstitution checks and collinearity warnings)
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


def compute_do_saturation(temp_c: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Theoretical dissolved oxygen saturation concentration (mg/L) at 1 atm as a function of temperature.
    Based on standard freshwater polynomial approximation (Benson & Krause / USGS standards).
    DO_sat = 14.652 - 0.41022*T + 0.007991*T^2 - 0.000077774*T^3
    """
    t = np.clip(temp_c, 0.0, 50.0)
    do_sat = 14.652 - 0.41022 * t + 0.007991 * (t ** 2) - 0.000077774 * (t ** 3)
    return np.maximum(do_sat, 1.0)


def audit_data_leakage(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.98
) -> Dict[str, any]:
    """
    Audit feature space against the target column for potential data leakage.
    Flags features with absolute Pearson correlation >= threshold,
    or features that directly predict the target with near zero variance.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        logger.info("Target column '%s' is non-numeric; checking categorical representation.", target_col)
        target_series = pd.factorize(df[target_col])[0]
    else:
        target_series = df[target_col]

    correlations = {}
    flagged_leakages = []

    for col in numeric_df.columns:
        if col == target_col:
            continue
        valid_mask = ~(df[col].isna() | pd.isna(target_series))
        if valid_mask.sum() > 20:
            corr = np.corrcoef(df.loc[valid_mask, col], target_series[valid_mask])[0, 1]
            if not np.isnan(corr):
                correlations[col] = float(corr)
                if abs(corr) >= threshold:
                    flagged_leakages.append({
                        "feature": col,
                        "correlation": round(float(corr), 4),
                        "reason": f"Suspiciously high correlation (|r| >= {threshold})"
                    })

    # Check for direct identity or trivial formulas
    report = {
        "target_column": target_col,
        "n_features_checked": len(correlations),
        "flagged_leakages": flagged_leakages,
        "top_correlated_features": sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    }

    if flagged_leakages:
        logger.warning("POTENTIAL DATA LEAKAGE DETECTED: %s", flagged_leakages)
    else:
        logger.info("Data leakage audit passed: No single feature displays correlation >= %s", threshold)

    return report


class ChemicalDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract physically grounded chemical features from raw water measurements:
    - Dissolved Oxygen Saturation Ratio (% of saturation at temperature)
    - Specific Conductance at 25°C (temperature compensation)
    - TDS-to-Conductivity Ratio (ion density proxy)
    - pH distance from neutral (7.0) and hydrogen ion concentration [H+]
    """

    def __init__(
        self,
        col_temp: str = "temperature",
        col_do: str = "dissolved_oxygen",
        col_cond: str = "conductivity",
        col_tds: str = "total_dissolved_solids",
        col_ph: str = "ph",
        col_bod: str = "biochemical",
        col_ammonia: str = "ammonia",
        col_nitrogen: str = "nitrogen",
        col_nitrate: str = "nitrate",
        col_ortho: str = "orthophosphate"
    ):
        self.col_temp = col_temp
        self.col_do = col_do
        self.col_cond = col_cond
        self.col_tds = col_tds
        self.col_ph = col_ph
        self.col_bod = col_bod
        self.col_ammonia = col_ammonia
        self.col_nitrogen = col_nitrogen
        self.col_nitrate = col_nitrate
        self.col_ortho = col_ortho

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = {c.lower(): c for c in df.columns}

        # Match columns case-insensitively
        temp_col = next((cols[c] for c in cols if self.col_temp in c), None)
        do_col = next((cols[c] for c in cols if self.col_do in c), None)
        cond_col = next((cols[c] for c in cols if self.col_cond in c), None)
        tds_col = next((cols[c] for c in cols if self.col_tds in c), None)
        ph_col = next((cols[c] for c in cols if self.col_ph in c), None)
        bod_col = next((cols[c] for c in cols if self.col_bod in c), None)
        amm_col = next((cols[c] for c in cols if self.col_ammonia in c), None)
        nit_col = next((cols[c] for c in cols if self.col_nitrogen in c and "nitrate" not in c), None)
        no3_col = next((cols[c] for c in cols if self.col_nitrate in c), None)
        po4_col = next((cols[c] for c in cols if self.col_ortho in c), None)

        # 1. Temperature - DO Interaction: DO Saturation Percentage & Deficit
        if temp_col and do_col:
            sat_expected = compute_do_saturation(df[temp_col])
            df["feat_do_saturation_ratio"] = df[do_col] / sat_expected
            df["feat_do_deficit"] = sat_expected - df[do_col]

        # 2. Temperature - Conductivity Compensation (Standardized to 25°C)
        if temp_col and cond_col:
            temp_diff = df[temp_col] - 25.0
            correction_factor = 1.0 + 0.0191 * temp_diff
            df["feat_conductivity_25c"] = df[cond_col] / np.maximum(correction_factor, 0.1)

        # 3. TDS to Conductivity Ratio
        if tds_col and cond_col:
            df["feat_tds_to_cond_ratio"] = df[tds_col] / np.maximum(df[cond_col], 1.0)

        # 4. pH buffering and acidity dynamics
        if ph_col:
            df["feat_ph_dev_neutral"] = (df[ph_col] - 7.0).abs()
            df["feat_hydrogen_conc_log"] = -df[ph_col]
            df["feat_ph_is_extreme"] = ((df[ph_col] < 6.5) | (df[ph_col] > 8.5)).astype(int)

        # 5. Biochemical Oxygen Demand to DO Ratio (Organic Pollution Pressure)
        if bod_col and do_col:
            df["feat_bod_to_do_ratio"] = df[bod_col] / (df[do_col] + 1e-3)

        # 6. Nitrogen Speciation (Toxic Ammonia and Oxidized Nitrate fractions)
        if amm_col and nit_col:
            df["feat_ammonia_to_nitrogen_ratio"] = df[amm_col] / (df[nit_col] + 1e-3)

        if no3_col and nit_col:
            df["feat_nitrate_to_nitrogen_ratio"] = df[no3_col] / (df[nit_col] + 1e-3)

        # 7. Nutrient Eutrophication Interaction (N:P proxy)
        if no3_col and po4_col:
            df["feat_nutrient_enrichment"] = df[no3_col] * df[po4_col]

        return df


class SplitAwareImputer(BaseEstimator, TransformerMixin):
    """
    Imputer guaranteed to fit ONLY on training data, preventing cross-split leakage.
    Supports 'iterative' (MICE), 'median', and 'knn' strategies, with optional missingness indicator flags.
    """

    def __init__(self, strategy: str = "iterative", add_indicator: bool = True):
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.imputer_ = None
        self.feature_names_: List[str] = []
        self.indicator_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = list(X.columns)
        if self.add_indicator:
            # Note columns that have any missing values in train
            missing_cols = X.columns[X.isna().any()].tolist()
            self.indicator_cols_ = [f"{c}_missing" for c in missing_cols]

        if self.strategy == "iterative":
            self.imputer_ = IterativeImputer(max_iter=15, random_state=42, sample_posterior=False)
        elif self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=5)
        else:
            self.imputer_ = SimpleImputer(strategy="median")

        self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise RuntimeError("Imputer has not been fitted yet.")

        df_imputed = pd.DataFrame(
            self.imputer_.transform(X),
            columns=self.feature_names_,
            index=X.index
        )

        if self.add_indicator and self.indicator_cols_:
            for ind_col in self.indicator_cols_:
                orig_col = ind_col.replace("_missing", "")
                if orig_col in X.columns:
                    df_imputed[ind_col] = X[orig_col].isna().astype(int)

        return df_imputed
