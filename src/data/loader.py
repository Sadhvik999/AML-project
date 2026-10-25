"""
Data Loader Module for Water Quality Prediction.
Handles raw data ingestion, schema inspection, sensor column identification,
and automated target discovery.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard canonical sensor synonyms mapped to standard feature names
SENSOR_SYNONYMS: Dict[str, List[str]] = {
    "ph": ["ph", "p_h", "ph_value", "water_ph"],
    "dissolved_oxygen": [
        "dissolved_oxygen", "do", "d_o", "dissolvedoxygen", "oxygen_dissolved",
        "do_mg_l", "dissolved_o2"
    ],
    "turbidity": ["turbidity", "turb", "turb_ntu", "turbidity_ntu"],
    "temperature": [
        "temperature", "temp", "water_temp", "temp_c", "water_temperature"
    ],
    "conductivity": [
        "conductivity", "cond", "specific_conductance", "ec", "cond_us_cm"
    ],
    "total_dissolved_solids": [
        "total_dissolved_solids", "tds", "solids", "dissolved_solids", "tds_mg_l"
    ],
    "hardness": ["hardness", "total_hardness"],
    "chloramines": ["chloramines", "chloramine"],
    "sulfate": ["sulfate", "sulphate", "so4"],
    "organic_carbon": ["organic_carbon", "toc", "carbon"],
    "trihalomethanes": ["trihalomethanes", "thm"],
    "biochemical_oxygen_demand": ["bod", "biochemical_oxygen_demand", "b_o_d"],
    "nitrate": ["nitrate", "no3", "nitrates"]
}

TARGET_SYNONYMS: List[str] = [
    "potability", "potable", "target", "wqi", "water_quality_index",
    "quality", "class", "water_quality_class", "status", "quality_rating",
    "ccme_values", "ccme_wqi", "ccme"
]


def find_raw_dataset(raw_dir: Union[str, Path] = "data/raw") -> Optional[Path]:
    """Find the primary raw dataset file in the data/raw directory."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Directory {raw_dir} does not exist.")

    supported_extensions = [".csv", ".parquet", ".xlsx", ".xls", ".gz"]
    candidates = [
        f for f in raw_path.iterdir()
        if (f.suffix.lower() in supported_extensions or f.name.endswith(".csv.gz")) and not f.name.startswith(".")
    ]

    if not candidates:
        logger.warning("No dataset files found in %s yet.", raw_dir)
        return None

    # Return the first candidate (or largest if multiple)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    logger.info("Found raw dataset candidate: %s (%d bytes)", candidates[0].name, candidates[0].stat().st_size)
    return candidates[0]


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset with robust delimiter and encoding handling."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    is_gz = path.name.endswith(".gz")
    if ext == ".csv" or is_gz:
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                # Pandas automatically decompresses .gz files
                compression = "gzip" if is_gz else None
                engine = "c" if is_gz else "python"
                df = pd.read_csv(path, encoding=encoding, compression=compression)
                logger.info("Loaded CSV '%s' with %d rows and %d columns (encoding: %s)",
                            path.name, len(df), df.shape[1], encoding)
                return df
            except Exception as e:
                logger.debug("Failed loading with %s: %s", encoding, e)
                continue
        raise ValueError(f"Unable to parse CSV file {path} with common encodings.")
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def inspect_schema(df: pd.DataFrame) -> Dict[str, any]:
    """Inspect schema, missingness, and column types."""
    info = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isna().sum().to_dict(),
        "missing_percentages": (df.isna().mean() * 100).round(2).to_dict(),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(exclude=[np.number]).columns),
    }
    return info


def map_sensor_columns(columns: List[str]) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Map raw column names to standardized canonical sensor names and detect candidate target.
    Returns: (canonical_mapping, target_column_name)
    """
    col_normalized = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in columns}
    canonical_mapping: Dict[str, str] = {}
    detected_target: Optional[str] = None

    for raw_col, norm_col in col_normalized.items():
        # Check target synonyms
        if norm_col in TARGET_SYNONYMS or any(t in norm_col for t in ["target", "potability", "wqi"]):
            detected_target = raw_col
            continue

        # Check sensor synonyms
        matched = False
        for canonical, synonyms in SENSOR_SYNONYMS.items():
            if norm_col in synonyms:
                canonical_mapping[raw_col] = canonical
                matched = True
                break
        if not matched:
            for canonical, synonyms in SENSOR_SYNONYMS.items():
                if any(syn in norm_col for syn in synonyms):
                    canonical_mapping[raw_col] = canonical
                    break

    return canonical_mapping, detected_target
