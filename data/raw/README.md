# Raw Water Quality Telemetry Datasets

This directory contains the primary physical and chemical multi-sensor telemetry datasets used for the **Water Quality Prediction via Gradient Boosting and Deep Neural Networks** project.

---

## 📂 Available Datasets

| File | Size | Description | Purpose |
| :--- | :---: | :--- | :--- |
| **[`water_quality_sample.csv`](water_quality_sample.csv)** | ~5.5 MB | 50,000 representative telemetry observations across monitoring sites. | Instant preview on GitHub browser interface and rapid local prototyping. |
| **`Combined_dataset.csv.gz`** | ~75 MB | Full compressed national CCME monitoring repository containing **2,827,977 observations**. | Full-scale training, benchmarking, and cross-validation (auto-decompressed by `pandas` / `src.data.loader`). |
| `Combined_dataset.csv` | 305 MB | Local uncompressed full repository (ignored from git tracking to comply with GitHub's 100 MB per-file limit). | Local execution when uncompressed. |

---

## 📊 Dataset Schema & Sensor Features

| Feature | Units | Description | Expected Physical Bounds |
| :--- | :---: | :--- | :---: |
| `Country` | Categorical | Country identifier (Canada, etc.) | N/A |
| `Area` | String / ID | Geographic monitoring station code | N/A |
| `Waterbody Type` | Categorical | River, Lake, Effluent, Estuarine, Bay, Canal, etc. | N/A |
| `Date` | Date (`DD-MM-YYYY`) | Telemetry sampling timestamp (1953 -- 2020) | N/A |
| `pH (ph units)` | pH scale | Acidity / Alkalinity level | $[0.0, 14.0]$ |
| `Temperature (cel)` | $^\circ\text{C}$ | Water temperature | $[-5.0, 50.0]$ |
| `Dissolved Oxygen (mg/l)` | $\text{mg/L}$ | Dissolved oxygen concentration | $[0.0, 30.0]$ |
| `Biochemical Oxygen Demand (mg/l)` | $\text{mg/L}$ | 5-day microbial oxygen consumption ($BOD_5$) | $[0.0, 1000.0]$ |
| `Ammonia (mg/l)` | $\text{mg/L}$ | Toxic un-ionized / total ammonium nitrogen | $[0.0, 500.0]$ |
| `Nitrate (mg/l)` | $\text{mg/L}$ | Oxidized nitrogen nutrient | $[0.0, 500.0]$ |
| `Nitrogen (mg/l)` | $\text{mg/L}$ | Total nitrogen concentration | $[0.0, 500.0]$ |
| `Orthophosphate (mg/l)` | $\text{mg/L}$ | Bioavailable phosphorus nutrient | $[0.0, 200.0]$ |
| `CCME_Values` | Float ($[0, 100]$) | **Continuous Target**: Canadian Water Quality Index score | $[0.0, 100.0]$ |
| `CCME_WQI` | Categorical | **Discrete Target**: Quality Tier (Excellent, Good, Fair, Marginal, Poor) | 5 Classes |

---

## 🚀 Loading Datasets in Python

Both files are supported natively with zero extra steps:
```python
import pandas as pd

# Load the full 2.82M-row compressed dataset:
df_full = pd.read_csv("data/raw/Combined_dataset.csv.gz")

# Or load via the project loader:
from src.data.loader import load_dataset
df = load_dataset("data/raw/Combined_dataset.csv.gz")
```
