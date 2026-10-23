# Water Quality Prediction via Gradient Boosting and Deep Neural Networks

**Phase 1: Applied Machine Learning (AML) Evaluation & Baseline Architecture**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://docs.pytest.org/)

---

## 📌 Project Overview & Abstract

Predict a documented water quality class or index from physical and chemical measurements (pH, dissolved oxygen, turbidity, temperature, conductivity, and dissolved solids).

### The "Hybrid" Logic: Interpretable Tabular Relationships to Learned Nonlinear Chemical Interactions
* **AML Baseline:** Gradient-boosted decision trees (**XGBoost**) and **Random Forest** establish a rigorous tabular baseline on physical-chemical sensors.
* **The Gap:** Tree models partition feature space with orthogonal axis-aligned cuts. While effective on heterogeneous tables and monotonic features, trees struggle to model smooth continuous diagonal manifolds (e.g., temperature-dependent oxygen solubility or acid-base buffering kinetics) without deep, high-variance staircases.
* **Deep Learning (Phase 2):** Training a small regularized Multi-Layer Perceptron (MLP) after the AML baseline to learn continuous smooth representations, comparing regression heads for continuous indices and classification heads for discrete tiers.

---

## 🗂️ Repository Structure

```text
AML-project/
├── configs/
│   └── config.yaml               # Central hyperparameter and experiment configuration
├── data/
│   ├── raw/                      # Put your raw dataset CSV file here
│   ├── processed/                # Cleaned, split-isolated train/val/test data
│   └── external/                 # External standards (WHO, CCME guidelines)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA and domain feature verification
├── reports/
│   ├── figures/                  # Publication-ready plots (calibration curves, SHAP, feature importance)
│   ├── presentation/
│   │   ├── presentation_slides.md# Complete slide deck outline and speaker notes
│   │   └── viva_voce_guide.md    # In-depth exam defense & Q&A guide
│   └── report_ieee/
│       └── main.tex              # Comprehensive IEEE-format LaTeX project report
├── src/
│   ├── data/
│   │   ├── loader.py             # Schema inspection, auto-target discovery, robust ingestion
│   │   └── clean.py              # Sensor physical bounds validation and anomaly flagging
│   ├── features/
│   │   └── engineering.py        # Leakage audit, DO saturation physics, split-aware imputer
│   ├── models/
│   │   ├── train_baseline.py     # Random Forest and XGBoost pipelines + 5-fold CV tuning
│   │   ├── evaluate.py           # MAE, RMSE, R2, Macro F1, Brier Score, ECE calibration
│   │   └── explainability.py     # TreeSHAP beeswarm, bar plots, permutation importance
│   └── main_phase1.py            # Master CLI orchestrator for Phase 1 execution
├── tests/
│   └── test_pipeline.py          # Unit & integration tests for physical bounds, leakage, and imputer
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation & setup guide
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
The project uses a dedicated Python 3.13 virtual environment with all required scientific libraries:
```bash
# Create and activate virtual environment (already set up in workspace)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Providing the Dataset
Drop your dataset CSV file into the `data/raw/` directory:
```bash
# Example destination:
data/raw/water_potability.csv
# OR
data/raw/water_quality_index.csv
```
The loader automatically inspects the schema, detects sensor columns, identifies the target column, and configures either continuous regression or discrete classification.

### 3. Run Automated Tests
Verify pipeline integrity, data leakage detection, and physics bounds logic:
```bash
.venv/bin/pytest tests/
```

### 4. Execute Complete Phase 1 Pipeline
Once your dataset CSV is placed in `data/raw/`, run:
```bash
PYTHONPATH=. .venv/bin/python -m src.main_phase1
```

---

## 📈 Empirical Baseline Benchmark Results (Phase 1)

### Continuous WQI Regression (`CCME_Values`)
| Model | MAE | RMSE | $R^2$ | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean Baseline)** | 13.9210 | 17.4456 | -0.0001 | 20.47% |
| **Random Forest (Breiman)** | 0.2544 | 0.7306 | 0.9982 | 0.36% |
| **XGBoost (Chen & Guestrin)** | **0.1520** | **0.4926** | **0.9992** | **0.22%** |

### Discrete Water Quality Classification (`CCME_WQI`, 5 Tiers)
| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Stratified)** | 29.09% | 19.72% | 0.1972 | 0.2909 |
| **Random Forest** | **99.29%** | **98.54%** | **0.9857** | **0.9929** |
| **XGBoost** | 98.84% | 97.87% | 0.9813 | 0.9884 |

---

## 📓 Interactive Jupyter Notebooks

| Notebook | Focus | Key Highlights |
| :--- | :--- | :--- |
| [`01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) | Dataset Quality & EDA | Sensor distributions, class balance, correlation heatmap, waterbody type frequencies. |
| [`02_data_cleaning_and_bounds.ipynb`](notebooks/02_data_cleaning_and_bounds.ipynb) | Physics Bounds & Cleansing | Out-of-bounds auditing (pH > 14, Temp > 50°C), left-censored detection limits, fault indicators. |
| [`03_feature_engineering_and_leakage.ipynb`](notebooks/03_feature_engineering_and_leakage.ipynb) | Feature Engineering & Leakage | DO saturation polynomial, BOD/DO ratio, nitrogen speciation, zero-leakage Pearson audit. |
| [`04_model_evaluation_and_shap.ipynb`](notebooks/04_model_evaluation_and_shap.ipynb) | Model Benchmark & Interpretability | Random Forest vs XGBoost, calibration curves, TreeSHAP beeswarm & bar plots. |

---

## 📊 Phase 1 Deliverables Summary

| Deliverable | Location | Details |
| :--- | :--- | :--- |
| **Project Report** | `reports/report_ieee/main.tex` | Full IEEE conference paper format covering theory, EDA, leakage checks, baseline results, and Phase 2 outlook. |
| **Presentation Deck** | `reports/presentation/presentation_slides.md` | 10 structured slides with speaker notes covering all grading rubrics. |
| **Viva Voce Defense** | `reports/presentation/viva_voce_guide.md` | Comprehensive Q&A addressing inductive biases, variance reduction, ECE, TreeSHAP, and data leakage. |
| **Code Quality & Tests**| `tests/test_pipeline.py` | Pytest test suite with 100% pass rate. |
| **Models & Artifacts**| `models/`, `reports/figures/` | Serialized joblib pipelines, calibration diagrams, and SHAP plots. |
