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

## 📋 Phase 1 Evaluation Structure

This project is strictly structured according to the Phase 1 evaluation sequence:

1. **[Data Cleaning](#1-data-cleaning):** Physical bounds auditing (pH, Temp, DO), detection limits, and non-destructive anomaly flagging.
2. **[Literature Review](#2-literature-review):** Breiman (2001), Chen & Guestrin (2016), Lundberg (2020), CCME WQI standards, and the Tabular-to-DL inductive bias gap.
3. **[Dataset Quality & EDA](#3-dataset-quality--eda):** 2.82M national CCME records across 12 waterbody types, sensor distributions, and class distributions.
4. **[Feature Engineering](#4-feature-engineering):** Temperature-dependent DO solubility polynomial, organic stress ratio, nitrogen speciation, and zero-leakage Pearson audit.
5. **[Theoretical Rigor](#5-theoretical-rigor):** Mathematical formulation of Random Forest (bagging variance reduction) vs XGBoost (second-order Taylor expansion).
6. **[Model Application](#6-model-application):** 5-Fold cross-validation, continuous regression ($R^2 = 0.9992$, $\text{RMSE} = 0.4926$), discrete classification ($99.29\%$ accuracy), and ECE calibration.
7. **[GitHub Repository & Code Quality](#7-github-repository--code-quality):** Clean modular architecture, automated Pytest suite (5/5 passing), and atomic conventional commits.
8. **[Project Presentation](#8-project-presentation):** 10-slide structured presentation deck with speaker notes.
9. **[Viva Voce](#9-viva-voce):** Comprehensive oral defense guide with authoritative answers to examiner questions.
10. **[Project Report](#10-project-report):** Complete two-column IEEE-format LaTeX conference paper ready for submission.

---

## 🗂️ Repository Structure

```text
AML-project/
├── configs/
│   └── config.yaml               # Central hyperparameter and experiment configuration
├── data/
│   ├── raw/                      # Raw dataset storage (e.g., Combined_dataset.csv)
│   ├── processed/                # Cleaned, split-isolated train/val/test data
│   └── external/                 # External standards (WHO, CCME guidelines)
├── notebooks/
│   ├── 01_data_cleaning_and_bounds.ipynb   # 1. Sensor bounds auditing & fault detection
│   ├── 02_dataset_quality_and_eda.ipynb    # 2. EDA, multi-sensor correlations, waterbodies
│   ├── 03_feature_engineering.ipynb        # 3. Chemical interactions & zero-leakage audit
│   └── 04_model_application.ipynb          # 4. RF vs XGBoost benchmark, calibration & SHAP
├── reports/
│   ├── figures/                  # Plots (calibration curves, SHAP beeswarm, importance)
│   ├── presentation/
│   │   ├── presentation_slides.md# 10-slide deck outline with detailed speaker notes
│   │   └── viva_voce_guide.md    # In-depth exam defense & Q&A guide
│   └── report_ieee/
│       └── main.tex              # Comprehensive IEEE-format LaTeX project report
├── src/
│   ├── data/
│   │   ├── loader.py             # Schema inspection & auto-target column detection
│   │   └── clean.py              # Sensor physical bounds validation and anomaly flagging
│   ├── features/
│   │   └── engineering.py        # Leakage audit, DO saturation physics, split-aware imputer
│   ├── models/
│   │   ├── train_baseline.py     # Random Forest and XGBoost pipelines + 5-fold CV tuning
│   │   ├── evaluate.py           # MAE, RMSE, R2, Macro F1, Brier Score, ECE calibration
│   │   └── explainability.py     # TreeSHAP beeswarm, bar plots, permutation importance
│   └── main_phase1.py            # Master CLI orchestrator for Phase 1 execution
├── tests/
│   └── test_pipeline.py          # Pytest suite (100% passing)
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation & setup guide
```

---

## 📈 Empirical Baseline Benchmark Results

### 1. Continuous WQI Regression (`CCME_Values` $\in [0, 100]$)
| Model | MAE | RMSE | $R^2$ | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean Baseline)** | 13.9210 | 17.4456 | -0.0001 | 20.47% |
| **Random Forest (Breiman)** | 0.2544 | 0.7306 | 0.9982 | 0.36% |
| **XGBoost (Chen & Guestrin)** | **0.1520** | **0.4926** | **0.9992** | **0.22%** |

### 2. Discrete Water Quality Classification (`CCME_WQI`, 5 Tiers)
| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Stratified)** | 29.09% | 19.72% | 0.1972 | 0.2909 |
| **Random Forest** | **99.29%** | **98.54%** | **0.9857** | **0.9929** |
| **XGBoost** | 98.84% | 97.87% | 0.9813 | 0.9884 |

---

## 🚀 Quickstart & Pipeline Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# Run automated tests
pytest tests/

# Execute Phase 1 Pipeline
PYTHONPATH=. python -m src.main_phase1
```

---

## 📊 Phase 1 Deliverables Summary

| Deliverable | Location | Details |
| :--- | :--- | :--- |
| **1. Data Cleaning** | `src/data/clean.py`, `notebooks/01_data_cleaning_and_bounds.ipynb` | Physical bounds validator and non-destructive anomaly flagging. |
| **2. Literature Review** | `reports/report_ieee/main.tex`, `reports/presentation/presentation_slides.md` | Theoretical comparison of bagging, boosting, and the tabular-to-DL gap. |
| **3. Dataset Quality & EDA** | `notebooks/02_dataset_quality_and_eda.ipynb` | Distributions, 12 waterbody types, correlation matrix on 2.82M records. |
| **4. Feature Engineering** | `src/features/engineering.py`, `notebooks/03_feature_engineering.ipynb` | DO solubility physics, nutrient ratios, and zero-leakage Pearson audit. |
| **5. Theoretical Rigor** | `reports/presentation/viva_voce_guide.md`, `reports/report_ieee/main.tex` | Mathematical formulation of Breiman (2001) and Chen & Guestrin (2016). |
| **6. Model Application** | `src/models/`, `notebooks/04_model_application.ipynb` | Holdout evaluation, ECE calibration curves, and TreeSHAP beeswarm plots. |
| **7. GitHub Repository & Code Quality**| `tests/test_pipeline.py`, `.gitignore` | Modular code, CI unit tests, and clean conventional commit history. |
| **8. Project Presentation** | `reports/presentation/presentation_slides.md` | 10 structured slides with comprehensive speaker notes. |
| **9. Viva Voce** | `reports/presentation/viva_voce_guide.md` | Detailed exam Q&A covering every rubric question. |
| **10. Project Report** | `reports/report_ieee/main.tex` | IEEE conference paper ready for compilation and submission. |
