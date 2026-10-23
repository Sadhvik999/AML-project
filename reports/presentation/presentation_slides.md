# Project Presentation Slides: Water Quality Prediction via AML

**Phase 1: Machine Learning Evaluation & Baseline Architecture**  
*Course:* Applied Machine Learning (AML)  
*Format:* Slide Deck Sequence Aligned with Required Evaluation Rubric

---

## Slide 1: Title & Project Overview
* **Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks
* **Subtitle:** Phase 1 – Applied Machine Learning (AML) Baseline & Systematic Validation
* **Core Problem:** Real-time water quality monitoring from multi-sensor physical & chemical telemetry, overcoming the multi-day incubation latency of laboratory assays.
* **The "Hybrid" Logic:** Tabular tree models (Phase 1) capture step-function thresholds; Deep Neural Networks (Phase 2) learn smooth nonlinear chemical interaction manifolds.
* **Speaker Note:** *"Welcome. Today we present Phase 1 of our project, detailing data cleaning, literature review, EDA, feature engineering, theoretical rigor, and empirical baselines."*

---

## Slide 2: Data Cleaning (Physics-Informed Sanity & Anomaly Flagging)
* **Sensor Physical Constraints:**
  * $\text{pH} \in [0.0, 14.0]$: In natural systems, readings outside 0–14 represent electrode drift (raw data peaked at 28.8).
  * $\text{Temperature} \in [-5.0^\circ\text{C}, 50.0^\circ\text{C}]$: Flagging uncalibrated thermal spikes (up to $97.5^\circ\text{C}$).
  * $\text{Dissolved Oxygen} \ge 0.0\text{ mg/L}$: Negative readings indicate galvanic zero-point drift.
  * $\text{Detection Limits}$: Left-censored zeros in Ammonia, Nitrate, Orthophosphate, and BOD.
* **Non-Destructive Protocol (`flag_and_clean_sensor_anomalies`):**
  * Convert physical violations to `NaN` and generate boolean indicators (`[sensor]_is_sensor_fault = 1`).
  * Preserves sample volume while signaling hardware malfunction to downstream models.
* **Speaker Note:** *"Cleaning is our first priority. Rather than discarding entire rows, our non-destructive protocol preserves statistical power while explicitly tracking sensor failures."*

---

## Slide 3: Literature Review & The Inductive Bias Gap
* **Breiman (2001) – Random Forests:**
  * Variance reduction through bagging decorrelated trees on random feature subspaces.
* **Chen & Guestrin (2016) – XGBoost:**
  * Gradient tree boosting optimized via second-order Taylor expansions with $\ell_1$ and $\ell_2$ leaf regularization.
* **Lundberg et al. (2020) – TreeSHAP:**
  * Exact coalitional game-theoretic Shapley attributions, eliminating Gini impurity cardinality bias.
* **The Inductive Bias Gap (Motivation for Phase 2):**
  * Trees partition feature space with axis-aligned orthogonal cuts ($x_j \le \theta$).
  * Chemical reaction surfaces and oxygen solubility follow smooth diagonal curves.
  * Trees require deep, high-variance staircases to approximate smooth diagonal manifolds.
* **Speaker Note:** *"Recognizing the orthogonal split limitation of tree ensembles provides the theoretical justification for introducing regularized MLPs in Phase 2."*

---

## Slide 4: Dataset Quality & Exploratory Data Analysis (EDA)
* **Dataset Scale:** 2,827,977 national water monitoring observations across 61,689 geographic stations (`Area`).
* **Environmental Diversity:** 12 waterbody types (River: 65.5%, Effluent: 21.3%, Lake: 5.4%, Estuarine: 1.7%).
* **Target Distributions:**
  * *Continuous Target (`CCME_Values`):* Range $[31.3, 100.0]$, Mean $= 85.05$, Std $= 17.65$.
  * *Discrete Target (`CCME_WQI`):* Excellent (36.1%), Good (36.5%), Marginal (11.6%), Fair (11.2%), Poor (4.5%).
* **Correlation Structure:** No single sensor dominates ($|r| < 0.75$ across all individual features), confirming the need for multi-sensor nonlinear modeling.
* **Speaker Note:** *"EDA confirms that water quality is a multi-parameter phenomenon requiring joint modeling of oxygen, organic waste, and nutrient kinetics."*

---

## Slide 5: Feature Engineering & Zero-Leakage Pipeline
* **Domain Chemical Interactions:**
  * *Temperature-dependent DO Saturation:* Theoretical solubility polynomial ($DO_{\text{sat}}(T)$), saturation ratio, and oxygen deficit.
  * *Organic Stress Ratio:* $BOD / (DO + 10^{-3})$.
  * *Nitrogen Speciation Fractions:* Ammonia fraction and Nitrate fraction relative to Total Nitrogen.
  * *Nutrient Enrichment Product:* Nitrate $\times$ Orthophosphate (eutrophication accelerator).
* **Automated Data Leakage Audit (`audit_data_leakage`):**
  * Evaluates Pearson correlation against target indices; flags any feature with $|r| \ge 0.98$. Zero leakage confirmed.
* **Split-Aware Imputation (`SplitAwareImputer`):**
  * Imputers fit exclusively on `X_train`, preventing lookahead leakage into test folds.
* **Speaker Note:** *"By encoding chemical solubility laws and strictly isolating training folds, we guarantee physical grounding and mathematical integrity."*

---

## Slide 6: Theoretical Rigor
* **Random Forest (Variance Minimization):**
  $$\text{Var}(\bar{f}_{\text{RF}}(x)) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
  * Driving tree correlation $\rho \to 0$ through bootstrap aggregating and random subspace feature selection.
* **XGBoost (Second-Order Taylor Regularization):**
  $$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
  * Analytical leaf weight: $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$.
  * Structural penalty $\gamma T$ controls model complexity and prevents overfitting.
* **Speaker Note:** *"We contrast bagging's variance reduction against boosting's sequential bias reduction with exact second-order curvature."*

---

## Slide 7: Model Application & Empirical Benchmark Results
* **Continuous Regression Benchmark (`CCME_Values`):**
  * *Dummy (Mean Baseline):* MAE = 13.92, RMSE = 17.45, $R^2 = -0.0001$
  * *Random Forest:* MAE = 0.2544, RMSE = 0.7306, $R^2 = 0.9982$
  * *XGBoost:* **MAE = 0.1520, RMSE = 0.4926, $R^2 = 0.9992$ (Superior)**
* **Discrete Classification Benchmark (`CCME_WQI`, 5 Quality Tiers):**
  * *Dummy (Stratified Baseline):* Accuracy = 29.09%, Macro F1 = 0.1972
  * *Random Forest:* **Accuracy = 99.29%, Macro F1 = 0.9857 (Top performer)**
  * *XGBoost:* Accuracy = 98.84%, Macro F1 = 0.9813
* **Probabilistic Calibration:** Reliability diagram demonstrates tight confidence calibration along the diagonal.
* **Speaker Note:** *"XGBoost excels in continuous index regression with an RMSE under 0.50, while Random Forest achieves 99.29% accuracy on discrete quality tiers."*

---

## Slide 8: GitHub Repository & Code Quality
* **Repository Architecture:**
  * Clean, modular package: `src/data/`, `src/features/`, `src/models/`.
  * Centralized configuration: `configs/config.yaml`.
  * 4 Dedicated Jupyter Notebooks: `01_data_cleaning`, `02_dataset_quality_and_eda`, `03_feature_engineering`, `04_model_application`.
* **Automated Test Suite:**
  * 5/5 unit tests passing under `pytest` (verifying DO saturation physics, bounds detection, leakage alerts, and imputer isolation).
* **Clean Git History:**
  * Atomic, conventional commit messages pushed to GitHub remote (`Sadhvik999/AML-project`).
* **Speaker Note:** *"Our repository enforces industry standards with automated CI tests, reproducible YAML configs, and modular source files."*

---

## Slide 9: Project Presentation & Viva Voce Defense
* **Key Defense Topics Prepared:**
  1. *Non-destructive sensor fault flagging rationale.*
  2. *Dual continuous vs discrete target formulation justification.*
  3. *Elimination of WQI formula leakage.*
  4. *Expected Calibration Error (ECE) and reliability diagrams for risk management.*
  5. *TreeSHAP attributions: Orthophosphate and Nitrogen identified as primary degradation drivers.*
* **Speaker Note:** *"We have comprehensively documented defense answers for every examiner question in our dedicated Viva Voce guide."*

---

## Slide 10: Project Report (IEEE Format) & Phase 2 Roadmap
* **IEEE Format Project Report:**
  * Formatted in official two-column `IEEEtran` specification ([`reports/report_ieee/main.tex`](reports/report_ieee/main.tex)).
  * Covers Abstract, Literature Review, Cleaning, Theory, Benchmark Tables, and TreeSHAP.
* **Roadmap to Phase 2 (Deep Neural Networks):**
  * Deploy a small regularized Multi-Layer Perceptron (MLP) with LayerNorm and smooth activations (GELU/SiLU).
  * Empirically test whether deep neural representations can improve upon tree models on smooth chemical interaction manifolds.
* **Speaker Note:** *"Phase 1 provides a rigorously validated benchmark, setting the stage for evaluating deep architectures in Phase 2."*
