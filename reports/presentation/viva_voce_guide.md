# Phase 1: Applied Machine Learning (AML) Viva Voce & Defense Guide

**Project Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
**Phase Focus:** Phase 1 – Machine Learning Evaluation (Baseline Tree Models, Theoretical Rigor, EDA, Feature Engineering & Validation)

---

## 1. Problem Formulation & Dataset Strategy

### Q1: What is the primary objective of this project, and how is Phase 1 scoped?
* **Answer:** The objective is to accurately predict water quality (either a continuous Water Quality Index $WQI \in [0, 100]$ or a discrete classification like Potability / Quality Tier) from multi-sensor physical and chemical readings (pH, dissolved oxygen, turbidity, temperature, conductivity, and dissolved solids).
* **Phase 1 Scope:** Phase 1 focuses exclusively on the **Applied Machine Learning (AML) baseline evaluation**. This establishes rigorous data quality auditing, physical constraint checks, split-isolated imputation, data leakage prevention, hyperparameter-tuned tree ensembles (Random Forest and XGBoost), metric benchmarking, probabilistic calibration, and explainability via TreeSHAP.

---

## 2. Theoretical Rigor & Model Justifications

### Q2: Why choose Random Forest and XGBoost as the baseline AML models?
* **Random Forest (Breiman, 2001):**
  * Employs *bagging* (Bootstrap Aggregating) and *random subspace feature selection*.
  * By training $B$ decorrelated decision trees and averaging their outputs ($\hat{f}_{RF}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)$), the variance of the ensemble decreases by a factor proportional to $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$, where $\rho$ is tree correlation.
  * Highly robust to tabular noise, unnormalized scales, and monotonic transformations.
* **XGBoost (Chen & Guestrin, 2016):**
  * Uses *gradient boosting* under a second-order Taylor approximation of the loss function:
    $$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
    where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$.
  * Built-in shrinkage (learning rate $\eta$), column subsampling, and explicit regularization ($\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$) prevent overfitting on small/medium tabular datasets.

### Q3: What is the theoretical "gap" between Tree Ensembles and Deep Neural Networks (MLP)?
* **Inductive Bias of Trees:** Tree-based models partition the input space using **axis-aligned orthogonal hyperplanes** ($x_j \le \theta$).
  * *Strength:* Fast, invariant to scale, handles tabular feature splits naturally.
  * *Limitation (The Gap):* In chemical systems, physical phenomena often follow smooth continuous manifolds (e.g. diagonal interactions like $x_1 + x_2 = c$ or polynomial rate equations). A decision tree requires an exponential staircase of splits to approximate a simple smooth diagonal boundary.
* **Role of the DL Model (Phase 2):** A regularized Multi-Layer Perceptron (MLP) learns continuous representations via smooth composite nonlinear activations $\sigma(W x + b)$, efficiently modeling smooth multi-sensor chemical kinetics, which complements the tabular strengths of tree models.

---

## 3. Data Quality, EDA & Physics-Informed Engineering

### Q4: How do you identify physically impossible sensor values and prevent silent model failures?
* We enforce natural domain bounds based on physical and chemical laws:
  * $\text{pH} \in [0, 14]$ (acidic to basic scale).
  * $\text{Dissolved Oxygen (DO)} \ge 0\text{ mg/L}$ (negative DO is a sensor zero-drift error).
  * $\text{Turbidity} \ge 0\text{ NTU}$ (negative turbidity is optical sensor fouling).
  * $\text{Conductivity} \ge 0\text{ }\mu\text{S/cm}$ and $\text{TDS} \ge 0\text{ mg/L}$.
* Rather than silently clipping or deleting rows, we implement `flag_and_clean_sensor_anomalies`: replacing impossible readings with `NaN` and appending a binary indicator column `[sensor]_is_sensor_fault = 1`, preserving sample count while alerting downstream estimators.

### Q5: What is Data Leakage in Water Quality modeling, and how is it audited?
* **Mechanism:** Many water datasets compute an index (e.g. CCME WQI or Weighted Arithmetic Index) using a deterministic formula based directly on pH, DO, and Turbidity. If a feature represents an intermediate sub-index ($q_i$) or has near-perfect correlation ($|r| \ge 0.98$), the model merely learns trivial inverse arithmetic rather than environmental dynamics.
* **Audit:** We run `audit_data_leakage()` before training, calculating Pearson correlation matrices and flagging any suspicious correlations or identity columns.

### Q6: Why must imputation be fitted strictly on the training fold?
* Pre-splitting imputation uses statistics (mean, median, or iterative regression covariance) calculated over the entire dataset, including the validation and test sets.
* This leaks test distribution parameters into training, causing artificially deflated test error and overfitting.
* Our `SplitAwareImputer` fits exclusively on `X_train` and transforms `X_test` with train statistics, alongside missingness indicators.

---

## 4. Evaluation Metrics & Calibration

### Q7: Why is accuracy alone insufficient for classification, and why do we report Expected Calibration Error (ECE)?
* **Class Imbalance & Cost:** Potable water vs non-potable water can be imbalanced; a naive classifier predicting the majority class achieves deceptively high accuracy. We report **Macro F1** and **Balanced Accuracy**.
* **Probabilistic Calibration (ECE):** In environmental health, decisions are risk-weighted (e.g. issuing a boil-water advisory). A model predicting $0.51$ probability shouldn't be treated with the same confidence as $0.99$.
* ECE partitions predictions into $M$ confidence bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  Lower ECE indicates reliable, calibrated posterior probabilities.

---

## 5. Explainability (TreeSHAP)

### Q8: Why prefer TreeSHAP over Gini Impurity (MDI)?
* **MDI Bias:** Mean Decrease in Impurity favors continuous, high-cardinality features and offers no directional insight (whether higher values increase or decrease the risk).
* **TreeSHAP:** Based on cooperative game theory Shapley values:
  $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
  It guarantees *efficiency*, *symmetry*, and *additivity*, providing exact local attributions and global beeswarm distributions showing directional effects.
