# Project Presentation Slides: Water Quality Prediction via AML

**Phase 1: Machine Learning Evaluation & Baseline Architecture**  
*Course:* Applied Machine Learning (AML)  
*Format:* Slide Deck Structure & Speaker Notes

---

## Slide 1: Title & Project Overview
* **Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks
* **Subtitle:** Phase 1 – Applied Machine Learning (AML) Baseline & Systematic Validation
* **Core Problem:** Real-time water quality assessment is hampered by slow laboratory assays. We predict water quality indices and safety classes from multi-sensor physical and chemical readings.
* **Key Challenge:** Real-world multi-sensor telemetry suffers from noise, cross-sensor correlations, missing readings, and sensor drift.
* **Speaker Note:** *"Welcome. Today we present Phase 1 of our project, establishing a rigorous machine learning foundation with Random Forest and XGBoost, prior to introducing Deep Neural Networks in Phase 2."*

---

## Slide 2: The 'Hybrid' Logic & The Theoretical Gap
* **The AML Baseline:** Random Forest and XGBoost excel on heterogeneous tabular data with step-function thresholds.
* **Identifying the Gap:**
  * Tree models use axis-aligned orthogonal partitions ($x_j \le \theta$).
  * Chemical kinetics and sensor interactions form smooth diagonal manifolds (e.g., $DO_{sat}$ as a cubic function of Temperature).
  * Ensembles require deep staircases of splits to approximate smooth manifolds.
* **The Hybrid Vision:** Strong AML tabular baselines provide interpretable benchmarks, paving the way for small regularized MLPs in Phase 2.
* **Speaker Note:** *"While trees handle unscaled tabular data effortlessly, they struggle with smooth continuous curves. Recognizing this inductive bias gap motivates our rigorous benchmark."*

---

## Slide 3: Sensor Data Quality & Physical Bounds Auditing
* **Sensor Physics Constraints:**
  * $\text{pH} \in [0.0, 14.0]$: Acid-base scale bounds.
  * $\text{Dissolved Oxygen (DO)} \ge 0\text{ mg/L}$: Negative DO signals sensor zero-drift.
  * $\text{Turbidity} \ge 0\text{ NTU}$: Optical backscatter cannot be negative.
  * $\text{Conductivity} \ge 0\text{ }\mu\text{S/cm}$ and $\text{TDS} \ge 0\text{ mg/L}$.
* **Anomaly Handling Protocol:**
  * We avoid silent row deletion or arbitrary clipping.
  * We flag anomalies with binary indicators (`[sensor]_is_sensor_fault`) and set values to `NaN` for controlled imputation.
* **Speaker Note:** *"Garbage in, garbage out. Before running any model, we enforce strict environmental physics to capture sensor failures."*

---

## Slide 4: Data Leakage Prevention & Split-Aware Imputation
* **Leakage Vector:** Deriving a target WQI from components (e.g. pH, DO, Turbidity) and feeding those identical components without auditing yields trivial formula memorization ($R^2 \approx 1.0$).
* **Automated Leakage Audit:** Pre-training correlation audit flags features with $|r| \ge 0.98$.
* **Imputation Integrity:**
  * Zero cross-split leakage: Imputers (`IterativeImputer` / `KNNImputer`) fit strictly on $X_{train}$.
  * Missingness indicators (`col_missing`) retain information about missingness mechanisms (MAR/MNAR).
* **Speaker Note:** *"Our pipeline guarantees mathematical isolation between training folds and holdout sets, preventing optimistic performance illusions."*

---

## Slide 5: Theoretical Foundations: Random Forest & XGBoost
* **Random Forest (Breiman, 2001):**
  * Variance reduction via bagging: $\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$.
  * High stability against noisy tabular sensor readings.
* **XGBoost (Chen & Guestrin, 2016):**
  * Second-order Taylor expansion objective:
    $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \gamma T + \frac{1}{2}\lambda \sum w_j^2$$
  * Exact leaf weight solution: $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$.
  * $L_1$ ($\alpha$) and $L_2$ ($\lambda$) regularization prevent tree overfitting.
* **Speaker Note:** *"We contrast bagging's variance reduction against gradient boosting's sequential bias minimization with second-order gradients."*

---

## Slide 6: Experimental Design & Cross-Validation
* **Holdout Validation:** 80% Train, 20% Holdout Test set.
* **5-Fold Cross-Validation:** Stratified for classification; K-Fold/Group for regression.
* **Hyperparameter Optimization:**
  * Randomized search over depth, learning rate, subsample ratios, and regularization coefficients.
  * Baseline comparison against Dummy (Mean / Stratified) predictor to confirm true learned signal.
* **Speaker Note:** *"All hyperparameters are tuned inside inner cross-validation folds to avoid test set snooping."*

---

## Slide 7: Evaluation Framework & Empirical Results
* **Continuous Regression Benchmark (`CCME_Values`):**
  * *Dummy (Mean):* MAE = 13.92, RMSE = 17.45, $R^2 = -0.0001$
  * *Random Forest:* MAE = 0.2544, RMSE = 0.7306, $R^2 = 0.9982$
  * *XGBoost:* **MAE = 0.1520, RMSE = 0.4926, $R^2 = 0.9992$ (Superior)**
* **Discrete Classification Benchmark (`CCME_WQI`, 5 Quality Tiers):**
  * *Dummy (Stratified):* Accuracy = 29.09%, Macro F1 = 0.1972
  * *Random Forest:* **Accuracy = 99.29%, Macro F1 = 0.9857 (Top performer)**
  * *XGBoost:* Accuracy = 98.84%, Macro F1 = 0.9813
* **Probabilistic Calibration:** Reliability diagram shows tight confidence calibration along the diagonal.
* **Speaker Note:** *"XGBoost outperforms Random Forest in continuous index regression with an RMSE under 0.50, while Random Forest achieved 99.29% accuracy on quality classification."*

---

## Slide 8: Explainability with TreeSHAP
* **Why TreeSHAP?**
  * Overcomes Gini impurity bias toward continuous variables.
  * Grounded in coalitional game theory with Shapley efficiency and additivity.
* **Actionable Insights:**
  * Beeswarm plots reveal feature impact directionality (e.g. low pH severely reduces potability/WQI).
  * Feature interaction terms uncover joint sensor tipping points.
* **Speaker Note:** *"SHAP provides transparency, showing environmental regulators exactly why the model flags water as unsafe."*

---

## Slide 9: Road to Phase 2: Deep Neural Networks (MLP)
* **Learnings from Phase 1:** Strong tabular benchmark established with explicit performance ceilings.
* **Phase 2 Implementation:**
  * Regularized Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout, and LayerNorm.
  * Smooth activation functions (GELU / SiLU) to capture nonlinear chemical combinations.
  * Regression head for continuous WQI / Multi-task heads for index + classification tiers.
  * Side-by-side comparison on identical holdout splits.
* **Speaker Note:** *"Phase 1 sets the bar. Phase 2 introduces deep learning to see if neural representations can close the gap on smooth chemical manifolds."*

---

## Slide 10: Conclusion & Repository Deliverables
* **Modular Codebase:** Fully structured package (`src/data`, `src/features`, `src/models`).
* **Test-Driven Rigor:** Automated unit tests verifying physical bounds, leakage prevention, and imputer isolation.
* **Deliverables:** IEEE LaTeX Project Report, interactive Jupyter notebooks, reproducible CLI execution, and Viva Voce defense pack.
* **Speaker Note:** *"Thank you. We are now ready to demonstrate the pipeline and address questions."*
