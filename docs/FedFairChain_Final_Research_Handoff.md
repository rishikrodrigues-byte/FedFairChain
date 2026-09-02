# FedFairChain — Final Research Handoff

## 1. Final Title

FedFairChain: A Federated Classifier-Chain Approach for Multi-Label Chronic Disease Prediction

---

## 2. Final Disease Targets

The final system predicts three chronic disease labels:

1. Diabetes
2. Cardiovascular Disease (CVD)
3. Hypertension

Kidney disease is NOT part of the final implementation.

---

## 3. Final Classifier-Chain Order

DIABETES → CVD → HYPERTENSION

All six chain permutations were evaluated during development.

The unconstrained highest validation Macro PR-AUC was obtained by:

CVD → DIABETES → HYPERTENSION

However, the project architecture required Diabetes to be the anchor/first condition.

Among Diabetes-first alternatives, the selected chain was:

DIABETES → CVD → HYPERTENSION

Do not claim that this was the overall best of all six orders.

---

## 4. Dataset

Dataset:
CDC NHANES 2017–March 2020 Pre-Pandemic

Adults aged 20+ in master dataset:
9,232

Final participants with all three valid labels:
8,652

Final split:

- Training: 6,056
- Validation: 1,298
- Test: 1,298

The final test set remained untouched during:

- model training;
- chain-order selection;
- threshold tuning;
- federated-round selection;
- fairness-weight selection.

---

## 5. Final Label Distribution

Final modelling cohort: 8,652 participants.

Diabetes:
- Positive: 1,383
- Negative: 7,269
- Prevalence: 15.98%

Hypertension:
- Positive: 3,132
- Negative: 5,520
- Prevalence: 36.20%

CVD:
- Positive: 1,110
- Negative: 7,542
- Prevalence: 12.83%

---

## 6. Final Prediction Features

The final model uses 15 predictors:

1. AGE
2. BMI
3. WAIST_CM
4. SBP_MEAN
5. DBP_MEAN
6. HBA1C
7. TOTAL_CHOLESTEROL
8. HDL_CHOLESTEROL
9. CREATININE
10. BUN
11. ALBUMIN
12. URIC_ACID
13. SERUM_GLUCOSE
14. SMOKING_STATUS
15. PHYSICAL_ACTIVITY_ANY

SEX is excluded from prediction features.

SEX is retained only for fairness-related training analysis and evaluation.

---

## 7. Preprocessing

Missing numerical values:
Training-set median imputation

Scaling:
StandardScaler

The fitted preprocessing pipeline is frozen and reused during validation, testing and application inference.

Important limitation:

For this prototype, preprocessing statistics were fitted on the overall development training set and then reused for simulated federated clients.

Therefore, the implementation is not an end-to-end privacy-preserving federated preprocessing system.

---

## 8. Model Type

The core classifier is:

SGDClassifier(loss="log_loss")

The lightweight linear classifier was chosen because its coefficients and intercepts can be directly aggregated with Federated Averaging.

Classifier-chain training uses teacher forcing:

During training:
true previous labels are passed to downstream classifiers.

During inference:
predicted previous labels are passed downstream unless a diagnosis is already known.

---

## 9. Centralized Models

M1:
Independent Centralized Classifiers

M2:
Centralized Classifier Chain

Final centralized chain:

DIABETES → CVD → HYPERTENSION

---

## 10. Federated Learning

M3:
Federated Classifier Chain

Training data were partitioned into four simulated non-IID hospital nodes.

Hospital 1:
1,767 participants

Hospital 2:
1,697 participants

Hospital 3:
1,325 participants

Hospital 4:
1,267 participants

Total:
6,056 training participants

The nodes were generated using a Dirichlet-based non-IID partition over multi-label disease patterns.

Training:

- 20 federated rounds
- 2 local epochs per round
- constant learning rate = 0.01
- FedAvg aggregation
- sample-count-weighted parameter averaging

Best standard federated validation round:
17

---

## 11. Fairness-Aware FedFairChain

M4:
FedFairChain

FedFairChain combines:

- federated classifier-chain training;
- sex-label fairness-aware reweighting;
- diagnosis-aware inference;
- SHAP explainability.

Fairness-aware weights are computed from aggregated SEX × target-label counts.

SEX is NOT supplied as an inference feature.

The fairness method should be described as:

sex-label reweighting / fairness-aware sample reweighting.

Do not claim that FedFairChain guarantees fairness.

---

## 12. Final Frozen Thresholds

Diabetes:
0.68

CVD:
0.62

Hypertension:
0.30

These thresholds were selected using validation data and were NOT tuned on the final test set.

---

## 13. Final Held-Out Model Comparison

| Model | Macro ROC-AUC | Macro PR-AUC | Macro F1 | Micro F1 | Hamming Loss | Exact Match |
|---|---:|---:|---:|---:|---:|---:|
| M1 Independent Centralized | 0.8556 | 0.6297 | 0.6120 | 0.6251 | 0.1944 | 0.5516 |
| M2 Centralized Chain | 0.8567 | 0.6305 | 0.6129 | 0.6294 | 0.1975 | 0.5478 |
| M3 Federated Chain | 0.8668 | 0.6456 | 0.6209 | 0.6459 | 0.1723 | 0.5948 |
| M4 FedFairChain | 0.8662 | 0.6420 | 0.6199 | 0.6421 | 0.1793 | 0.5794 |

Interpretation:

M3 achieved the strongest overall predictive performance.

M4 retained very similar predictive performance while improving average fairness measurements.

Do not claim that M4 outperformed M3 on all predictive metrics.

---

## 14. Final FedFairChain Per-Disease Results

### Diabetes

ROC-AUC:
0.9327

PR-AUC:
0.7944

Precision:
0.7539

Recall:
0.6923

F1:
0.7218

Balanced Accuracy:
0.8246

### Hypertension

ROC-AUC:
0.8299

PR-AUC:
0.6912

Precision:
0.6116

Recall:
0.8124

F1:
0.6978

Balanced Accuracy:
0.7602

### CVD

ROC-AUC:
0.8360

PR-AUC:
0.4403

Precision:
0.3447

Recall:
0.6084

F1:
0.4401

Balanced Accuracy:
0.7194

CVD is the most difficult label and has the lowest PR-AUC.

---

## 15. Final Fairness Results

Mean absolute fairness gaps:

### Standard Federated Chain M3

Demographic Parity gap:
0.0667

Equal Opportunity gap:
0.1316

False Positive Rate gap:
0.0503

### FedFairChain M4

Demographic Parity gap:
0.0532

Equal Opportunity gap:
0.1211

False Positive Rate gap:
0.0404

Approximate changes:

Demographic Parity:
~20% reduction

Equal Opportunity:
~8% reduction

False Positive Rate:
~20% reduction

Interpretation:

FedFairChain improved average fairness measurements with only a small predictive-performance trade-off.

Fairness improvements were not uniform across every disease.

Important example:

CVD Equal Opportunity gap:

M3:
0.0743

M4:
0.0932

Therefore, do NOT claim that every fairness metric improved for every disease.

---

## 16. Diagnosis-Aware Inference

Each disease can be supplied as:

- Known Positive
- Known Negative
- Unknown

If known:

the system does not predict that disease again.

Its known binary status is inserted into the classifier chain.

If unknown:

the trained classifier predicts it normally.

Because the chain is directional:

DIABETES → CVD → HYPERTENSION

known Diabetes can affect:
- CVD
- Hypertension

known CVD can affect:
- Hypertension

known Hypertension cannot change:
- Diabetes
- CVD

There is no backward information propagation.

---

## 17. Final Diagnosis-Aware Test Result

Strongest held-out scenario:

Known Diabetes + Known CVD
→ Predict Hypertension

Standard Hypertension PR-AUC:
0.6912

Diagnosis-aware PR-AUC:
0.7234

Improvement:
+0.0323

Standard F1:
0.6978

Diagnosis-aware F1:
0.7084

Improvement:
+0.0106

This demonstrates that reliable upstream diagnosis information can improve downstream prediction.

---

## 18. Explainability

SHAP is applied independently to each classifier-chain stage.

Diabetes receives:
15 predictors

CVD receives:
15 predictors
+ DIABETES_CONTEXT

Hypertension receives:
15 predictors
+ DIABETES_CONTEXT
+ CVD_CONTEXT

Important global SHAP findings:

### Diabetes

Important features included:

- HbA1c
- Age
- Total cholesterol
- BMI
- Physical activity
- systolic BP

### CVD

Important features included:

- Age
- Smoking status
- Total cholesterol
- BUN
- BMI
- HDL cholesterol
- Diabetes context

### Hypertension

Important features included:

- Age
- Diabetes context
- CVD context
- BMI
- systolic BP
- diastolic BP
- uric acid

SHAP explains model behavior.

It must NOT be interpreted as causal medical evidence.

---

## 19. Application

Final application stack:

FastAPI backend
+
HTML/CSS/JavaScript web interface

Input:
15 clinical/lifestyle features

Optional:
Known Diabetes
Known CVD
Known Hypertension

Output:

- disease probability for unknown diagnoses;
- positive/negative prediction;
- known diagnosis status where supplied;
- classifier-chain context;
- per-disease SHAP explanations;
- research-use disclaimer.

---

## 20. Final Contribution Statement

FedFairChain integrates several components into one chronic-disease multi-label prediction framework:

1. multi-label classifier-chain modelling;
2. simulated non-IID federated learning;
3. fairness-aware sex-label reweighting;
4. diagnosis-aware classifier-chain inference;
5. per-label SHAP explainability;
6. an end-to-end web prediction application.

Novelty should NOT be claimed as "first ever" unless supported by a dedicated literature review.

Use wording such as:

"the proposed FedFairChain framework"

or

"the proposed diagnosis-aware inference mechanism."

---

## 21. Privacy Wording

Safe wording:

"Raw patient-level rows remain within their simulated hospital partitions during federated model training, while model parameters are aggregated centrally using FedAvg."

Do NOT write:

"FedFairChain guarantees complete patient privacy."

Do NOT write:

"Data can never be reconstructed."

The project does not implement:

- differential privacy;
- secure aggregation;
- homomorphic encryption;
- cryptographic privacy guarantees.

---

## 22. Main Limitations

1. Hospital nodes are simulated from NHANES rather than real institutions.
2. NHANES target labels contain self-reported diagnosis information.
3. CVD is represented as a composite target.
4. Federated preprocessing is not independently performed at each hospital.
5. No formal differential privacy or secure aggregation is implemented.
6. Fairness improvements vary across targets and metrics.
7. Results are derived from one NHANES cohort.
8. Clinical deployment and external validation were not performed.
9. SHAP explanations are associational/model-based, not causal.
10. The application is a research prototype and not a diagnostic device.

---

## 23. Final Research Figures

Use figures from:

results/final_figures/

Main figures:

01_final_model_performance.png
02_fedfairchain_per_disease.png
03_fairness_m3_vs_m4.png
04_equal_opportunity_by_disease.png
05_federated_training_convergence.png
06_diagnosis_aware_improvement.png
07_exact_match_accuracy.png
08_hamming_loss.png

SHAP:

results/shap_diabetes_global.png
results/shap_cvd_global.png
results/shap_hypertension_global.png

---

## 24. Final Status

Implementation audit:

Passed:
88

Warnings:
0

Failures:
0

Final status:

FEDFAIRCHAIN IMPLEMENTATION AUDIT PASSED

The modelling pipeline and application are frozen after held-out test evaluation.