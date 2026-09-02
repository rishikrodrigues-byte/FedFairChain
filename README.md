# FedFairChain

**FedFairChain: A Federated Classifier-Chain Approach for Multi-Label Chronic Disease Prediction**

FedFairChain is a research prototype for multi-label prediction of:

- Diabetes
- Cardiovascular Disease (CVD)
- Hypertension

The system combines:

- multi-label classifier chains;
- simulated federated learning;
- fairness-aware training;
- diagnosis-aware inference;
- SHAP explainability;
- a FastAPI prediction service;
- a browser-based patient input interface.

> This project is intended for academic research only and is not a medical diagnostic system.

---

## 1. Final Architecture

```text
NHANES 2017–March 2020 Pre-Pandemic
                │
                ▼
       Dataset Construction
                │
                ▼
      Train / Validation / Test
         70% / 15% / 15%
                │
                ▼
     Independent Baseline Models
                │
                ▼
     Classifier-Chain Comparison
                │
                ▼
 Final Chain:
 DIABETES → CVD → HYPERTENSION
                │
                ▼
     Diagnosis-Aware Inference
                │
                ▼
   4 Simulated Non-IID Hospitals
                │
                ▼
        Local Model Training
                │
                ▼
               FedAvg
                │
                ▼
     Fairness-Aware Reweighting
                │
                ▼
           FedFairChain
                │
                ▼
         SHAP Explainability
                │
                ▼
       FastAPI + Web Interface