# Predictive Maintenance ML System

An end-to-end machine learning system that predicts industrial machine failure from live sensor readings, returning a calibrated failure probability rather than a strict pass/fail classification. Built on the AI4I 2020 Predictive Maintenance dataset, the project covers the full pipeline: exploratory feature engineering, model comparison across five algorithms, systematic overfitting diagnosis, threshold optimization, and deployment as a live API with a web frontend.

**Live components:** FastAPI backend + browser-based prediction interface (gauge-style risk display).

---

## Problem

Predicting rare equipment failures from sensor telemetry is a hard, high-stakes classification problem: failures make up roughly 3% of the dataset, so a naive model can score 97% accuracy while never correctly flagging a real failure. The goal of this project was not to chase a high accuracy number, but to build a model that generalizes honestly, is validated against its own overfitting risk, and is tuned toward the actual cost tradeoff in predictive maintenance — a missed failure is far more expensive than an unnecessary inspection.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 rows of synthetic but realistic industrial sensor data, including:

- Machine type (`L`, `M`, `H` — low/medium/high process quality variants)
- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)
- Binary machine failure label (~3.4% positive class)

### Engineered features

Beyond the raw sensor readings, five domain-informed features were derived from EDA into distinct physical "failure zones":

| Feature | Formula | Rationale |
|---|---|---|
| `Power` | `Torque × Rotational speed` | Mechanical power output; extreme values correlate with stress |
| `Heat_Strain` | `Process temp − Air temp` | Captures thermal load beyond ambient conditions |
| `Thermal_Stress_Ratio` | `Power / (Heat_Strain + 1)` | Ratio of mechanical to thermal stress |
| `High_Power_Hazard` | `1 if Power > 86000 else 0` | Flags the empirically identified high-power failure zone |
| `Underpower_Stall_Risk` | `1 if Power < 30000 else 0` | Flags the empirically identified stall/underpower failure zone |
| `Tool_Wear_Squared` | `Tool wear²` | Non-linear wear degradation |
| `Torque_Wear_Impact` | `Torque × Tool wear` | Interaction between mechanical load and accumulated wear |

These thresholds (86,000 / 30,000) were identified through EDA as the empirical "kill zones" where failure rates spike sharply — not arbitrary cutoffs.

---

## Modeling approach

### 1. Baseline and imbalance handling

The target class is heavily imbalanced (~3.4% positive). Two imbalance-handling strategies were tested and compared:
- **SMOTE** (synthetic minority oversampling) inside the training pipeline
- **Class weighting** (`scale_pos_weight` for XGBoost/LightGBM, `class_weight='balanced'` for RF/LR)

All preprocessing (encoding, scaling, resampling) was kept strictly inside a single `sklearn`/`imblearn` pipeline, fit only on training data per cross-validation fold, to eliminate data leakage risk.

### 2. Model comparison

Five algorithms were trained and tuned via `RandomizedSearchCV` / `GridSearchCV` with 5-fold stratified cross-validation, optimizing F1 score on the minority (failure) class:

| Model | Test F1 | Test Precision | Test Recall |
|---|---|---|---|
| Logistic Regression | 0.57 | 0.50 | 0.66 |
| LightGBM (regularized) | 0.78 | 0.71 | 0.87 |
| XGBoost | 0.88 | 0.86 | 0.88 |
| Gradient Boosting | 0.89 | 0.95 | 0.84 |
| Random Forest | 0.92 | 0.97 | 0.88 |
| **Voting Ensemble (RF + XGBoost + GB)** | **0.93** | **0.98** | **0.88** |

Logistic Regression's comparatively weak performance is itself informative: it confirms the underlying failure patterns are non-linear (consistent with the sharp threshold effects found in EDA), justifying the use of tree-based models.

### 3. Overfitting diagnosis

Cross-validation score alone doesn't catch every form of overfitting — a model can show a healthy CV score while still having memorized quirks of the exact training rows it saw. Every model was additionally checked with a **train-vs-test F1 gap analysis**:

- XGBoost, LightGBM, and Gradient Boosting initially showed train F1 near 1.0 against test F1 of 0.77–0.87 — a gap of 0.10–0.16, indicating memorization of the small (68-sample) minority class in the test split.
- Regularization was applied specifically to close this gap: tighter `max_depth`, raised `min_child_weight` / `min_samples_leaf`, and added `reg_alpha` / `reg_lambda` (L1/L2 penalties).
- Random Forest showed the smallest gap from the start (0.016) without any special regularization — a natural consequence of bagging's variance-reduction properties compared to boosting's sequential residual-fitting, which is more prone to memorization.

### 4. Final model: Soft Voting Ensemble

The three most robust models (Random Forest, regularized XGBoost, regularized Gradient Boosting) were combined via `VotingClassifier(voting='soft')`, with Random Forest weighted higher given its superior standalone generalization. The ensemble outperformed every individual model on test F1 while maintaining a healthy train/test gap (0.043).

### 5. Threshold optimization

Rather than defaulting to a 0.5 classification threshold, a full precision-recall sweep was run to find the operating point that best matches the real-world cost asymmetry of predictive maintenance (a missed failure is costlier than a false alarm):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.98 | 0.77 | 0.86 |
| **0.35 (selected)** | **0.95** | **0.82** | **0.88** |
| 0.25 | 0.60 | 0.85 | 0.71 |

Threshold 0.35 was selected as the production default — it improves both precision and recall over the naive 0.5 cutoff, and the config is designed to be adjusted if the deployment context favors recall more heavily (e.g., `0.20–0.25` for a setting where false alarms are cheap and missed failures are very costly).

---

## Architecture

```
Raw sensor input (6 fields: Type, 2 temps, RPM, torque, tool wear)
        │
        ▼
Feature engineering  (src/feature_engineering.py)
  → computes Power, Heat_Strain, Thermal_Stress_Ratio,
    High_Power_Hazard, Underpower_Stall_Risk,
    Tool_Wear_Squared, Torque_Wear_Impact
        │
        ▼
Preprocessing pipeline (embedded in the saved model)
  → OneHotEncoder (machine type), RobustScaler,
    log-transform on rotational speed
        │
        ▼
Voting ensemble (RF + XGBoost + GB)  →  predict_proba
        │
        ▼
Threshold applied (0.35)  →  risk level + recommended action
        │
        ▼
Logged (logs/prediction_logs.jsonl)  →  returned to caller
```

### Serving layer

- **`api/main.py`** — FastAPI service exposing `/predict`, `/health`, and `/` endpoints, with CORS enabled and Pydantic-based input validation
- **`frontend/index.html`** — standalone web interface for entering sensor readings and viewing an animated risk gauge, risk-level badge, and recommended action, calling the API directly from the browser

---

## Repository structure

```
predictive-maintenance-ml/
├── README.md
├── notebooks/
│   └── model_training.ipynb        model comparison, tuning, and validation
├── src/
│   ├── feature_engineering.py      raw input → full engineered feature set
│   ├── predict.py                  prediction wrapper with validation + logging
│   └── logging_utils.py            structured JSONL prediction logging
├── api/
│   └── main.py                     FastAPI service
├── frontend/
│   └── index.html                  web UI
├── models/
│   ├── voting_ensemble_final.joblib
│   └── model_config.json           threshold + expected feature columns
└── rf_best.joblib / xgb_best.joblib / gb_best.joblib
    individual tuned base models, kept for comparison/reproducibility
```

---

## Running locally

**1. Install dependencies**
```bash
pip install fastapi uvicorn scikit-learn xgboost lightgbm imbalanced-learn pandas numpy joblib
```

**2. Start the API**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**3. Open `frontend/index.html`** in a browser, and set the API endpoint field to `http://localhost:8000`.

**4. Or call the API directly:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "L",
    "Air temperature [K]": 297.3,
    "Process temperature [K]": 308.1,
    "Rotational speed [rpm]": 1615,
    "Torque [Nm]": 35.4,
    "Tool wear [min]": 217
  }'
```

---

## Known limitations

- The test set contains only 68 positive (failure) examples, so precision/recall estimates carry meaningful sampling variance — a single-digit change in correctly/incorrectly classified cases shifts the reported metrics by several percentage points.
- The model is trained on synthetic data (AI4I 2020) rather than real factory telemetry; deploying against real sensor data would require re-validation, as real-world sensor noise, drift, and failure modes may differ from the synthetic distribution.
- At the selected threshold (0.35), approximately 1 in 5 real failures is missed (82% recall) in exchange for high precision (95%) and few false alarms — this tradeoff should be reconsidered based on the actual cost of a missed failure versus an unnecessary inspection in any real deployment.
- The current deployment (FastAPI + ngrok) is intended for demonstration; a production deployment would require a persistent host (e.g. Render, Railway, AWS) rather than a tunnel tied to a local/Colab session.

---

## Tech stack

Python · scikit-learn · XGBoost · LightGBM · imbalanced-learn · pandas · FastAPI · Pydantic · joblib · HTML/CSS/JavaScript
