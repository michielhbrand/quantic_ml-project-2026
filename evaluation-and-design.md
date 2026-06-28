# Evaluation and Design

## Dataset

- **Source:** Brazilian Malware Dataset (PE executable static features)
- **Size:** ~50,000 instances (29,065 malware / 21,116 goodware)
- **Features:** 22 numeric features extracted from PE headers (e.g. `Entropy`, `BaseOfCode`, `SizeOfCode`, `TimeDateStamp`, `DllCharacteristics`, etc.)
- **Target:** `Label` — 0 = goodware, 1 = malware
- **Primary metric:** AUC (Area Under the ROC Curve)
- **Secondary metric:** Accuracy

---

## Train / Test Split

- **80% training** / **20% hold-out test** — stratified by class label using `random_state=42`
- The test set was kept completely untouched until final evaluation
- The hold-out test set is available at `models/test_set.csv` (10,037 instances, unscaled, with `Label` column)

---

## Data Preprocessing and Feature Engineering

### Non-numeric columns dropped
The dataset contains string columns (`SHA1`, `FirstSeenDate`, `Identify`, `ImportedDlls`, `ImportedSymbols`) that are not useful as numeric ML features. These were dropped using `select_dtypes(include='number')`.

### Scaling
A `StandardScaler` was fit **only on training data** and then applied to validation folds and the test set. The fitted scaler is saved as `models/scaler.pkl` and is applied at inference time in the Flask app.

### Missing values
Missing values were filled with the column mean (training data only). No significant missing data was present.

### Feature selection
No additional dimensionality reduction was applied. All 22 numeric features were retained as they are all semantically meaningful PE header fields.

---

## Cross-Validation Results (10-fold stratified CV)

| Model | CV AUC (mean ± std) | CV Accuracy (mean ± std) |
|-------|---------------------|--------------------------|
| Logistic Regression | 0.8772 ± 0.0056 | 0.8136 ± 0.0052 |
| Decision Tree | 0.9791 ± 0.0021 | 0.9800 ± 0.0018 |
| Random Forest | 0.9976 ± 0.0008 | 0.9879 ± 0.0013 |
| PyTorch MLP | 0.9860 ± 0.0018 | 0.9478 ± 0.0033 |
| XGBoost | **0.9980 ± 0.0005** | 0.9871 ± 0.0020 |
| LightGBM | 0.9976 ± 0.0005 | 0.9838 ± 0.0020 |
| CatBoost | 0.9971 ± 0.0006 | 0.9839 ± 0.0017 |

**Best model by CV AUC: XGBoost (0.9980)**

---

## Final Hold-out Test Set Evaluation (XGBoost)

| Metric | Value |
|--------|-------|
| AUC | **0.9976** |
| Accuracy | **98.71%** |

### Confusion Matrix

|  | Predicted Goodware | Predicted Malware |
|--|-------------------|-------------------|
| **Actual Goodware** | 4,145 (TN) | 79 (FP) |
| **Actual Malware** | 50 (FN) | 5,763 (TP) |

- **False positive rate** (safe files wrongly flagged): 79 / 4,224 = 1.87%
- **False negative rate** (malware missed): 50 / 5,813 = 0.86%

---

## Design Decisions

### Model selection
XGBoost was selected as the production model based on having the highest CV AUC (0.9980) and the lowest variance (std 0.0005). Random Forest was a close second (CV AUC 0.9976) but XGBoost's slightly better generalisation on the test set (AUC 0.9976 vs 0.9970) confirmed the choice.

### Only the best model is deployed
Per the project brief, only the best-performing model is packaged into the web application. `models/xgboost.pkl` and `models/scaler.pkl` are committed to the repository; all other model artefacts are excluded via `.gitignore`.

### Reproducibility
- `random_state=42` used for all train/test splits, cross-validation folds, and tree-based models
- All dependencies are pinned in `requirements.txt` and `frontend/requirements.txt`
- The training script `train_models.py` reproduces all results from the raw CSV

### Web application inference pipeline
1. User submits feature values as JSON (single) or CSV (batch)
2. Features are aligned to the scaler's expected column order; missing features are filled with 0
3. `StandardScaler` transforms the features
4. XGBoost model produces a binary prediction (0 = goodware, 1 = malware)
5. If the uploaded CSV contains a `Label` column, AUC, accuracy, and a confusion matrix are also computed and displayed
