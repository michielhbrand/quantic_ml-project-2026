# Malware Detection ML Project

Machine learning-based static malware detection — Quantic Introduction to Machine Learning Project 2026.

**Live application:** https://malware-detector-idgb.onrender.com

---

## Overview

This project builds and evaluates seven machine learning models that classify Windows PE executable files as malware or goodware, selects the best-performing model, and deploys it as a Flask web application with a CI/CD pipeline.

---

## Models

| # | Model | Family |
|---|-------|--------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Tree |
| 3 | Random Forest | Ensemble (bagging) |
| 4 | PyTorch MLP | Neural network |
| 5 | XGBoost | Ensemble (boosting) |
| 6 | LightGBM | Ensemble (boosting) |
| 7 | CatBoost | Ensemble (boosting) |

All models are evaluated with 10-fold stratified cross-validation. The best model by CV AUC is selected for production deployment.

---

## Results

See [`evaluation-and-design.md`](evaluation-and-design.md) for the full CV results table and design decisions.

**Best model: XGBoost**

| Metric | Hold-out test set (20%) |
|--------|------------------------|
| AUC | 0.9976 |
| Accuracy | 98.71% |

---

## Repository Structure

```
quantic_ml-project-2026/
├── dataset/
│   └── brazilian-malware-dataset/
│       └── brazilian-malware.csv        # ~50k PE file feature rows
├── frontend/
│   ├── app.py                           # Flask application
│   ├── requirements.txt                 # Web app dependencies
│   └── templates/
│       └── index.html                   # Web UI
├── ml_models/                           # Jupyter notebooks (one per model)
├── models/
│   ├── xgboost.pkl                      # Production model (best)
│   ├── scaler.pkl                       # StandardScaler (fit on training data)
│   ├── best_model.txt                   # Name of best model
│   ├── results.json                     # CV + test metrics for all 7 models
│   └── test_set.csv                     # 20% hold-out test set (unscaled)
├── tests/
│   ├── test_unit.py                     # Unit tests (preprocessing, model wrapper)
│   ├── test_integration.py              # Integration tests (Flask routes)
│   └── test_smoke.py                    # Smoke tests (health, predict, model_info)
├── .github/workflows/
│   ├── ci-feature.yml                   # Runs tests on feature branch pushes
│   └── ci-main.yml                      # Runs tests + deploy job on main
├── train_models.py                      # Full training pipeline
├── requirements.txt                     # Training dependencies
├── deployed.md                          # Live URL
├── evaluation-and-design.md            # CV results, test metrics, design decisions
└── ai-tooling.md                        # AI tools used
```

---

## Local Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train all models

```bash
python train_models.py
```

This loads the dataset, fits a `StandardScaler` on the training split, trains all 7 models with 10-fold CV, evaluates on the hold-out test set, and writes artefacts to `models/`.

### 4. Run the web application locally

```bash
cd frontend
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Web Application

The Flask app exposes two sections:

**1. Single Instance Prediction**
Enter PE feature values as JSON. Use "Load Goodware Demo" or "Load Malware Demo" to pre-fill with verified sample rows from the dataset.

**2. Batch Prediction**
Upload a CSV file. If the file contains a `Label` column, the app also computes AUC, accuracy, and a confusion matrix. The file `models/test_set.csv` (10,037 rows) can be used to reproduce the hold-out test results.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/predict_single` | Predict a single instance (JSON body) |
| POST | `/predict_batch` | Batch prediction / evaluation (CSV upload) |
| GET | `/health` | Health check |
| GET | `/model_info` | Loaded model name and all CV results |

---

## CI/CD Pipeline

Branch protection on `main` requires all changes to go through a pull request with the `test` CI check passing.

| Trigger | Workflow | Jobs |
|---------|----------|------|
| Push to any feature branch | `ci-feature.yml` | `test` (pytest) |
| Pull request → `main` | `ci-main.yml` | `test` (pytest) |
| Push to `main` (after merge) | `ci-main.yml` | `test` → `deploy` (if tests pass) |

### Running tests locally

```bash
pip install pytest
pytest tests/ -v --tb=short
```

---

## Reproducibility

- `random_state=42` used for all train/test splits, cross-validation, and tree-based models
- `StandardScaler` is fit only on training data; the fitted scaler is saved and reused at inference time
- All dependencies are pinned in `requirements.txt` and `frontend/requirements.txt`
