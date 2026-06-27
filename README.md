# Malware Detection ML Project

Machine Learning-based Static Malware Detection System with Web Interface

## Project Overview

This project implements a complete machine learning pipeline for malware detection, including:
- **7 ML Models**: 4 baseline + 3 additional high-performing models
- **Cross-validation**: 10-fold stratified CV for model selection
- **Web Application**: Flask-based UI for predictions and evaluation
- **Production Ready**: Model packaging and deployment

## Models Implemented

### Baseline Models (Required)
1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree** - Non-linear tree-based classifier
3. **Random Forest** - Ensemble of decision trees
4. **PyTorch MLP** - Neural network with 2 hidden layers

### Additional High-Performing Models
5. **XGBoost** - Gradient boosting (tree-based)
6. **LightGBM** - Fast gradient boosting (tree-based)
7. **CatBoost** - Gradient boosting with categorical support (tree-based)

All models span at least two different algorithm families as required.

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train Models

```bash
python train_models.py
```

This will:
- Load and preprocess the Brazilian malware dataset
- Split data into 80% train / 20% test (stratified)
- Train all 7 models with 10-fold cross-validation
- Evaluate on hold-out test set
- Save all models to `models/` directory
- Generate results summary

Expected output:
- `models/scaler.pkl` - Feature scaler
- `models/logistic_regression.pkl` - Trained LR model
- `models/decision_tree.pkl` - Trained DT model
- `models/random_forest.pkl` - Trained RF model
- `models/pytorch_mlp.pth` - Trained MLP weights
- `models/xgboost.pkl` - Trained XGB model
- `models/lightgbm.pkl` - Trained LGB model
- `models/catboost.pkl` - Trained CB model
- `models/results.json` - CV and test metrics for all models
- `models/best_model.txt` - Name of best performing model
- `models/test_set.csv` - Hold-out test set for demo

### 4. Run Web Application

```bash
cd frontend
python app.py
```

Or use the convenience script:
```bash
./run_app.sh
```

Access the application at: `http://localhost:5000`

## Web Application Features

### 1. Single Instance Prediction
- Manual feature entry via JSON
- Pre-filled demo data button
- Displays prediction: Malware or Goodware
- Shows which model was used

### 2. Batch Prediction
- Upload CSV file with multiple instances
- Returns predictions for all instances
- Supports files with or without labels

### 3. Model Evaluation
- Upload CSV file with "Label" column
- Displays performance metrics:
  - **AUC** (Area Under ROC Curve)
  - **Accuracy**
  - **Confusion Matrix**
- Can demo with `models/test_set.csv`

## Dataset Information

**Source**: Brazilian Malware Dataset
- **Total Instances**: ~50,000
- **Features**: 27 numeric features from PE file analysis
- **Target**: Binary (0=Goodware, 1=Malware)
- **Class Distribution**: ~42% Goodware, ~56% Malware

**Features Include**:
- BaseOfCode, BaseOfData, Characteristics
- Entropy, FileAlignment, ImageBase
- Machine, Magic, NumberOfSections
- Size metrics, Timestamps, etc.

## Model Training Protocol

1. **Data Split**: 80% train, 20% test (stratified by class)
2. **Preprocessing**: StandardScaler fit on training data only
3. **Cross-Validation**: 10-fold stratified CV on training set
4. **Metrics**: AUC (primary), Accuracy (secondary)
5. **Final Model**: Best CV model retrained on full training set
6. **Test Evaluation**: Single evaluation on held-out test set

## Project Structure

```
MLProject/
├── dataset/
│   └── brazilian-malware-dataset/
│       └── brazilian-malware.csv
├── frontend/
│   ├── app.py                    # Flask application
│   ├── templates/
│   │   └── index.html           # Web UI
│   ├── requirements.txt         # Frontend dependencies
│   └── README.md
├── models/                       # Generated after training
│   ├── *.pkl                    # Trained sklearn models
│   ├── pytorch_mlp.pth          # PyTorch model weights
│   ├── scaler.pkl               # Feature scaler
│   ├── results.json             # All model results
│   ├── best_model.txt           # Best model name
│   └── test_set.csv             # Hold-out test set
├── ml_models_code_examples/     # Reference implementations
├── train_models.py              # Main training script
├── requirements.txt             # Project dependencies
├── run_app.sh                   # Convenience script
└── README.md                    # This file
```

## API Endpoints

- `GET /` - Main web interface
- `POST /predict_single` - Single instance prediction
- `POST /predict_batch` - Batch prediction and evaluation
- `GET /health` - Health check (for CI/CD)
- `GET /model_info` - Get model information and results

## Expected Results

Based on the dataset and models, you should expect:
- **AUC**: 0.95 - 0.99 (excellent discrimination)
- **Accuracy**: 0.90 - 0.98 (high accuracy)
- **Best Models**: Typically XGBoost, LightGBM, or CatBoost

## Notes

- All models use `random_state=42` for reproducibility
- Preprocessing is fit only on training data to prevent leakage
- PyTorch MLP uses 2 hidden layers [64, 32] with dropout
- Cross-validation uses stratified folds to maintain class balance
- The best model is automatically selected based on CV AUC

## Troubleshooting

### Models not loading in web app
- Ensure you've run `train_models.py` first
- Check that `models/` directory exists with all files

### Import errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

### PyTorch installation issues
- For CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- For GPU: Follow PyTorch official installation guide

## Requirements

- Python 3.8+
- 4GB+ RAM (for model training)
- ~500MB disk space (for models and dataset)

## License

This is an educational project for the Introduction to Machine Learning course.
