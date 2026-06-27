# Machine Learning Models - Individual Notebooks

This directory contains individual Jupyter notebooks for each machine learning model used in the malware detection project.

## Overview

Each notebook follows the project brief's step-by-step instructions and includes:
1. Data loading and exploration
2. Preprocessing with proper train/test split (80/20)
3. Feature scaling (where appropriate)
4. Model training
5. 10-fold stratified cross-validation
6. Evaluation on hold-out test set
7. Feature importance analysis
8. Model serialization for production

## Notebooks

### Baseline Models (Required)

1. **[01_logistic_regression.ipynb](01_logistic_regression.ipynb)**
   - Simple linear model for binary classification
   - Fast training and prediction
   - Interpretable coefficients
   - Good baseline performance

2. **[02_decision_tree.ipynb](02_decision_tree.ipynb)**
   - Tree-based model with interpretable rules
   - No feature scaling required
   - Includes tree visualization
   - Prone to overfitting

3. **[03_random_forest.ipynb](03_random_forest.ipynb)**
   - Ensemble of decision trees
   - Reduces overfitting compared to single trees
   - Provides feature importance rankings
   - Includes Out-of-Bag (OOB) score

4. **[04_pytorch_mlp.ipynb](04_pytorch_mlp.ipynb)**
   - Deep learning neural network
   - Multi-layer perceptron with 2 hidden layers
   - Includes training visualization
   - Requires more data and computation

### Additional Models (High-Performance)

5. **[05_xgboost.ipynb](05_xgboost.ipynb)**
   - Extreme Gradient Boosting
   - State-of-the-art performance on structured data
   - Built-in regularization
   - Multiple importance metrics

6. **[06_lightgbm.ipynb](06_lightgbm.ipynb)**
   - Light Gradient Boosting Machine
   - Faster training than XGBoost
   - Lower memory usage
   - Leaf-wise tree growth

7. **[07_catboost.ipynb](07_catboost.ipynb)**
   - Categorical Boosting
   - Handles categorical features automatically
   - Robust to overfitting
   - **Note:** Requires Python 3.13 or lower

## How to Use

### Prerequisites

Make sure you have all dependencies installed:
```bash
# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r ../requirements.txt
```

### Running a Notebook

1. **Start Jupyter Notebook or JupyterLab:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open any notebook** from the list above

3. **Run all cells** sequentially (Cell > Run All) or step through them one by one

4. **Review results** including:
   - Cross-validation scores (AUC and Accuracy)
   - Test set performance
   - Confusion matrix
   - Feature importance

### Testing Individual Models

Each notebook is self-contained and can be run independently. This allows you to:
- Test each model separately
- Compare performance across models
- Experiment with different hyperparameters
- Understand how each algorithm works

## Model Comparison

After running all notebooks, compare the results:

| Model | Speed | Accuracy | Interpretability | Overfitting Risk |
|-------|-------|----------|------------------|------------------|
| Logistic Regression | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Low |
| Decision Tree | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | High |
| Random Forest | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Medium |
| PyTorch MLP | ⚡ | ⭐⭐⭐ | ⭐ | Medium-High |
| XGBoost | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ | Low |
| LightGBM | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ | Low |
| CatBoost | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ | Very Low |

## Output Files

Each notebook saves its trained model to the `../models/` directory:
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `random_forest.pkl`
- `pytorch_mlp.pth` (+ `pytorch_mlp_config.json`)
- `xgboost.pkl`
- `lightgbm.pkl`
- `catboost.pkl` (+ `catboost.cbm`)

The scaler is also saved as `scaler.pkl` for consistent preprocessing in production.

## Key Metrics

All models are evaluated using:
- **Primary Metric:** AUC (Area Under ROC Curve)
- **Secondary Metric:** Accuracy
- **Additional:** Confusion Matrix, Precision, Recall, F1-Score

## Tips

1. **Start with baseline models** (Logistic Regression, Decision Tree) to establish a baseline
2. **Try ensemble methods** (Random Forest, XGBoost, LightGBM) for better performance
3. **Experiment with PyTorch MLP** if you want to explore deep learning
4. **Use CatBoost** if you have categorical features or want robust performance
5. **Compare cross-validation results** to select the best model
6. **Test on hold-out set** only once to avoid overfitting

## Troubleshooting

### CatBoost Installation Issues
If you encounter issues with CatBoost on Python 3.14+:
```bash
# Create a new environment with Python 3.13
python3.13 -m venv venv_py313
source venv_py313/bin/activate
pip install -r ../requirements.txt
```

### Memory Issues
If you run out of memory:
- Reduce `n_estimators` in ensemble models
- Use smaller `batch_size` in PyTorch MLP
- Close other applications

### Slow Training
To speed up training:
- Use `n_jobs=-1` to utilize all CPU cores
- Reduce cross-validation folds (e.g., 5 instead of 10)
- Use GPU for PyTorch (if available)

## Next Steps

After testing individual models:
1. Compare results across all notebooks
2. Select the best performing model
3. Use the consolidated [`train_models.py`](../train_models.py) script to train all models at once
4. Deploy the best model using the Flask app in [`frontend/`](../frontend/)

## Project Structure

```
MLProject/
├── ml_models/                    # Individual model notebooks (you are here)
│   ├── 01_logistic_regression.ipynb
│   ├── 02_decision_tree.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_pytorch_mlp.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_catboost.ipynb
│   └── README.md
├── train_models.py              # Consolidated training script
├── frontend/                    # Flask web application
├── dataset/                     # Malware dataset
└── models/                      # Saved models (created after training)
```

## References

- Project Brief: [`ml_project_brief.pdf`](../ml_project_brief.pdf)
- Training Script: [`train_models.py`](../train_models.py)
- Web Application: [`frontend/app.py`](../frontend/app.py)
