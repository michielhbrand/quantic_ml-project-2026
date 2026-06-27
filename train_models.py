"""
Machine Learning Pipeline for Malware Detection
Implements all required models for the project:
- Baseline: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP
- Additional: XGBoost, LightGBM, CatBoost
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("WARNING: CatBoost not available. Skipping CatBoost model.")
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

class MalwareDataset:
    """Handle data loading and preprocessing"""
    
    def __init__(self, data_path='dataset/brazilian-malware-dataset/brazilian-malware.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self):
        """Load data and perform initial preprocessing"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Separate features and target
        if 'Label' not in df.columns:
            raise ValueError("Label column not found in dataset")
        
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Handle non-numeric columns
        # Drop columns that are not useful for ML (SHA1, FirstSeenDate, Identify, etc.)
        # Use select_dtypes to handle all non-numeric types (object, StringDtype, etc.)
        non_numeric_cols = X.select_dtypes(exclude='number').columns.tolist()
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.select_dtypes(include='number')
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            X = X.fillna(X.mean())
        
        print(f"Final feature shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets (stratified)"""
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values


class PyTorchMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(PyTorchMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPipeline:
    """Main ML Pipeline for training and evaluating models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_sklearn_model(self, name, model, cv_folds=10):
        """Train and evaluate sklearn-compatible models with cross-validation"""
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        cv_results = cross_validate(
            model, self.X_train, self.y_train,
            cv=cv,
            scoring=['roc_auc', 'accuracy'],
            return_train_score=False,
            n_jobs=-1
        )
        
        cv_auc_mean = cv_results['test_roc_auc'].mean()
        cv_auc_std = cv_results['test_roc_auc'].std()
        cv_acc_mean = cv_results['test_accuracy'].mean()
        cv_acc_std = cv_results['test_accuracy'].std()
        
        print(f"Cross-Validation Results ({cv_folds}-fold):")
        print(f"  AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        print(f"  Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
        
        # Train on full training set
        model.fit(self.X_train, self.y_train)
        
        # Test set evaluation
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print(f"\nTest Set Results:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        
        self.models[name] = model
        self.results[name] = {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_acc_mean': cv_acc_mean,
            'cv_acc_std': cv_acc_std,
            'test_auc': test_auc,
            'test_acc': test_acc
        }
        
        return model
    
    def train_pytorch_mlp(self, name='PyTorch MLP', epochs=50, batch_size=256, cv_folds=10):
        """Train PyTorch MLP with cross-validation"""
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        input_dim = self.X_train.shape[1]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_aucs = []
        cv_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Train model for this fold
            model = PyTorchMLP(input_dim)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Prepare data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_fold_train),
                torch.FloatTensor(y_fold_train).unsqueeze(1)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation fold
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(X_fold_val)).numpy()
                val_preds = (val_outputs > 0.5).astype(int).flatten()
                
                fold_auc = roc_auc_score(y_fold_val, val_outputs)
                fold_acc = accuracy_score(y_fold_val, val_preds)
                
                cv_aucs.append(fold_auc)
                cv_accs.append(fold_acc)
        
        cv_auc_mean = np.mean(cv_aucs)
        cv_auc_std = np.std(cv_aucs)
        cv_acc_mean = np.mean(cv_accs)
        cv_acc_std = np.std(cv_accs)
        
        print(f"Cross-Validation Results ({cv_folds}-fold):")
        print(f"  AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        print(f"  Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
        
        # Train final model on full training set
        final_model = PyTorchMLP(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        final_model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Test set evaluation
        final_model.eval()
        with torch.no_grad():
            test_outputs = final_model(torch.FloatTensor(self.X_test)).numpy()
            test_preds = (test_outputs > 0.5).astype(int).flatten()
            
            test_auc = roc_auc_score(self.y_test, test_outputs)
            test_acc = accuracy_score(self.y_test, test_preds)
        
        print(f"\nTest Set Results:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        
        self.models[name] = final_model
        self.results[name] = {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_acc_mean': cv_acc_mean,
            'cv_acc_std': cv_acc_std,
            'test_auc': test_auc,
            'test_acc': test_acc
        }
        
        return final_model
    
    def train_catboost_model(self, name='CatBoost', cv_folds=10):
        """Train CatBoost with manual CV to avoid sklearn cross_validate incompatibility"""
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_aucs = []
        cv_accs = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            fold_model = cb.CatBoostClassifier(
                iterations=100,
                random_state=RANDOM_STATE,
                verbose=False
            )
            fold_model.fit(X_fold_train, y_fold_train)

            fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            fold_pred = fold_model.predict(X_fold_val)

            cv_aucs.append(roc_auc_score(y_fold_val, fold_proba))
            cv_accs.append(accuracy_score(y_fold_val, fold_pred))

        cv_auc_mean = np.mean(cv_aucs)
        cv_auc_std = np.std(cv_aucs)
        cv_acc_mean = np.mean(cv_accs)
        cv_acc_std = np.std(cv_accs)

        print(f"Cross-Validation Results ({cv_folds}-fold):")
        print(f"  AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        print(f"  Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")

        # Train final model on full training set
        final_model = cb.CatBoostClassifier(
            iterations=100,
            random_state=RANDOM_STATE,
            verbose=False
        )
        final_model.fit(self.X_train, self.y_train)

        y_pred = final_model.predict(self.X_test)
        y_pred_proba = final_model.predict_proba(self.X_test)[:, 1]

        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_acc = accuracy_score(self.y_test, y_pred)

        print(f"\nTest Set Results:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")

        self.models[name] = final_model
        self.results[name] = {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_acc_mean': cv_acc_mean,
            'cv_acc_std': cv_acc_std,
            'test_auc': test_auc,
            'test_acc': test_acc
        }

        return final_model

    def train_all_models(self):
        """Train all required models"""
        
        # Baseline Models
        print("\n" + "="*60)
        print("BASELINE MODELS")
        print("="*60)
        
        # 1. Logistic Regression
        self.train_sklearn_model(
            'Logistic Regression',
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        )
        
        # 2. Decision Tree
        self.train_sklearn_model(
            'Decision Tree',
            DecisionTreeClassifier(random_state=RANDOM_STATE)
        )
        
        # 3. Random Forest
        self.train_sklearn_model(
            'Random Forest',
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        )
        
        # 4. PyTorch MLP
        self.train_pytorch_mlp('PyTorch MLP')
        
        # Additional Models
        print("\n" + "="*60)
        print("ADDITIONAL HIGH-PERFORMING MODELS")
        print("="*60)
        
        # 5. XGBoost
        self.train_sklearn_model(
            'XGBoost',
            xgb.XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            )
        )
        
        # 6. LightGBM
        self.train_sklearn_model(
            'LightGBM',
            lgb.LGBMClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        )
        
        # 7. CatBoost (if available)
        if CATBOOST_AVAILABLE:
            self.train_catboost_model()
        else:
            print("\nSkipping CatBoost (not installed)")
    
    def get_best_model(self):
        """Return the best model based on CV AUC"""
        best_name = max(self.results, key=lambda x: self.results[x]['cv_auc_mean'])
        return best_name, self.models[best_name], self.results[best_name]
    
    def print_summary(self):
        """Print summary of all model results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'CV AUC':<20} {'CV Accuracy':<20} {'Test AUC':<12} {'Test Acc':<12}")
        print("-"*80)
        
        for name, results in self.results.items():
            cv_auc = f"{results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}"
            cv_acc = f"{results['cv_acc_mean']:.4f} ± {results['cv_acc_std']:.4f}"
            test_auc = f"{results['test_auc']:.4f}"
            test_acc = f"{results['test_acc']:.4f}"
            print(f"{name:<20} {cv_auc:<20} {cv_acc:<20} {test_auc:<12} {test_acc:<12}")
        
        print("="*80)
        
        best_name, _, best_results = self.get_best_model()
        print(f"\nBest Model (by CV AUC): {best_name}")
        print(f"  CV AUC: {best_results['cv_auc_mean']:.4f} ± {best_results['cv_auc_std']:.4f}")
        print(f"  Test AUC: {best_results['test_auc']:.4f}")
        print(f"  Test Accuracy: {best_results['test_acc']:.4f}")
    
    def save_models(self, output_dir='models'):
        """Save all models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving models to {output_dir}/...")
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'PyTorch MLP':
                model_file = output_path / f"{name.replace(' ', '_').lower()}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  Saved: {model_file}")
        
        # Save PyTorch model
        if 'PyTorch MLP' in self.models:
            torch_file = output_path / 'pytorch_mlp.pth'
            torch.save(self.models['PyTorch MLP'].state_dict(), torch_file)
            print(f"  Saved: {torch_file}")
        
        # Save results
        results_file = output_path / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Saved: {results_file}")
        
        # Save best model name
        best_name, _, _ = self.get_best_model()
        best_model_file = output_path / 'best_model.txt'
        with open(best_model_file, 'w') as f:
            f.write(best_name)
        print(f"  Saved: {best_model_file}")
        
        print("All models saved successfully!")


def main():
    """Main execution function"""
    print("="*80)
    print("MALWARE DETECTION ML PIPELINE")
    print("="*80)
    
    # Load and preprocess data
    dataset = MalwareDataset()
    X, y = dataset.load_and_preprocess()
    X_train, X_test, y_train, y_test = dataset.split_data(X, y)
    
    # Save scaler for later use
    Path('models').mkdir(exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print("Scaler saved to models/scaler.pkl")
    
    # Save test set for evaluation
    test_df = pd.DataFrame(X_test)
    test_df['Label'] = y_test
    test_df.to_csv('models/test_set.csv', index=False)
    print(f"Test set saved to models/test_set.csv ({len(test_df)} samples)")
    
    # Train all models
    pipeline = MLPipeline(X_train, X_test, y_train, y_test)
    pipeline.train_all_models()
    
    # Print summary
    pipeline.print_summary()
    
    # Save models
    pipeline.save_models()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
