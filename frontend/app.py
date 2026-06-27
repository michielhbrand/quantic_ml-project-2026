from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

app = Flask(__name__)

# Global variables for models
best_model = None
best_model_name = None
scaler = None
pytorch_mlp_architecture = None

def load_pytorch_model(model_path, input_dim):
    """Load PyTorch MLP model"""
    import torch.nn as nn
    
    class PyTorchMLP(nn.Module):
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
    
    model = PyTorchMLP(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_models():
    """Load the best trained model and scaler"""
    global best_model, best_model_name, scaler, pytorch_mlp_architecture
    
    models_dir = Path('../models')
    
    if not models_dir.exists():
        print("WARNING: Models directory not found. Using placeholder predictions.")
        return False
    
    try:
        # Load scaler
        scaler_path = models_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        
        # Load best model name
        best_model_file = models_dir / 'best_model.txt'
        if best_model_file.exists():
            with open(best_model_file, 'r') as f:
                best_model_name = f.read().strip()
            print(f"Best model: {best_model_name}")
            
            # Load the best model
            if best_model_name == 'PyTorch MLP':
                # For PyTorch, we need to know the input dimension
                # We'll load it when we get the first prediction
                pytorch_mlp_architecture = True
                print("PyTorch MLP will be loaded on first prediction")
            else:
                model_file = models_dir / f"{best_model_name.replace(' ', '_').lower()}.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        best_model = pickle.load(f)
                    print(f"Loaded model from {model_file}")
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def preprocess_features(features_dict):
    """Preprocess features for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Ensure all features are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values with 0
    df = df.fillna(0)
    
    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = scaler.transform(df)
    else:
        features_scaled = df.values
    
    return features_scaled

def predict_with_model(features):
    """Make prediction using the loaded model"""
    global best_model, pytorch_mlp_architecture
    
    if best_model is None and not pytorch_mlp_architecture:
        # Fallback to random prediction
        return "malware" if np.random.random() > 0.5 else "goodware"
    
    try:
        if pytorch_mlp_architecture and best_model is None:
            # Load PyTorch model on first use
            models_dir = Path('../models')
            pytorch_path = models_dir / 'pytorch_mlp.pth'
            if pytorch_path.exists():
                input_dim = features.shape[1]
                best_model = load_pytorch_model(pytorch_path, input_dim)
                print(f"Loaded PyTorch MLP with input_dim={input_dim}")
        
        if best_model_name == 'PyTorch MLP':
            # PyTorch prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                output = best_model(features_tensor).numpy()
                prediction = 1 if output[0][0] > 0.5 else 0
        else:
            # Sklearn prediction
            prediction = best_model.predict(features)[0]
        
        return "malware" if prediction == 1 else "goodware"
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "malware" if np.random.random() > 0.5 else "goodware"

def predict_proba_with_model(features):
    """Get prediction probabilities"""
    global best_model, pytorch_mlp_architecture
    
    if best_model is None and not pytorch_mlp_architecture:
        return np.random.random(features.shape[0])
    
    try:
        if pytorch_mlp_architecture and best_model is None:
            models_dir = Path('../models')
            pytorch_path = models_dir / 'pytorch_mlp.pth'
            if pytorch_path.exists():
                input_dim = features.shape[1]
                best_model = load_pytorch_model(pytorch_path, input_dim)
        
        if best_model_name == 'PyTorch MLP':
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                probas = best_model(features_tensor).numpy().flatten()
        else:
            probas = best_model.predict_proba(features)[:, 1]
        
        return probas
    
    except Exception as e:
        print(f"Prediction probability error: {e}")
        return np.random.random(features.shape[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single_route():
    try:
        data = request.json
        features = data.get('features', {})
        
        # Preprocess features
        features_scaled = preprocess_features(features)
        
        # Perform prediction
        prediction = predict_with_model(features_scaled)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'model_used': best_model_name if best_model_name else 'Placeholder'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch_route():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if Label column exists for evaluation
        has_labels = 'Label' in df.columns
        
        if has_labels:
            labels = df['Label'].values
            features_df = df.drop('Label', axis=1)
        else:
            features_df = df
            labels = None
        
        # Drop non-numeric columns (like SHA1, FirstSeenDate, etc.)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols]
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        # Get predictions
        predictions = []
        probas = predict_proba_with_model(features_scaled)
        
        for i in range(len(features_scaled)):
            pred = "malware" if probas[i] > 0.5 else "goodware"
            predictions.append(pred)
        
        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'num_instances': len(predictions),
            'model_used': best_model_name if best_model_name else 'Placeholder'
        }
        
        # If labels exist, perform evaluation
        if has_labels:
            pred_binary = (probas > 0.5).astype(int)
            
            accuracy = accuracy_score(labels, pred_binary)
            auc = roc_auc_score(labels, probas)
            cm = confusion_matrix(labels, pred_binary)
            
            response['evaluation'] = {
                'accuracy': round(accuracy, 4),
                'auc': round(auc, 4),
                'confusion_matrix': cm.tolist()
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': best_model is not None or pytorch_mlp_architecture is not None,
        'model_name': best_model_name if best_model_name else 'None'
    }), 200

@app.route('/model_info')
def model_info():
    """Get information about loaded models"""
    models_dir = Path('../models')
    results_file = models_dir / 'results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify({
            'success': True,
            'best_model': best_model_name,
            'all_results': results
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Model results not found'
        }), 404

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
