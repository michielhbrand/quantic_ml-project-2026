# Malware Detection Web Application

This is a Flask-based web application for malware detection using machine learning.

## Features

1. **Single Instance Prediction**: Enter feature values manually or use pre-filled demo data
2. **Batch Prediction**: Upload a CSV file with multiple instances for batch prediction
3. **Model Evaluation**: Upload a CSV file with labels to evaluate model performance (AUC, Accuracy, Confusion Matrix)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

- `GET /` - Main web interface
- `POST /predict_single` - Single instance prediction
- `POST /predict_batch` - Batch prediction and evaluation
- `GET /health` - Health check endpoint

## File Structure

```
frontend/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
└── static/               # Static files (if needed)
```

## Notes

- The current implementation uses placeholder prediction functions
- Replace the placeholder functions with actual trained model for production use
- Model loading and prediction logic should be implemented in `app.py`
