# MLOps Project: Image Classification with MLflow

This project implements a complete MLOps pipeline for image classification, featuring experiment tracking, model training, hyperparameter tuning, model serving, and monitoring.

## Features

### 1. Experiment Tracking
- MLflow integration for experiment tracking
- Comprehensive metric logging
- Parameter tracking and visualization
- Artifact management

#### Tracked Metrics
- **Training Metrics**:
  - Loss (training and validation)
  - Accuracy (training and validation)
  - Learning rate
  - Batch processing time
  - Epoch duration
- **Model Performance Metrics**:
  - Precision (weighted average)
  - Recall (weighted average)
  - F1-score (weighted average)
  - Confusion matrix
- **Resource Metrics**:
  - GPU memory usage
  - CPU utilization
  - Training time per epoch
  - Total training duration

### 2. Model Training and Tuning
- Automated hyperparameter tuning with Hyperopt
- Simple train/validation split
- Early stopping implementation
- Model checkpointing

#### Hyperparameter Search Space
- **Model Architecture**:
  - Learning rate: [1e-5, 1e-3] (log scale)
  - Batch size: [16, 32, 64, 128]
  - Number of epochs: [10, 50]
  - Image size: [96, 128, 224]
  - Model type: ['simple_cnn', 'resnet18', 'resnet18_backbone']
- **Optimization**:
  - Optimizer: ['adam', 'sgd']
  - Weight decay: [0, 1e-4, 1e-3]
  - Momentum (for SGD): [0.9, 0.99]
- **Regularization**:
  - Dropout rate: [0.1, 0.5]
  - Data augmentation intensity: [0.1, 0.5]
- **Early Stopping**:
  - Patience: [3, 10]
  - Min delta: [1e-4, 1e-3]

### 3. Model Deployment
- FastAPI-based model serving
- RESTful API endpoints
- Input validation and error handling
- Automatic model loading from registry

#### API Endpoints
- **Prediction**: 
  - `/predict`: Upload an image for classification
  - Returns prediction, confidence score, class name, and class probabilities
- **Health & Monitoring**:
  - `/health`: Check service health and model configuration
  - `/metrics`: View current performance metrics
  - `/metrics/names`: Get list of available metrics
  - `/metrics/history`: Get historical values for a specific metric
  - `/predictions/summary`: Get summary of recent predictions
  - `/alerts`: View recent alerts for model drift or issues
  - `/model-info`: Get information about the current model
- **Feedback**:
  - `/feedback`: Submit ground truth for a prediction for monitoring

### 4. Performance Monitoring
- Real-time prediction logging in SQLite database
- Performance metric calculation without requiring ground truth
- Enhanced drift detection between prediction windows
- Comprehensive alert system for model degradation

#### Drift Detection Methodology
- **Performance Drift**:
  - Window size: 100 predictions
  - Thresholds: 
    - Confidence: 0.05 (5% change)
    - Distribution: 0.2 (20% change)
  - Metrics monitored:
    - Confidence drift (mean confidence change)
    - Prediction distribution drift (class distribution change)

#### Monitoring Metrics
- **Prediction-Based Metrics** (no ground truth required):
  - Total predictions
  - Average confidence
  - Min/max confidence
  - Class distribution
  - Prediction timing

- **Performance Metrics** (when ground truth available):
  - Accuracy
  - Precision
  - Recall
  - F1-score

#### Database Structure
- **Predictions Table**: Stores all predictions with metadata
- **Metrics Table**: Stores calculated metrics over time
- **Alerts Table**: Stores detected issues and alerts

### 5. Model Registry
- MLflow Model Registry integration
- Version control for models
- Stage management (Staging, Production, Archived)
- Model comparison capabilities

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Usage

### Training and Tuning
```bash
# Run hyperparameter tuning
python hyperopt_tuning.py

# Register the best model
python model_registry.py
```

### Model Serving
```bash
# Start the FastAPI server
python serve_model.py
```

The API will be available at `http://localhost:8000`

### Making Predictions
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
     
# Response format
{
  "prediction": 0,
  "confidence": 0.96,
  "class_name": "class_0",
  "probabilities": {
    "class_0": 0.96,
    "class_1": 0.04
  }
}
```

### Checking Metrics
```bash
# Get current metrics
curl http://localhost:8000/metrics

# Get metric names
curl http://localhost:8000/metrics/names

# Get historical values for a specific metric
curl http://localhost:8000/metrics/history?metric_name=avg_confidence&limit=10

# Get prediction summary
curl http://localhost:8000/predictions/summary?limit=10

# Get alerts
curl http://localhost:8000/alerts?limit=10
```

### Providing Feedback
```bash
# Submit ground truth for a prediction
curl -X POST "http://localhost:8000/feedback?prediction_id=1&correct_label=0&notes=good+prediction"
```

## Project Structure
```
.
├── hyperopt_tuning.py    # Hyperparameter tuning implementation
├── model_registry.py     # MLflow Model Registry integration
├── monitoring.py         # Performance monitoring system
├── serve_model.py        # FastAPI model serving
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Code Organization
The project follows a modular design pattern:
- `hyperopt_tuning.py`: Implements hyperparameter optimization with train/val split
- `model_registry.py`: Handles model registration, versioning, and stage transitions
- `monitoring.py`: Provides comprehensive model monitoring and drift detection
- `serve_model.py`: Implements FastAPI app with prediction and monitoring endpoints

## License
This project is licensed under the MIT License - see the LICENSE file for details. 