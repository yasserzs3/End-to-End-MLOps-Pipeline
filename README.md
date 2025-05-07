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
- K-fold cross-validation
- Early stopping implementation
- Model checkpointing

#### Hyperparameter Search Space
- **Model Architecture**:
  - Learning rate: [1e-5, 1e-3] (log scale)
  - Batch size: [16, 32, 64, 128]
  - Number of epochs: [10, 50]
  - Image size: [128, 224, 256]
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

#### Cross-Validation Strategy
- 5-fold cross-validation
- Stratified sampling for balanced folds
- Validation metrics averaged across folds
- Best model selection based on mean validation accuracy

### 3. Model Deployment
- FastAPI-based model serving
- RESTful API endpoints
- Input validation and error handling
- Automatic model loading from registry

### 4. Performance Monitoring
- Real-time prediction logging
- Performance metric calculation
- Drift detection
- Alert system for model degradation

#### Drift Detection Methodology
- **Performance Drift**:
  - Window size: 1000 predictions
  - Threshold: 0.05 (5% degradation)
  - Metrics monitored:
    - Accuracy drift
    - Confidence distribution drift
    - Prediction distribution drift

- **Statistical Tests**:
  - Kolmogorov-Smirnov test for distribution changes
  - Chi-square test for categorical drift
  - Z-score for mean shift detection

- **Alert Thresholds**:
  - Warning: > 5% degradation
  - Critical: > 10% degradation
  - Immediate action: > 20% degradation

- **Monitoring Windows**:
  - Short-term: Last 1000 predictions
  - Medium-term: Last 10000 predictions
  - Long-term: All predictions

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

# Train model with specific parameters
python train_cnn.py
```

### Model Serving
```bash
# Start the FastAPI server
python serve_model.py
```

The API will be available at `http://localhost:8000` with the following endpoints:
- `/predict`: Make predictions on images
- `/health`: Check service health
- `/metrics`: View performance metrics
- `/alerts`: View recent alerts
- `/model-info`: Get information about the current model

### Model Registry
```python
from model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register best model
version = registry.register_best_model(
    experiment_name="hyperopt_tuning",
    metric="mean_cv_accuracy"
)

# Transition to production
registry.transition_model_stage(version, "Production")
```

### Monitoring
```python
from monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor()

# View metrics
metrics = monitor.calculate_metrics()

# Check for drift
drift = monitor.detect_drift()

# Plot metrics
monitor.plot_metrics('accuracy')
```

## Project Structure
```
.
├── hyperopt_tuning.py    # Hyperparameter tuning implementation
├── train_cnn.py         # Model training script
├── serve_model.py       # FastAPI model serving
├── model_registry.py    # MLflow Model Registry integration
├── monitoring.py        # Performance monitoring system
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 