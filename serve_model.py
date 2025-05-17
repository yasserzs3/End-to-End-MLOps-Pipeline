import mlflow
import mlflow.pytorch
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torchvision.transforms as transforms
from pydantic import BaseModel
import uvicorn
import os
from monitoring import ModelMonitor
from model_registry import ModelRegistry
import json
import sqlite3
import pandas as pd

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using the trained model",
    version="1.0.0"
)

# Global variables
model = None
transform = None
device = None
monitor = ModelMonitor()
registry = ModelRegistry()
model_config = None

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    class_name: str
    probabilities: dict

def load_model():
    """Load the best model from MLflow Model Registry."""
    global model, transform, device, model_config
    
    try:
        # Load model from registry
        model = registry.load_model(stage="Production")
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Get model details and configuration
        model_details = registry.get_model_details()
        print(f"Loaded model version {model_details['version']} from {model_details['stage']} stage")
        
        # Load model configuration from MLflow
        model_config = {
            'img_size': model_details.get('img_size', 224),  # Default to 224 if not found
            'model_name': model_details.get('model', 'simple_cnn'),
            'class_names': {0: "class_0", 1: "class_1"}  # Update with your actual class names
        }
        
        # Create transform based on model configuration
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((model_config['img_size'], model_config['img_size'])),
            transforms.ToTensor(),
            normalize,
        ])
        
        print(f"Model configuration loaded: {json.dumps(model_config, indent=2)}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    load_model()

def preprocess_image(image_bytes):
    """Preprocess the image for model input."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transform
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(device)
    
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error preprocessing image: {str(e)}. Please ensure the file is a valid image."
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction on an uploaded image.
    
    Args:
        file: The image file to classify (supported formats: jpg, jpeg, png)
        
    Returns:
        PredictionResponse: The prediction results including:
            - prediction: The predicted class index
            - confidence: The confidence score for the prediction
            - class_name: The name of the predicted class
            - probabilities: Dictionary of class probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image file."
            )
        
        # Read the image file
        contents = await file.read()
        
        # Validate file size (e.g., max 10MB)
        if len(contents) > 10 * 1024 * 1024:  # 10MB in bytes
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Preprocess the image
        image_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            # Get probabilities for all classes
            class_probs = {
                model_config['class_names'][i]: float(probabilities[0][i].item())
                for i in range(len(model_config['class_names']))
            }
        
        # Log prediction for monitoring
        monitor.log_prediction(
            prediction=prediction,
            confidence=confidence,
            features={
                "filename": file.filename,
                "file_size": len(contents),
                "content_type": file.content_type,
                "probabilities": json.dumps(class_probs),
                "model_version": registry.get_model_details()["version"]
            }
        )
        
        # Check for drift
        drift_indicators = monitor.detect_drift()
        if drift_indicators['drift_detected']:
            monitor.generate_alert(
                alert_type="DRIFT_DETECTED",
                message=f"Model drift detected: {', '.join(drift_indicators['reasons'])}",
                severity="WARNING",
                additional_data=drift_indicators['metrics'] if 'metrics' in drift_indicators else None
            )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            class_name=model_config['class_names'][prediction],
            probabilities=class_probs
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_config": model_config
    }

@app.get("/metrics")
async def get_metrics():
    """Get current model performance metrics."""
    try:
        metrics = monitor.calculate_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating metrics: {str(e)}"
        )

@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """Get recent alerts."""
    try:
        alerts = monitor.get_recent_alerts(limit=limit)
        return alerts.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving alerts: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the currently loaded model."""
    try:
        model_details = registry.get_model_details()
        return {
            **model_details,
            "model_config": model_config
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )

@app.get("/metrics/history")
async def get_metrics_history(metric_name: str, limit: int = 10):
    """Get historical values for a specific metric."""
    try:
        conn = sqlite3.connect("model_monitoring.db")
        query = f'''
        SELECT timestamp, metric_value
        FROM metrics
        WHERE metric_name = ?
        ORDER BY timestamp DESC
        LIMIT {limit}
        '''
        
        df = pd.read_sql_query(query, conn, params=(metric_name,))
        conn.close()
        
        if df.empty:
            return []
            
        # Convert timestamp to string for JSON serialization
        df['timestamp'] = df['timestamp'].astype(str)
        return df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics history: {str(e)}"
        )

@app.get("/metrics/names")
async def get_metric_names():
    """Get the names of all available metrics."""
    try:
        conn = sqlite3.connect("model_monitoring.db")
        query = '''
        SELECT DISTINCT metric_name
        FROM metrics
        ORDER BY metric_name
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {"metric_names": df['metric_name'].tolist()}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metric names: {str(e)}"
        )

@app.get("/predictions/summary")
async def get_predictions_summary(limit: int = 100):
    """Get a summary of recent predictions."""
    try:
        conn = sqlite3.connect("model_monitoring.db")
        query = f'''
        SELECT timestamp, prediction, confidence
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {"count": 0, "predictions": []}
            
        # Convert timestamp to string for JSON serialization
        df['timestamp'] = df['timestamp'].astype(str)
        
        return {
            "count": len(df),
            "avg_confidence": float(df['confidence'].mean()),
            "predictions": df.to_dict(orient='records')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving predictions summary: {str(e)}"
        )

@app.post("/feedback")
async def record_feedback(prediction_id: int, correct_label: int, notes: str = None):
    """Record feedback for a prediction (update ground truth)."""
    try:
        conn = sqlite3.connect("model_monitoring.db")
        cursor = conn.cursor()
        
        # Update the prediction with ground truth
        cursor.execute('''
        UPDATE predictions
        SET ground_truth = ?
        WHERE id = ?
        ''', (correct_label, prediction_id))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with ID {prediction_id} not found"
            )
            
        conn.commit()
        conn.close()
        
        # Generate an alert if this is feedback
        monitor.generate_alert(
            alert_type="FEEDBACK_RECEIVED",
            message=f"Ground truth updated for prediction {prediction_id}: {correct_label}",
            severity="INFO",
            additional_data={"notes": notes} if notes else None
        )
        
        return {"status": "success", "message": f"Ground truth updated for prediction {prediction_id}"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recording feedback: {str(e)}"
        )

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000) 