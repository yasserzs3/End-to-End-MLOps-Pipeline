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

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using the trained model",
    version="1.0.0"
)

# Global variables
model = None
transform = None
device = None

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    class_name: str

def load_model():
    """Load the best model from MLflow."""
    global model, transform, device
    
    try:
        # Get the best run from the hyperopt_tuning experiment
        experiment = mlflow.get_experiment_by_name("hyperopt_tuning")
        if experiment is None:
            raise ValueError("No hyperopt_tuning experiment found")
        
        # Get the best run based on mean_cv_accuracy
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        best_run = runs.loc[runs['metrics.mean_cv_accuracy'].idxmax()]
        
        # Load the model
        model_uri = f"runs:/{best_run.run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        
        # Set up device and transform
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Get the image size from the run parameters
        img_size = int(best_run['params.img_size'])
        
        # Create transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
        print(f"Model loaded successfully from run {best_run.run_id}")
        print(f"Model parameters: {best_run['params']}")
        
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
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction on an uploaded image.
    
    Args:
        file: The image file to classify
        
    Returns:
        PredictionResponse: The prediction results
    """
    try:
        # Read the image file
        contents = await file.read()
        
        # Preprocess the image
        image_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Map prediction to class name (you might want to load this from a config file)
        class_names = {0: "class_0", 1: "class_1"}  # Update with your actual class names
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            class_name=class_names[prediction]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000) 