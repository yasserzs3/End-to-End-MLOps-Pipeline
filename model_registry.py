import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Dict, Optional, List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_registry.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_name: str = "ImageClassificationModel"):
        """Initialize the model registry.
        
        Args:
            registry_name: Name of the registered model in MLflow
        """
        self.registry_name = registry_name
        self.client = MlflowClient()
        
        # Create the registered model if it doesn't exist
        try:
            self.client.get_registered_model(self.registry_name)
        except Exception:
            self.client.create_registered_model(self.registry_name)
            logger.info(f"Created new registered model: {self.registry_name}")
    
    def register_best_model(self, 
                          experiment_name: str = "hyperopt_tuning",
                          metric: str = "mean_cv_accuracy",
                          description: Optional[str] = None) -> str:
        """Register the best model from an experiment.
        
        Args:
            experiment_name: Name of the MLflow experiment
            metric: Metric to use for selecting the best model
            description: Optional description for the model version
            
        Returns:
            Version number of the registered model
        """
        try:
            # Get the experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} not found")
            
            # Get the best run
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            best_run = runs.loc[runs[f'metrics.{metric}'].idxmax()]
            
            # Get the model URI
            model_uri = f"runs:/{best_run.run_id}/model"
            
            # Register the model
            version = self.client.create_model_version(
                name=self.registry_name,
                source=model_uri,
                run_id=best_run.run_id,
                description=description or f"Best model from {experiment_name} based on {metric}"
            )
            
            # Log model details
            logger.info(f"Registered model version {version.version} from run {best_run.run_id}")
            logger.info(f"Model metrics: {best_run[f'metrics.{metric}']}")
            
            return version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def transition_model_stage(self, 
                             version: str,
                             stage: str,
                             archive_existing_versions: bool = True):
        """Transition a model version to a new stage.
        
        Args:
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=self.registry_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned model version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {str(e)}")
            raise
    
    def get_model_details(self, version: Optional[str] = None) -> Dict:
        """Get details about a specific model version or the latest version.
        
        Args:
            version: Optional version number. If None, returns latest version.
            
        Returns:
            Dictionary containing model details
        """
        try:
            if version:
                model_version = self.client.get_model_version(
                    name=self.registry_name,
                    version=version
                )
            else:
                model_version = self.client.get_latest_versions(
                    name=self.registry_name,
                    stages=["Production"]
                )[0]
            
            return {
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "status": model_version.status,
                "description": model_version.description
            }
            
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            raise
    
    def list_model_versions(self, stages: Optional[List[str]] = None) -> pd.DataFrame:
        """List all versions of the registered model.
        
        Args:
            stages: Optional list of stages to filter by
            
        Returns:
            DataFrame containing model versions
        """
        try:
            versions = self.client.search_model_versions(
                f"name='{self.registry_name}'"
            )
            
            if stages:
                versions = [v for v in versions if v.current_stage in stages]
            
            # Convert to DataFrame
            versions_df = pd.DataFrame([{
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
                "description": v.description
            } for v in versions])
            
            return versions_df
            
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            raise
    
    def load_model(self, stage: str = "Production"):
        """Load a model from the registry.
        
        Args:
            stage: Stage of the model to load
            
        Returns:
            Loaded PyTorch model
        """
        try:
            model_uri = f"models:/{self.registry_name}/{stage}"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions.
        
        Args:
            version1: First version number
            version2: Second version number
            
        Returns:
            Dictionary containing comparison metrics
        """
        try:
            # Get run IDs for both versions
            v1_details = self.client.get_model_version(
                name=self.registry_name,
                version=version1
            )
            v2_details = self.client.get_model_version(
                name=self.registry_name,
                version=version2
            )
            
            # Get metrics for both runs
            v1_metrics = mlflow.get_run(v1_details.run_id).data.metrics
            v2_metrics = mlflow.get_run(v2_details.run_id).data.metrics
            
            # Compare metrics
            comparison = {
                "version1": {
                    "version": version1,
                    "metrics": v1_metrics
                },
                "version2": {
                    "version": version2,
                    "metrics": v2_metrics
                },
                "differences": {
                    metric: v2_metrics.get(metric, 0) - v1_metrics.get(metric, 0)
                    for metric in set(v1_metrics.keys()) | set(v2_metrics.keys())
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise 