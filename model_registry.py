import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import os

class ModelRegistry:
    def __init__(self):
        """Initialize the MLflow client and set up the model registry."""
        self.client = MlflowClient()
        self.model_name = "image_classifier"
        
    def load_model(self, stage="Production"):
        """Load a model from the specified stage."""
        try:
            # Get the latest version in the specified stage
            model_version = self.client.get_latest_versions(
                name=self.model_name,
                stages=[stage]
            )[0]
            
            # Load the model using MLflow
            model = mlflow.pytorch.load_model(model_version.source)
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model from {stage} stage: {str(e)}")
    
    def get_model_details(self):
        """Get details about the currently loaded model."""
        try:
            # Get the latest version in Production
            model_version = self.client.get_latest_versions(
                name=self.model_name,
                stages=["Production"]
            )[0]
            
            # Get run details
            run = self.client.get_run(model_version.run_id)
            
            return {
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "model": run.data.params.get("model", "simple_cnn"),
                "img_size": int(run.data.params.get("img_size", 224)),
                "metrics": run.data.metrics,
                "parameters": run.data.params
            }
            
        except Exception as e:
            raise Exception(f"Error getting model details: {str(e)}")
    
    def register_best_model(self, experiment_name, metric, description=""):
        """Register the best model from an experiment."""
        try:
            # Get the experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise Exception(f"Experiment {experiment_name} not found")
            
            # Get all runs from the experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"]
            )
            
            if runs.empty:
                raise Exception(f"No runs found in experiment {experiment_name}")
            
            # Get the best run
            best_run = runs.iloc[0]
            
            # Register the model
            model_uri = f"runs:/{best_run.run_id}/model"
            model_details = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )
            
            # Update model version description if provided
            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=model_details.version,
                    description=description
                )
            
            return model_details.version
            
        except Exception as e:
            raise Exception(f"Error registering best model: {str(e)}")
    
    def transition_model_stage(self, version, stage, archive_existing_versions=False):
        """Transition a model version to a new stage."""
        try:
            # Archive existing versions in the target stage if requested
            if archive_existing_versions:
                existing_versions = self.client.get_latest_versions(
                    name=self.model_name,
                    stages=[stage]
                )
                for existing_version in existing_versions:
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=existing_version.version,
                        stage="Archived"
                    )
            
            # Transition the new version to the target stage
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            
        except Exception as e:
            raise Exception(f"Error transitioning model stage: {str(e)}")

def register_and_promote_best_model():
    """Register and promote the best model from hyperopt tuning."""
    # Initialize the registry
    registry = ModelRegistry()
    
    # Register the best model from hyperopt tuning
    version = registry.register_best_model(
        experiment_name="hyperopt_tuning",
        metric="val_acc",  # Using validation accuracy as the metric
        description="Best model from hyperopt tuning with simple_cnn"
    )
    
    # Promote the model to Production
    registry.transition_model_stage(
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"Model version {version} has been registered and promoted to Production")

if __name__ == "__main__":
    register_and_promote_best_model() 