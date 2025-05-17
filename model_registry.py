from model_registry import ModelRegistry

def register_and_promote_best_model():
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