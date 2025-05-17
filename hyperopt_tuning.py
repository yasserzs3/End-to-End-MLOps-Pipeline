import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import torch
from experiment import run_experiment
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import ImageCSVDataset
from torch.utils.data import DataLoader
from models import get_model
from training_utils import train_one_epoch, validate

# Define the search space for hyperparameters
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
    'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
    'epochs': scope.int(hp.quniform('epochs', 5, 20, 5)),
    'img_size': scope.int(hp.quniform('img_size', 64, 256, 32)),
    'model': hp.choice('model', ['simple_cnn', 'resnet18_backbone']),
    'transform': hp.choice('transform', ['transformed', 'augmented']),
    'freeze_backbone': True
}

def get_transform(transform_type, img_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transform_type == 'raw':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    elif transform_type == 'transformed':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    elif transform_type == 'augmented':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

def objective(params):
    """
    Objective function for hyperopt optimization.
    Returns the negative validation accuracy (since hyperopt minimizes).
    """
    # Set up MLflow run
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Prepare data with simple train/validation split
        df = pd.read_csv('data/train.csv')
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        
        # Create datasets and data loaders
        train_dataset = ImageCSVDataset(train_df, 'data/images', transform=get_transform(params['transform'], params['img_size']))
        val_dataset = ImageCSVDataset(val_df, 'data/images', transform=get_transform('transformed', params['img_size']))
        
        train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=int(params['batch_size']), shuffle=False, num_workers=2)
        
        # Set up model, criterion, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(params['model'], num_classes=2, img_size=int(params['img_size']), 
                        freeze_backbone=params['freeze_backbone']).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop with early stopping
        best_val_acc = 0
        best_model_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(int(params['epochs'])):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_acc', train_acc, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_acc', val_acc, step=epoch)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                # Save the best model to MLflow
                mlflow.pytorch.log_model(model, 'model')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Return negative best validation accuracy (since hyperopt minimizes)
        return {'loss': -best_val_acc, 'status': STATUS_OK}

def run_hyperopt_tuning(max_evals=50):
    """
    Run hyperparameter optimization using Hyperopt.
    """
    # Set up MLflow experiment
    mlflow.set_experiment('hyperopt_tuning')
    
    # Initialize trials object
    trials = Trials()
    
    # Run optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    print("Best hyperparameters:", best)
    return best

if __name__ == '__main__':
    best_params = run_hyperopt_tuning(max_evals=10) 