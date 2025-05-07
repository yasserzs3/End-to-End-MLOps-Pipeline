import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import torch
from experiment import run_experiment
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import pandas as pd
from datasets import ImageCSVDataset
from torch.utils.data import DataLoader, SubsetRandomSampler

# Define the search space for hyperparameters
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
    'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
    'epochs': scope.int(hp.quniform('epochs', 5, 30, 5)),
    'img_size': scope.int(hp.quniform('img_size', 64, 256, 32)),
    'model': hp.choice('model', ['simple_cnn', 'resnet18', 'resnet18_backbone']),
    'transform': hp.choice('transform', ['raw', 'transformed', 'augmented']),
    'freeze_backbone': hp.choice('freeze_backbone', [True, False])
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

        # Prepare data for k-fold cross-validation
        df = pd.read_csv('data/train.csv')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize lists to store metrics for each fold
        fold_accuracies = []
        fold_losses = []
        
        # Perform k-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_dataset = ImageCSVDataset(df, 'data/images', transform=get_transform(params['transform'], params['img_size']))
            val_dataset = ImageCSVDataset(df, 'data/images', transform=get_transform('transformed', params['img_size']))
            
            train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']), 
                                    sampler=train_sampler, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=int(params['batch_size']), 
                                  sampler=val_sampler, num_workers=2)
            
            # Run experiment for this fold
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_model(params['model'], num_classes=2, img_size=int(params['img_size']), 
                            freeze_backbone=params['freeze_backbone']).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training loop with early stopping
            best_val_acc = 0
            patience = 5
            patience_counter = 0
            
            for epoch in range(int(params['epochs'])):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                # Log metrics
                mlflow.log_metric(f'fold_{fold}_train_loss', train_loss, step=epoch)
                mlflow.log_metric(f'fold_{fold}_train_acc', train_acc, step=epoch)
                mlflow.log_metric(f'fold_{fold}_val_loss', val_loss, step=epoch)
                mlflow.log_metric(f'fold_{fold}_val_acc', val_acc, step=epoch)
                
                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            fold_accuracies.append(best_val_acc)
            fold_losses.append(val_loss)
        
        # Calculate average metrics across folds
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_loss = np.mean(fold_losses)
        
        # Log aggregate metrics
        mlflow.log_metric('mean_cv_accuracy', mean_accuracy)
        mlflow.log_metric('std_cv_accuracy', std_accuracy)
        mlflow.log_metric('mean_cv_loss', mean_loss)
        
        # Return negative mean accuracy (since hyperopt minimizes)
        return {'loss': -mean_accuracy, 'status': STATUS_OK}

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
    best_params = run_hyperopt_tuning(max_evals=50) 