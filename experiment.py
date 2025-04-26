import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import mlflow
import mlflow.pytorch

from datasets import ImageCSVDataset
from models import get_model
from train_utils import train_one_epoch, validate

# Configurable constants can be passed as arguments if needed

def run_experiment(transform_name, train_transform, val_transform, model_name='simple_cnn',
                   data_dir='data/images', csv_path='data/train.csv', img_size=128, batch_size=32, epochs=10, lr=1e-3, num_classes=2):
    # Data
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_dataset = ImageCSVDataset(train_df, data_dir, transform=train_transform)
    val_dataset = ImageCSVDataset(val_df, data_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_classes=num_classes, img_size=img_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name=f"{model_name}_{transform_name}"):
        mlflow.log_param('transform', transform_name)
        mlflow.log_param('img_size', img_size)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('model', model_name)
        best_val_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_acc', train_acc, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_acc', val_acc, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                mlflow.pytorch.log_model(model, 'model') 