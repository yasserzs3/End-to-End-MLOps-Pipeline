import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import io

from datasets import ImageCSVDataset
from models import get_model
from train_utils import train_one_epoch, validate

# Configurable constants can be passed as arguments if needed

def run_experiment(transform_name, train_transform, val_transform, model_name='simple_cnn',
                   data_dir='data/images', csv_path='data/train.csv', img_size=128, batch_size=32, epochs=10, lr=1e-3, num_classes=2, freeze_backbone=False):
    # Data
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_dataset = ImageCSVDataset(train_df, data_dir, transform=train_transform)
    val_dataset = ImageCSVDataset(val_df, data_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_classes=num_classes, img_size=img_size, freeze_backbone=freeze_backbone).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_model_state = None

    with mlflow.start_run(run_name=f"{model_name}_{transform_name}"):
        mlflow.log_param('transform', transform_name)
        mlflow.log_param('img_size', img_size)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('model', model_name)
        mlflow.log_param('freeze_backbone', freeze_backbone)
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # ROC-AUC calculation
            model.eval()
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.cpu().numpy())
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            try:
                if num_classes == 2:
                    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                else:
                    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
                mlflow.log_metric('val_roc_auc', roc_auc, step=epoch)
            except ValueError:
                # This can happen if only one class is present in y_true
                roc_auc = float('nan')

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_acc', train_acc, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_acc', val_acc, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Val ROC-AUC: {roc_auc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                mlflow.pytorch.log_model(model, 'model')
        # Save the best model weights locally
        if best_model_state is not None:
            torch.save(best_model_state, 'best_model.pth')

        # Calculate and log additional metrics at the end of training
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate precision, recall, F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        mlflow.log_metric('val_precision', precision)
        mlflow.log_metric('val_recall', recall)
        mlflow.log_metric('val_f1', f1)

        # Calculate and log confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        mlflow.log_image(buf, 'confusion_matrix.png')
        plt.close()

        # Other metrics you can track (examples):
        # - Precision, Recall, F1-score (per epoch or at the end)
        # - Confusion matrix (as an artifact)
        # - ROC-AUC (for binary/multiclass)
        # - Learning rate (if using a scheduler)
        # - Per-class accuracy
        # - Inference time, model size, etc. 