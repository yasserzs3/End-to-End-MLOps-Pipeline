# CNN Image Classification Pipeline with PyTorch & MLflow

This project provides a modular pipeline for image classification using PyTorch, with experiment tracking via MLflow. The codebase is organized for easy extension to new models (e.g., ResNet, ViT) and data augmentations.

## Directory Structure

```
.
├── data/
│   ├── images/                # All image files (JPEG)
│   ├── train.csv              # Training image IDs and labels
│   ├── test.csv               # Test image IDs
│   └── sample_submission.csv  # Submission format example
├── datasets.py                # Custom PyTorch Dataset
├── models.py                  # Model definitions and factory
├── train_utils.py             # Training and validation loops
├── experiment.py              # MLflow experiment runner
├── train_cnn.py               # Main script to launch experiments
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- Pillow
- tqdm
- matplotlib
- mlflow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
- Place your images in `data/images/`.
- Ensure `data/train.csv` contains columns: `img_id`, `label` (without `.jpeg` in `img_id`).

### 2. Run Training Experiments

You can now control the model, data transform, batch size, epochs, learning rate, and image size from the command line:

```bash
python train_cnn.py [--model MODEL] [--transform TRANSFORM] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LEARNING_RATE] [--img-size IMG_SIZE] [--freeze-backbone]
```

#### Model Options
- `simple_cnn`: A small custom convolutional neural network.
- `resnet18`: The full ResNet18 architecture from torchvision, with the final layer replaced for your number of classes. Trains the entire model end-to-end.
- `resnet18_backbone`: Uses ResNet18 as a feature extractor (backbone) and adds a custom head (`Linear -> ReLU -> Linear`).
  - You can freeze the backbone (so only the head is trained) by adding the `--freeze-backbone` flag.

#### CLI Options
- `--model`: Model architecture to use (`simple_cnn`, `resnet18`, `resnet18_backbone`). Default: `simple_cnn`.
- `--transform`: Data transform to use (`raw`, `transformed`, `augmented`, `all`). Default: `all` (runs all three).
- `--batch-size`: Batch size for training. Default: 32
- `--epochs`: Number of training epochs. Default: 10
- `--lr`: Learning rate. Default: 0.001
- `--img-size`: Image size (height and width). Default: 128
- `--freeze-backbone`: (Only for `resnet18_backbone`) Freeze the backbone and train only the custom head.

#### Example Usage

- Train all transforms with the default model:
  ```bash
  python train_cnn.py
  ```
- Train only with augmented data:
  ```bash
  python train_cnn.py --transform augmented
  ```
- Train ResNet18 with normalized data:
  ```bash
  python train_cnn.py --model resnet18 --transform transformed
  ```
- Train ResNet18 as a backbone (custom head, trainable):
  ```bash
  python train_cnn.py --model resnet18_backbone
  ```
- Train ResNet18 as a backbone (custom head, frozen backbone):
  ```bash
  python train_cnn.py --model resnet18_backbone --freeze-backbone
  ```
- Custom batch size and epochs:
  ```bash
  python train_cnn.py --batch-size 64 --epochs 20
  ```
- Custom learning rate and image size:
  ```bash
  python train_cnn.py --lr 0.0005 --img-size 224
  ```
- All options together:
  ```bash
  python train_cnn.py --model resnet18_backbone --transform augmented --batch-size 16 --epochs 30 --lr 0.0001 --img-size 256 --freeze-backbone
  ```

All options and results are logged to MLflow.

### 3. View MLflow Results

Start the MLflow UI:
```bash
mlflow ui
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Adding New Models

1. **Edit `models.py`:**
   - Add your new model class.
   - Update the `get_model` function to return your model when its name is passed.
2. **Edit `train_cnn.py`:**
   - Change the `model_name` argument in the `run_experiment` call to your new model's name.

Example (for ResNet18 as backbone):
```python
run_experiment(name, t, transformed_transform, model_name='resnet18_backbone', freeze_backbone=True)
```

## Customizing Transforms
- Edit the transform pipelines in `train_cnn.py` to add/remove augmentations or normalization as needed.

## License
MIT 