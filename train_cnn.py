import mlflow
import argparse
import torchvision.transforms as transforms
from experiment import run_experiment

# Config
DEFAULT_IMG_SIZE = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
EXPERIMENT_NAME = 'cnn_image_classification'

TRANSFORM_OPTIONS = ['raw', 'transformed', 'augmented']
MODEL_OPTIONS = ['simple_cnn', 'resnet18', 'resnet18_backbone']

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN with configurable model and data transform')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=MODEL_OPTIONS, help='Model architecture to use')
    parser.add_argument('--transform', type=str, default='all', choices=TRANSFORM_OPTIONS + ['all'], help='Data transform to use')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE, help='Image size (height and width)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone when using resnet18_backbone')
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.transform == 'all':
        transform_names = TRANSFORM_OPTIONS
    else:
        transform_names = [args.transform]

    for name in transform_names:
        train_transform = get_transform(name, args.img_size)
        # Always use transformed for validation
        val_transform = get_transform('transformed', args.img_size)
        run_experiment(
            name,
            train_transform,
            val_transform,
            model_name=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            img_size=args.img_size,
            freeze_backbone=args.freeze_backbone
        ) 