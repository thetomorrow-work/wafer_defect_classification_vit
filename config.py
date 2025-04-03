import torch

# Device configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Training parameters
CONFIG = {
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'patience': 5,
    'num_classes': 8,
    'image_size': 224,
    'model_save_path': 'model/best_model_m1pro.pth',
    'metrics_save_path': 'metrics/training_metrics.png'
}
