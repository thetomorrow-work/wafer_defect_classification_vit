# main.py
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from data_processing import get_data_loaders
from model import MultiLabelViT
from train import train_model
from validate import validate_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot F1 score
    ax2.plot(epochs, history['val_f1'], 'g-', label='F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Multi-label Image Classification with Vision Transformer')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the numpy data file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device (use MPS for Mac M1-Pro GPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create data loaders
    print("Loading and preparing data...")
    train_loader, val_loader = get_data_loaders(
        args.data_path, 
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = MultiLabelViT(num_classes=8, num_heads=8)
    model = model.to(device)
    
    # Configure training
    training_config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Train model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, device, training_config)
    
    # Plot training history
    plot_training_history(history)
    
    # Validate final model
    print("\nFinal model evaluation:")
    validate_model(model, val_loader, device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, f"{args.checkpoint_dir}/final_model.pth")
    print(f"Final model saved to {args.checkpoint_dir}/final_model.pth")

if __name__ == "__main__":
    main()
