# validate.py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def validate_model(model, data_loader, device):
    """
    Validate the multi-label classification model
    
    Args:
        model: The model to validate
        data_loader: DataLoader for validation data
        device: Device to validate on
    
    Returns:
        metrics (dict with accuracy, f1, precision, recall)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating"):
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert outputs to binary predictions (threshold = 0.5)
            preds = (outputs > 0.5).float()
            
            # Store predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = {
        'sample_accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'micro_f1': f1_score(all_labels, all_preds, average='micro'),
        'samples_f1': f1_score(all_labels, all_preds, average='samples'),
        'precision': precision_score(all_labels, all_preds, average='samples', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='samples', zero_division=0)
    }
    
    # Print metrics
    print("\nValidation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics
