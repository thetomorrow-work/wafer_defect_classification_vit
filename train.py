# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np

def train_model(model, train_loader, val_loader, device, config):
    """
    Train the multi-label classification model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (mps, cuda, or cpu)
        config: Dictionary containing training parameters
    """
    # Unpack configuration
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 1e-4)
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for images, labels in val_pbar:
                # Move to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and labels
                pred_labels = (outputs > 0.5).float()
                all_preds.append(pred_labels.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='samples', zero_division=1)

        
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(f1)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Save if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'f1_score': f1
            }, model_path)
            print(f"  New best model saved to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'f1_score': f1
            }, model_path)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    return model, history
