# data_processing.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

class MultiLabelDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: numpy array of shape (no_images, lis) where lis is (img, labels)
            transform: Optional transform to be applied on images
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, labels = self.data[idx]
        
        # Convert numpy array to torch tensor
        img = torch.from_numpy(img).float()
        # Convert from (H,W,C) to (C,H,W) format
        img = img.permute(2, 0, 1)
        
        # Convert labels to tensor
        labels = torch.from_numpy(labels).float()
        
        if self.transform:
            img = self.transform(img)
            
        return img, labels

def get_data_loaders(data_path, batch_size=32, train_ratio=0.8):
    """Create train and validation data loaders"""
    # Load the numpy array data
    data = np.load(data_path, allow_pickle=True)
    
    # Split into train and validation sets
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    split_idx = int(train_ratio * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = MultiLabelDataset(data[train_indices], transform=train_transform)
    val_dataset = MultiLabelDataset(data[val_indices], transform=val_transform)
    
    # Create data loaders optimized for Apple Silicon
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Optimized for Mac
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
