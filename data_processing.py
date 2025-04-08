# data_processing.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultiLabelDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 2.0  # Normalize to 0–1
        label = self.labels[idx]

        # Convert (H, W) → (1, H, W) → (3, H, W)
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)  # Make 3 channels

        # Convert to tensor
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def get_data_loaders(data_path, batch_size=32, val_split=0.2):
    npz_data = np.load(data_path)

    # Get keys and values from npz
    if isinstance(npz_data, np.lib.npyio.NpzFile):
        keys = list(npz_data.keys())
        images = npz_data[keys[0]]
        labels = npz_data[keys[1]]
    else:
        raise ValueError("Loaded data is not a valid .npz file")

    data = {
        'images': images,
        'labels': labels
    }

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Transforms (ViT expects 224x224 and normalized RGB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5,), (0.5,))  # Already 3-channel by this point
    ])

    train_dataset = MultiLabelDataset({
        'images': images[train_indices],
        'labels': labels[train_indices]
    }, transform=transform)

    val_dataset = MultiLabelDataset({
        'images': images[val_indices],
        'labels': labels[val_indices]
    }, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
