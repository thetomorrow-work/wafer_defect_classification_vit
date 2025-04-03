# model.py
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class MultiLabelViT(nn.Module):
    def __init__(self, num_classes=8, num_heads=8):
        super(MultiLabelViT, self).__init__()
        
        # Create a ViT configuration with 8 attention heads
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=num_heads,  # Using 8 heads as requested
            intermediate_size=3072,
            hidden_act="gelu",
            image_size=224,
            patch_size=16,
            num_labels=num_classes,
            problem_type="multi_label_classification"
        )
        
        # Initialize the ViT model with our custom configuration
        self.vit = ViTForImageClassification(config)
        
        # Replace the classifier head with a custom one for multi-label
        self.vit.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # Add sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pixel_values):
        # Forward pass through ViT
        outputs = self.vit(pixel_values=pixel_values)
        logits = outputs.logits
        
        # Apply sigmoid activation for multi-label classification
        return self.sigmoid(logits)
