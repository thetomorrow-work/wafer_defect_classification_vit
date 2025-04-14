import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from tqdm import tqdm
from model import MultiLabelViT 
from data_processing import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model
model = MultiLabelViT()  
model.load_state_dict(torch.load("checkpoints/best_model.pth")["model_state_dict"])
model.to(device)
model.eval()

data_path= r"C:\Users\nithi\Downloads\archive (4)\Wafer_Map_Datasets.npz"
_, val_loader = get_data_loaders(data_path, batch_size=32, val_split=0.2) 

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Collecting Predictions"):
        images = images.to(device)
        outputs = model(images)
        probs = outputs.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

# Stack everything
y_probs = np.vstack(all_probs)
y_true = np.vstack(all_labels)

# ----- 1. Per-Class F1 Score -----
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)
f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=1)

# Plot F1 per class
plt.figure(figsize=(10, 6))
plt.bar(range(len(f1_per_class)), f1_per_class, color='teal')
plt.xlabel("Class Index")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores")
plt.xticks(range(len(f1_per_class)))
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/f1_per_class.png")
plt.close()

# Save F1 scores to CSV
pd.DataFrame(f1_per_class, columns=["F1 Score"]).to_csv("plots/f1_per_class.csv", index_label="Class")

# ----- 2. Classification Report -----
report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
pd.DataFrame(report).transpose().to_csv("plots/classification_report.csv")

# ----- 3. Threshold Effect Plot -----
thresholds = np.linspace(0, 1, 50)
f1s = []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    f1 = f1_score(y_true, y_pred_t, average='samples', zero_division=1)
    f1s.append(f1)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1s, marker='o', linestyle='-', color='purple')
plt.xlabel("Threshold")
plt.ylabel("F1 Score (samples average)")
plt.title("Effect of Threshold on F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/f1_threshold_curve.png")
plt.close()
