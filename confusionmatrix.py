import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from model import MultiLabelViT
from data_processing import get_data_loaders

# === Load Model ===
model_path = "checkpoints/final_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultiLabelViT(num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

# === Load Data ===
_, val_loader = get_data_loaders(r"C:\Users\nithi\Downloads\archive (4)\Wafer_Map_Datasets.npz", batch_size=32)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)

# === Generate Multi-label Confusion Matrices ===
conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

# === Print summary table ===
defect_names = ['Center', 'Donut', 'EdgeLoc', 'EdgeRing', 'Loc', 'NearFull', 'Scratch', 'Random']
print(f"{'Defect':<10}  TP   FP   TN   FN")
print("-" * 35)
for i, name in enumerate(defect_names):
    tn, fp, fn, tp = conf_matrices[i].ravel()
    print(f"{name:<10}  {tp:<4} {fp:<4} {tn:<4} {fn:<4}")
