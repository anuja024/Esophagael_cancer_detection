import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# -----------------------
# Device setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load model (8 classes)
# -----------------------
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # 8 classes in Kvasir dataset
model.load_state_dict(torch.load("resnet18_kvasir.pth", map_location=device))
model = model.to(device)
model.eval()

# -----------------------
# Data transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# Validation dataset
# -----------------------
val_dir = "val"  # make sure you have a "val" folder with class subfolders
val_ds = datasets.ImageFolder(val_dir, transform=transform)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# -----------------------
# Evaluation
# -----------------------
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)   # probabilities
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -----------------------
# Metrics
# -----------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=val_ds.classes,
    output_dict=True
)

print("Classification Report Generated")

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ Metrics saved to metrics.json")

print("===== ROC STARTED =====")
# -----------------------
# ROC-AUC (Multi-class)
# -----------------------
from sklearn.preprocessing import label_binarize
import numpy as np

y_true = np.array(all_labels)
y_scores = np.array(all_probs)

# Binarize labels
y_true_bin = label_binarize(y_true, classes=list(range(8)))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Average AUC
avg_auc = np.mean(list(roc_auc.values()))
print("Average AUC:", avg_auc)

# Save ROC curve
plt.figure()
plt.plot(fpr[0], tpr[0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Class 0)")
plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")