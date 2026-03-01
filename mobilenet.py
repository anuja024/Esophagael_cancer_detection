import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json

# -----------------------
# Grad-CAM helper function
# -----------------------
def save_gradcam(model, img_tensor, target_class, save_path, layer_name="features.18", device="cpu"):
    model.eval()
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = dict([*model.named_modules()])[layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    output = model(img_tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(overlay * 255)

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_ds = datasets.ImageFolder(root="train", transform=train_tf)
    val_ds = datasets.ImageFolder(root="val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

   
    # -----------------------
# Model: MobileNetV2
# -----------------------
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # Training loop
    num_epochs = 1
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    all_preds, all_labels = [], []

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss_hist.append(running_loss / len(train_loader))
        train_acc_hist.append(correct / total)

        # Validation
        model.eval()
        running_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss_hist.append(running_loss / len(val_loader))
        val_acc_hist.append(correct / total)
        
        # After validation
        report = classification_report(all_labels, all_preds, target_names=train_ds.classes, output_dict=True)

        # Save report as JSON
        with open("metrics.json", "w") as f:
            json.dump(report, f, indent=4)

        print("✅ Metrics saved in metrics.json")

        # ✅ Print and stop/save after first epoch
        print(f"Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc_hist[-1]:.3f}, Val Acc: {val_acc_hist[-1]:.3f}")
        
        if epoch == 0:
            # Save model
            model_path = r"mobilenet_kvasir.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Trained model saved as '{model_path}'")

            # Save training history
            import pickle
            with open("train_history.pkl", "wb") as f:
                pickle.dump({
                    'train_loss': train_loss_hist,
                    'val_loss': val_loss_hist,
                    'train_acc': train_acc_hist,
                    'val_acc': val_acc_hist
                }, f)
            print("✅ Training history saved as 'train_history.pkl'")
            print("🛑 Training stopped after 1 epoch.")
            break




    # Plot loss and accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label="Train")
    plt.plot(val_loss_hist, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    # Classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_ds.classes, yticklabels=train_ds.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Save trained model
    model_path = r"mobilenet_kvasir.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Trained model saved as '{model_path}'")

    # Grad-CAM for all validation images
    gradcam_dir = "gradcam_outputs"
    os.makedirs(gradcam_dir, exist_ok=True)

    model.eval()
    count = 0
    for imgs, labels in val_loader:
        for j in range(len(imgs)):
            class_name = val_ds.classes[labels[j]]
            class_folder = os.path.join(gradcam_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            save_path = os.path.join(class_folder, f"gradcam_{count}_{class_name}.jpg")
            save_gradcam(model, imgs[j], labels[j].item(), save_path, device=device)
            count += 1

    print(f"✅ Grad-CAM images saved in '{gradcam_dir}' (organized by class)")
