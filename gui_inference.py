import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model setup
# -----------------------
num_classes = 8
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("resnet18_kvasir.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]

cancerous_classes = {"dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", "polyps", "ulcerative-colitis"}
non_cancerous_classes = {"normal-cecum", "normal-pylorus", "normal-z-line"}

# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------
# Grad-CAM
# -----------------------
def generate_gradcam(img_path, pred_class_idx, layer_name="layer4"):
    model.eval()
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = dict([*model.named_modules()])[layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    output = model(img_t)
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class_idx] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1,2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    img = np.array(img.resize((224,224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(overlay*255)

    gradcam_path = "gradcam_temp.jpg"
    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return gradcam_path

# -----------------------
# Prediction
# -----------------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        pred_class = class_names[pred.item()]
        if pred_class in cancerous_classes:
            return pred.item(), "Cancerous"
        else:
            return pred.item(), "Non-cancerous"

# -----------------------
# GUI Functions
# -----------------------
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        pred_idx, result = predict_image(file_path)

        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel_input.configure(image=img_tk)
        panel_input.image = img_tk

        gradcam_path = generate_gradcam(file_path, pred_idx)
        grad_img = Image.open(gradcam_path).resize((224, 224))
        grad_tk = ImageTk.PhotoImage(grad_img)
        panel_grad.configure(image=grad_tk)
        panel_grad.image = grad_tk

        if result == "Cancerous":
            result_label.configure(text=f"⚠️ Prediction: {result}", text_color="red")
        else:
            result_label.configure(text=f"✅ Prediction: {result}", text_color="green")

# -----------------------
# Modern GUI (CustomTkinter)
# -----------------------
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("CancerNet(Esophageal Cancer Detection with Grad-CAM)")
root.geometry("800x600")

btn = ctk.CTkButton(root, text="📂 Select Image", command=open_file, width=200, height=40, font=("Arial", 14))
btn.pack(pady=20)

frame = ctk.CTkFrame(root)
frame.pack(pady=10)

panel_input = ctk.CTkLabel(frame, text="Input Image", width=224, height=224)
panel_input.grid(row=0, column=0, padx=20)

panel_grad = ctk.CTkLabel(frame, text="Grad-CAM Heatmap", width=224, height=224)
panel_grad.grid(row=0, column=1, padx=20)

result_label = ctk.CTkLabel(root, text="", font=("Arial", 20, "bold"))
result_label.pack(pady=20)

exit_btn = ctk.CTkButton(root, text="❌ Exit", command=root.destroy, width=200, height=40, fg_color="red")
exit_btn.pack(pady=10)

root.mainloop()
