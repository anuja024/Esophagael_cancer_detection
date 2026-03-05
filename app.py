import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# -----------------------
# PAGE CONFIG + PREMIUM UI
# -----------------------
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 0rem;
        }

        .upload-card {
            background: #1e1e2f;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #333;
        }

        .center {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# CLASSES
# -----------------------
class_names = [
    'dyed-lifted-polyps','dyed-resection-margins','esophagitis',
    'normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis'
]

cancer_classes = ['polyps','ulcerative-colitis','dyed-lifted-polyps']

# -----------------------
# METRICS
# -----------------------
model_metrics = {
    "ResNet18": {"Accuracy": 0.93, "F1": 0.92, "AUC": 0.95},
    "MobileNetV2": {"Accuracy": 0.90, "F1": 0.90, "AUC": 0.99}
}

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8)
        model.load_state_dict(torch.load("resnet18_kvasir.pth", map_location="cpu"))
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 8)
        model.load_state_dict(torch.load("mobilenet_kvasir.pth", map_location="cpu"))

    model = model.to(device)
    model.eval()
    return model, device

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------
# GRAD-CAM
# -----------------------
def generate_gradcam(model, img_tensor, target_class, device):
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    if isinstance(model, models.ResNet):
        target_layer = model.layer4
    else:
        target_layer = model.features[-1]

    fw = target_layer.register_forward_hook(forward_hook)
    bw = target_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    output = model(img_tensor)

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot)

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i,w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam,0)
    cam = cv2.resize(cam,(224,224))
    cam = (cam - cam.min())/(cam.max()+1e-8)

    fw.remove()
    bw.remove()
    return cam

# -----------------------
# PDF REPORT
# -----------------------
def create_pdf(pred_class, confidence, result, model_name, 
               input_image_path, gradcam_path, metrics):

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)

    styles = getSampleStyleSheet()

    # -----------------------
    # Styles
    # -----------------------
    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        alignment=TA_CENTER,
        spaceAfter=15
    )

    section_style = ParagraphStyle(
        'section',
        parent=styles['Heading2'],
        spaceAfter=10
    )

    normal_style = styles['Normal']

    content = []

    # -----------------------
    # TITLE
    # -----------------------
    content.append(Paragraph("CancerNet: Esophageal Cancer Detection Report", title_style))

    # -----------------------
    # MODEL INFO
    # -----------------------
    content.append(Paragraph("Model Information", section_style))
    content.append(Paragraph(f"<b>Model Used:</b> {model_name}", normal_style))
    content.append(Spacer(1, 10))

    # -----------------------
    # PREDICTION
    # -----------------------
    content.append(Paragraph("Prediction Result", section_style))
    content.append(Paragraph(f"<b>Detected Class:</b> {pred_class}", normal_style))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.2f}%", normal_style))
    content.append(Paragraph(f"<b>Diagnosis:</b> {result}", normal_style))
    content.append(Spacer(1, 10))
    
    # -----------------------
    # METRICS TABLE
    # -----------------------
    content.append(Paragraph("Model Performance", section_style))

    data = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics['Accuracy']*100:.2f}%"],
        ["F1 Score", f"{metrics['F1']:.2f}"],
        ["AUC", f"{metrics['AUC']:.2f}"]
    ]

    table = Table(data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ]))

    content.append(table)
    content.append(Spacer(1, 15))

    # -----------------------
    # IMAGES
    # -----------------------
    content.append(Paragraph("Visual Analysis", section_style))

    try:
        input_img = Image(input_image_path, width=2.5*inch, height=2.5*inch)
        gradcam_img = Image(gradcam_path, width=2.5*inch, height=2.5*inch)

        img_table = Table([[input_img, gradcam_img]])
        content.append(img_table)

    except:
        content.append(Paragraph("Images could not be loaded.", normal_style))

    content.append(Spacer(1, 15))

    # -----------------------
    # DISCLAIMER
    # -----------------------
    content.append(Paragraph("Disclaimer", section_style))
    content.append(Paragraph(
        "This report is generated using an AI-based deep learning model. "
        "It is intended for research and educational purposes only and should not "
        "be considered as a substitute for professional medical diagnosis.",
        normal_style
    ))

        # -----------------------
    # Build PDF
    # -----------------------
    doc.build(content)

    return temp_file.name

# -----------------------
# TITLE
# -----------------------
st.markdown("<h1 class='center'>🧠 CancerNet</h1>", unsafe_allow_html=True)

# -----------------------
# TOP BAR
# -----------------------
# ---------- MODEL + TOGGLE (PREMIUM INLINE UI) ----------
st.markdown("""
<style>
.container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
}
.toggle-right {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1,3])

with col1:
    selected_model = st.selectbox("Model", ["ResNet18", "MobileNetV2"])

with col2:
    st.markdown('<div class="toggle-right">', unsafe_allow_html=True)
    comparison_mode = st.toggle("Compare")
    st.markdown('</div>', unsafe_allow_html=True)

model, device = load_model(selected_model)

# -----------------------
# CENTERED PREMIUM UPLOAD CARD
# -----------------------
_, col_mid, _ = st.columns([1,2,1])

with col_mid:
    #st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# MAIN
# -----------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(output,dim=1)
        pred = torch.argmax(probs,dim=1)

    pred_class = class_names[pred.item()]
    confidence = probs[0][pred.item()].item()

    if pred_class in cancer_classes:
        result = "🔴 Cancerous"
        color = "#ff4b4b"
    else:
        result = "🟢 Non-Cancerous"
        color = "#00c853"

    cam = generate_gradcam(model,img_tensor,pred.item(),device)

    img_np = np.array(image.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    overlay = heatmap*0.4 + img_np
    overlay = overlay/np.max(overlay)

    _,col1,col2,col3,_ = st.columns([0.3,1,1,1,0.3])

    with col1:
        st.image(image,width=240)

    with col2:
        st.image(overlay,width=190)

    with col3:
        st.write(f"**{pred_class}**")
        st.progress(float(confidence))
        st.caption(f"{confidence*100:.1f}%")

        st.markdown(f"<h4 style='color:{color}; text-align:center;'>{result}</h4>",unsafe_allow_html=True)

        # -----------------------
        # PDF DOWNLOAD 
        # -----------------------
        try:
            metrics = model_metrics[selected_model]

            input_path = "temp_input.jpg"
            gradcam_path = "temp_gradcam.jpg"

            image.save(input_path)
            cv2.imwrite(gradcam_path, np.uint8(overlay * 255))

            pdf_path = create_pdf(
            pred_class,
            confidence,
            result,
            selected_model,
            input_path,
            gradcam_path,
            metrics
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
            "📄 Download Report",
            f,
            file_name="Cancer_Report.pdf",
            use_container_width=True
            )

        except Exception as e:
            st.error(f"PDF Error: {e}")

# -----------------------
# COMPARISON
# -----------------------
if comparison_mode:
    st.markdown(
    "<h3 style='text-align: center;'>📊 Model Performance Comparison</h3>",
    unsafe_allow_html=True
)
    # Create center alignment
    left_space, center, right_space = st.columns([1, 2, 1])

    with center:
        col1, col2 = st.columns(2)

    

    # -----------------------
    # ResNet18
    # -----------------------
    
    with col1:
        st.markdown(
            "<h5 style='text-align: center;'> 🔵 ResNet18</h5>",
            unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex; justify-content:center;">
    <div style="
        background-color:#1e1e2f;
        padding:12px;
        border-radius:10px;
        width:220px;
        font-size:14px;
        text-align: center;
    ">
        <b>Accuracy:</b> 93.00%<br>
        <b>F1 Score:</b> 0.92<br>
        <b>AUC:</b> 0.95
    </div>
    """, unsafe_allow_html=True)

    # -----------------------
    # MobileNetV2
    # -----------------------
    with col2:
        st.markdown(
            "<h5 style='text-align: center;'> 🟢 MobileNetV2</h5>",
            unsafe_allow_html=True)
        
        st.markdown("""
            <div style="display:flex; justify-content:center;">
            <div style="
            background-color:#1e1e2f;
            padding:12px;
            border-radius:10px;
            width:220px;
            font-size:14px;
            text-align: center;
            ">
            <b>Accuracy:</b> 90.00%<br>
            <b>F1 Score:</b> 0.90<br>
            <b>AUC:</b> 0.99
            </div>
            """, unsafe_allow_html=True)
        
# -----------------------
# FOOTER
# -----------------------
st.markdown(
    "<p style='text-align:center; font-size:11px;'>⚠️ AI-based prediction, not medical diagnosis</p>",
    unsafe_allow_html=True
)