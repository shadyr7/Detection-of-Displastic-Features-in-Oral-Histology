import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import OralHistologyCNN  # Assuming the model is in model.py

# Set Title
st.title("🦷 Oral Histology Dysplasia Detection")
st.sidebar.header("📤 Upload Image")

# Load trained model
@st.cache_resource
def load_model():
    checkpoint = torch.load("oral_histology_cnn.pth", map_location=torch.device("cpu"))
    model = OralHistologyCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model, checkpoint

model, checkpoint = load_model()

# Define Grad-CAM function
def get_gradcam(image, model, target_layer):
    """Compute Grad-CAM heatmap for a given image and target model layer."""
    activations = []
    gradients = []
    
    # Hook for capturing activations and gradients
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Register hooks
    layer = dict(model.named_modules()).get(target_layer)
    if layer is None:
        raise ValueError(f"Layer '{target_layer}' not found in the model.")

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    image = image.unsqueeze(0)
    output = model(image)
    class_idx = output.argmax().item()
    score = output[0, class_idx]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Get activations and gradients
    activations = activations[0].detach()
    gradients = gradients[0].detach()

    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # Global average pooling
    cam = torch.sum(weights * activations, dim=1).squeeze(0)
    cam = F.relu(cam)  # Apply ReLU to remove negative values

    # Normalize Grad-CAM
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.numpy()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    return cam, class_idx

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Upload and Process Image
uploaded_file = st.sidebar.file_uploader("📂 Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = transform(image)

        # Compute Grad-CAM
        cam, class_idx = get_gradcam(image_tensor, model, "conv3")
        class_names = ["Normal", "Dysplastic"]
        prediction_text = f"Prediction: **{class_names[class_idx]}**"
        st.subheader(prediction_text)

        # Convert Grad-CAM heatmap
        cam_resized = cv2.resize(cam, (128, 128))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        image_np = np.array(image.resize((128, 128)))
        overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        # Display Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        with col2:
            st.image(overlayed_image, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

        # Display Model Performance Metrics
        st.sidebar.subheader("📊 Model Performance Metrics")
        st.sidebar.text(f"✅ Accuracy: {checkpoint.get('accuracy', 'N/A')}")
        st.sidebar.text(f"🎯 Precision: {checkpoint.get('precision', 'N/A')}")
        st.sidebar.text(f"📢 Recall: {checkpoint.get('recall', 'N/A')}")
        st.sidebar.text(f"📏 F1-score: {checkpoint.get('f1', 'N/A')}")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
