import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from src.gradcam import generate_gradcam, preprocess_image, overlay_heatmap
from src.model import get_model

# Load class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
model = get_model(num_classes=4)
model.load_state_dict(torch.load("models/resnet18_brain_mri.pt", map_location="cpu"))
model.eval()

# Main prediction function
def predict(image):
    img_tensor, orig_img = preprocess_image(image)
    heatmap, pred_class = generate_gradcam(model, img_tensor, target_layer=model.layer4[1])
    result_img = overlay_heatmap(orig_img, heatmap)
    pred_label = class_names[pred_class]
    return pred_label, result_img

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[gr.Label(label="Predicted Tumor Class"), gr.Image(label="Grad-CAM Heatmap")],
    title="Brain Tumor MRI Classifier",
    description="Upload a brain MRI image. Model will classify the tumor type and highlight important regions using Grad-CAM."
)

if __name__ == "__main__":
    demo.launch()
