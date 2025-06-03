import cv2
import torch
from src.model import get_model
from src.gradcam import preprocess_image, generate_gradcam, overlay_heatmap

# Load model
model = get_model(num_classes=4)
model.load_state_dict(torch.load("models/resnet18_brain_mri.pt", map_location="cpu"))

# Use an actual test image path here
img_path = "data/test/glioma/Te-gl_0010.jpg"  # ← change to a real file in your test folder

# Preprocess image
img_tensor, orig_img = preprocess_image(img_path)

# Generate Grad-CAM
heatmap, pred_class = generate_gradcam(model, img_tensor, model.layer4)
overlay = overlay_heatmap(heatmap, orig_img)

# Save and show
cv2.imwrite("gradcam_output.jpg", overlay)
print(f"Predicted class: {pred_class}")
print("Grad-CAM image saved as gradcam_output.jpg")
