from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, np.array(img)

def generate_gradcam(model, img_tensor, target_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    gradcam_map = (weights * acts).sum(dim=1).squeeze()
    gradcam_map = torch.relu(gradcam_map)
    gradcam_map = gradcam_map / gradcam_map.max()

    hook_f.remove()
    hook_b.remove()

    return gradcam_map.detach().cpu().numpy(), pred_class

def overlay_heatmap(gradcam_map, original_image):
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    return overlay
