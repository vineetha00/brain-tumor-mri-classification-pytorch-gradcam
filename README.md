# 🧠 Brain Tumor Classification from MRI Scans

This project trains a deep learning model to classify MRI brain scans into 4 categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## 📂 Dataset
Kaggle: Brain Tumor Classification (MRI)  
Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Folders used:
- `data/training/`
- `data/testing/`
- `data/val/` (created from training split)

## 🧠 Model
- Backbone: ResNet18
- Trained using PyTorch
- Validation Accuracy: ~100%

## 🔍 Grad-CAM Visualization
Highlight tumor regions using Grad-CAM. Output saved as `gradcam_output.jpg`.

## 🧪 Run
```bash
python main.py           # Train model
python test_gradcam.py   # Run Grad-CAM visualization
