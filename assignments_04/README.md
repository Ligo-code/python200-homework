# Week 4 — CNN Model Comparison & Transfer Learning

This assignment explores pretrained convolutional neural networks (CNNs) using PyTorch, including model comparison, performance benchmarking, feature extraction, and transfer learning.

---

## 📌 Contents

### 🔹 Warm-up
Covers PyTorch fundamentals and basic model inference.

- Tensor operations and reshaping  
- GPU vs CPU usage  
- Pretrained model loading (ResNet18)  
- Image preprocessing and inference  

👉 Notebook:  
https://www.kaggle.com/code/kseniiazakharova/warmup-04


---

### 🔹 Project
Full analysis of CNN models on the Intel Image Classification dataset.

#### ✅ Task Highlights

- Model comparison:
  - ResNet18
  - MobileNetV3-Small
  - EfficientNet-B0  

- Inference and confidence analysis  
- Latency benchmarking (ms/image)  
- Visualization:
  - Confidence distribution (boxplot)
  - Model comparison grid  

- Feature extraction:
  - Removing classification head
  - Generating embeddings
  - PCA visualization  

- Transfer learning:
  - Fine-tuning ResNet18 classification head
  - Training on a small dataset (300 images)
  - Comparing original vs fine-tuned predictions  

👉 Notebook:  
https://www.kaggle.com/code/kseniiazakharova/project-04

---

## 🧠 Key Insights

- Pretrained ImageNet models do not directly match custom dataset labels (label mismatch).
- Different models focus on different visual features, which can lead to inconsistent predictions.
- Confidence does not guarantee correctness — models can be confidently wrong.
- Feature embeddings show that pretrained models already learn meaningful visual representations.
- Fine-tuning even a small part of a network can significantly improve task-specific predictions.

---

## ⚙️ Technologies Used

- Python  
- PyTorch  
- torchvision  
- NumPy  
- Matplotlib  
- scikit-learn (PCA)

---

## 🚀 Summary

This project demonstrates how to:

- Use pretrained CNNs for inference  
- Compare models based on speed and prediction behavior  
- Extract and visualize learned features  
- Apply transfer learning with minimal training data  

It highlights the practical tradeoffs between model size, latency, and prediction quality in real-world ML systems.