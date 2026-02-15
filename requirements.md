
# AI-Based Cervical Cytology Screening System  

# 1. Project Overview
The system performs **automated classification of microscopic cervical cytology cell images** into **Normal** and **Abnormal** categories using deep learning.

Low-resolution, smartphone-like images were **simulated from the SipakMed dataset** by applying JPEG compression levels **Q=95** and **Q=50**, enabling evaluation in **resource-constrained, real-world screening conditions**.

The system also provides:

- **Explainability using Grad-CAM**
- **Robustness evaluation under compression variability**
- **Confidence-based screening support for clinicians**

This solution is intended for **screening and triage**, not standalone diagnosis.

# 2. Functional Requirements

## 2.1 Image Input
The system shall:

- Accept **JPEG image format**
- Support compression levels:
  - **Q = 95**
  - **Q = 50**


## 2.2 Image Preprocessing
The preprocessing pipeline shall:

- Resize images to **224 × 224 pixels**
- Normalize pixel intensity values
- Convert images to **tensor representation**
- Transfer tensors to **CPU/GPU device**


## 2.3 Classification Module
The classification component shall:

- Use a **Convolutional Neural Network (CNN)**  
  - Architecture: **ResNet18 with CBAM attention**
- Perform **binary classification**:
  - Normal  
  - Abnormal
- Produce:
  - **Predicted label**
  - **Confidence score**


## 2.4 Explainability Module
The explainability component shall:

- Implement **Grad-CAM visualization**
- Generate **heatmap overlays**
- Highlight **regions influencing model prediction**
- Support **clinician interpretability**


## 2.5 Batch Processing
Training and inference shall support:

- **Batch size:** 32  
- **Training epochs:** 10  


## 2.6 Evaluation
The system shall compute:

- **Classification accuracy**
- **Misclassification tracking**
- **Prediction confidence reporting**

# 3. Non-Functional Requirements

## 3.1 Performance
The system shall provide:

- **Near real-time inference capability**
- Efficient **batch processing performance**


## 3.2 Reliability
The system shall ensure:

- **Consistent predictions at JPEG Q=95**
- **Robust performance at JPEG Q=50**
- Stable inference across **compression variability**


## 3.3 Usability
The system shall provide:

- **Clear and interpretable output**
- **Visual Grad-CAM explanations**
- **Human-in-the-loop validation workflow**


# 4. Hardware Requirements

Minimum environment:

- **Standard CPU-compatible system**
- **≥ 8 GB RAM**

Recommended:

- **GPU acceleration** for faster training and inference


# 5. Software Requirements

Core environment:

- **Python 3.10.19**

Deep learning & image processing:

- **PyTorch 2.5.1** — model training and inference  
- **Torchvision 0.20.1** — preprocessing, datasets, pretrained models  
- **OpenCV 4.7.0** — image processing and heatmap overlay  
- **NumPy 1.26.4** — numerical computation and normalization  
- **Matplotlib 3.10.8** — visualization and Grad-CAM display  

Model components:

- **ResNet18 (Torchvision pretrained CNN)**  
- **CBAM (custom attention module integration)**  
- **Grad-CAM (custom implementation using OpenCV)**  

# 6. Assumptions

- All images are **correctly labeled**.
- The task is strictly **binary classification**.
- Dataset structure follows **ImageFolder format**.
- Input images represent **single-cell cytology**.

# 7. Limitations

- Supports **binary classification only**.
- Performance depends on **training data quality and diversity**.
- **Compression artifacts** may reduce accuracy at lower JPEG quality.
- Not intended as a **standalone diagnostic system**.
