
# Training Guide

This directory contains scripts used to train the **ResNet18 + CBAM + Grad-CAM** model on cervical cytology images.

These scripts are provided for **reproducibility, experimentation, and research purposes**.

# Requirements

## System

- Python 3.10.19

## Python Dependencies

| Library | Version |
|-------|------|
| PyTorch | 2.5.1 |
| Torchvision | 0.20.1 |
| OpenCV | 4.7.0 |
| NumPy | 1.26.4 |
| Matplotlib | 3.10.8 |

### Model Components

- **ResNet18** – Provided by Torchvision
- **CBAM (Convolutional Block Attention Module)** – Custom implementation integrated into the ResNet18 architecture
- **Grad-CAM** – Custom visualization implementation using OpenCV

# Training Steps

## 1. Prepare Dataset

Prepare cervical cytology datasets such as:

- SipakMed dataset
- Other Pap smear or cytology image datasets

Ensure images are organized into training and testing directories.

Example structure:

```
dataset/
│
├── TRAINING
│   ├── Normal
│   └── Abnormal
│
└── TEST
    ├── Normal
    └── Abnormal
```

## 2. Preprocess Images

Run preprocessing scripts to:

- Resize images
- Normalize pixel values
- Prepare tensors for model training

Typical preprocessing includes:

- Resize → 224 x 224
- Normalize with ImageNet statistics

## 3. Train the Model

Execute the training script:

```bash
python train_cbam_model.py
```

Training performs:

- Forward pass
- Loss calculation
- Backpropagation
- Accuracy monitoring

## 4. Save Model Weights

After training completes, the model weights are saved as:

```
resnet18_pap_cbam.pth
```

This file is required for deployment and inference.

# Training Outputs

The training pipeline produces:

- Model accuracy
- Training loss curves
- Grad-CAM visualizations
- Saved model weights (.pth)

# Important Notes

- Training scripts are intended **only for model training and experimentation**.
- Deployment and application serving instructions are provided in **DEPLOYMENT.md**.
