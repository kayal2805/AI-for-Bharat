
## An Exploratory Explainability‑First AI Approach to Cervical Cancer Cytology Screening (PapML)

## Introduction

This project deploys a **ResNet18 model enhanced with CBAM attention and Grad‑CAM visualization** using **FastAPI inside a Docker container**.

The application allows users to upload cytology images and receive:

- **Binary classification:** Normal vs Abnormal
- **Grad‑CAM heatmap:** highlights nuclei regions for interpretability

The solution is optimized for **CPU‑only EC2 instances**, ensuring cost‑effective deployment without large CUDA dependencies.

# Features

- **ResNet18 backbone** for image classification
- **CBAM (Convolutional Block Attention Module)** for improved feature extraction
- **Grad‑CAM** for visual interpretability
- **FastAPI backend** with two interfaces:

### Swagger UI
Auto‑generated API documentation available at:

```
/docs
```

Allows developers to test API endpoints directly.

### Jinja2 Web Interface

User‑facing interface that allows:

- Image upload
- Prediction results display
- Grad‑CAM visualization

Additional features:

- Dockerized deployment for reproducibility
- CPU‑only PyTorch build for smaller container size
- File format support: `.jpg`, `.bmp`, and other Pillow‑supported formats
- Sample dataset included for testing

# How to Use

This section explains how to interact with the deployed application.

### Access the Application

```
http://3.238.82.0:8000/
```

### Upload an Image

1. Click **Upload**
2. Select an image file
3. Click **Submit**

### Test Data

You may use the included **SipakMed dataset samples** for testing.

### Image Requirements

- Single‑cell cervical cytology images
- Preferred formats:
  - `.jpeg`
  - `.bmp`
- Images must be downloaded locally before upload.

### Output

The application returns:

- **Classification result** (Normal / Abnormal)
- **Grad‑CAM heatmap visualization**

You can click the **Back button** on the results page to upload another image.

# Deployment Instructions

The model has already been trained and deployed.

If you wish to deploy it locally or on your own EC2 instance, follow the steps below.

> **Note:** `papml-app` (Pap Machine Learning) is the Docker image name used in this project.  
You may use any name when building your Docker image.

# Requirements

## System

- Python 3.10
- Docker
- AWS EC2 (t3.medium or larger recommended)

## Python Dependencies

Defined in `requirements.txt`:

```
torch==2.2.2+cpu
torchvision==0.17.2+cpu
fastapi==0.134.0
uvicorn==0.41.0
numpy==1.26.4
opencv-python==4.9.0.80
python-multipart==0.0.22
pillow==11.1.0
```

## System Packages

Required for OpenCV:

```
libgl1
libglib2.0-0
```

# Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Build the Docker Image

```bash
docker build -t papml-app .
```

### 3. Run the Container

```bash
docker run -p 8000:8000 papml-app
```

# Deployment on AWS EC2

### 1. Launch EC2 Instance

Recommended configuration:

- OS: Ubuntu
- Instance: **t3.medium** or larger

### 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io
```

### 3. Copy Project Files

Upload your project files to EC2 using:

- `scp`
- `git clone`
- SSH file transfer tools

### 4. Build and Run the Container

```bash
docker build -t papml-app .
docker run -d -p 8000:8000 papml-app
```

### Optional: Mount Project Folder for Iteration

```bash
docker run -d -p 8000:8000   -v /home/ec2-user/PrototypeBuilding:/app   papml-app
```

### 5. Configure Security Group

Allow inbound traffic on:

```
Port: 8000
Protocol: TCP
```

### 6. Access the Application

```
http://<EC2-Public-IP>:8000/
```

Replace `<EC2-Public-IP>` with your EC2 instance public IP.

# Final Setup Summary

- Dockerfile installs OpenCV system dependencies
- Lean **CPU‑only container build (~600‑700 MB)** vs large GPU builds
- Fully functional **FastAPI app with ResNet18 + CBAM + Grad‑CAM inference**
- Improved UI features:
  - Back button
  - Clean upload interface
  - Cache‑busting for image refresh
- Clear deployment workflow for **Uvicorn and Docker**
