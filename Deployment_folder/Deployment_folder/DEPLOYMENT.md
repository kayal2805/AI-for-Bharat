
# Deployment Guide

This directory contains the code and configuration files required to deploy the trained **ResNet18 + CBAM + Grad-CAM** model using **FastAPI** and **Docker**.

> **Note**  
> `papml-app` (Pap Machine Learning) is the Docker image name/tag used in this project.  
> You may change this name while building the Docker image if desired.

> **Important**  
> Ensure that the dataset and model weights are available locally before running the application.


# Local System Deployment

Follow the steps below to deploy the application on your local machine.

## 1. Clone the repository

```bash
git clone <repository-url>
cd Deployment_folder
```

## 2. Build the Docker image

```bash
docker build -t papml-app .
```

## 3. Run the container locally

```bash
docker run -p 8000:8000 papml-app
```

## 4. Access the application

Open the browser and navigate to:

```
http://localhost:8000/
```

# AWS EC2 Deployment

The application can also be deployed on an AWS EC2 instance.

## 1. Launch EC2 Instance

Recommended configuration:

- OS: Ubuntu
- Instance Type: t3.medium (recommended)

## 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io
```

## 3. Copy project files to EC2

Upload the project directory using:

- scp
- git clone
- or file transfer via SSH tools

## 4. Build and run the Docker container

```bash
docker build -t papml-app .
docker run -d -p 8000:8000 papml-app
```

## Optional: Mount EC2 project folder for live iteration

```bash
docker run -d -p 8000:8000   -v /home/ec2-user/PrototypeBuilding:/app   papml-app
```

## 5. Configure EC2 Security Group

Allow inbound traffic on:

```
Port: 8000
Protocol: TCP
```

## 6. Access the deployed application

```
http://<EC2-Public-IP>:8000/
```

Replace `<EC2-Public-IP>` with your instance's public IP address.


# Deployment Summary

- Docker is used for containerized deployment.
- The application exposes port **8000**.
- FastAPI serves the Pap cell classification application.
- Grad-CAM visualizations are generated during inference.

Important Note:
In the deployment module, we intended to provide the .pth file of ResNet18. However, the file size was large (~44 MB), and even after compression it remained ~40 MB. Hence, we were unable to attach it. We can send it via email if required. Alternatively, the file can be generated using the code that we have already shared.  