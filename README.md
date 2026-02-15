AI-Based Cervical Cytology Screening System

An explainability-first deep learning system for automated cervical cytology screening, designed for low-resource, community-level deployment using smartphone-compatible microscopy images.
The goal is to enable early detection support while maintaining clinical transparency and trust.

Overview

This project performs binary classification of cervical cytology images into:

1. Normal

2. Abnormal

Explanation

The model serves as a screening and triage assistant, helping identify suspicious samples that require expert review.
It is not intended to replace clinicians, but to reduce workload and improve early detection in settings where specialists are scarce.

The system integrates:

1. ResNet18 + CBAM attention → improves feature focus on cellular morphology

2. Grad-CAM explainability → visually shows why the model predicted abnormality

3. Robustness to low-quality JPEG images (Q95 & Q50) → ensures usability with smartphone microscopy

4. Confidence-based clinician triage → prioritizes uncertain or high-risk cases

Problem Motivation

Cervical cancer remains a high-burden yet preventable disease, but screening uptake is limited due to:

1. Low awareness

2. Gender-related discomfort during sampling

3. Shortage of trained staff and laboratory infrastructure

Explanation

Many regions lack cytology experts, laboratory equipment, and structured screening programs.
An AI-assisted approach allows rapid preliminary screening, enabling:

1. Earlier referral of abnormal cases

2. Reduced diagnostic delays

3. Scalable screening in resource-constrained environments

Key Innovation

1. Explainability-first AI, not just accuracy

2. Generalizes across low-quality, heterogeneous real-world images

3. Designed for community screening and triage

4. Enhances clinical trust, transparency, and deployability

Explanation

Most medical AI models focus only on prediction accuracy in ideal lab datasets.
This system instead prioritizes:

1. Real-world usability

2. Interpretability for clinicians

3. Deployment in underserved populations

This makes the solution clinically meaningful, not just technically accurate.

System Architecture

Workflow:
Capture → Preprocess → ResNet18-CBAM Inference → Grad-CAM Explainability →
Clinician Review → Final Screening Report → Audit Logging

Explanation of Workflow Steps

1. Capture
Cytology images are obtained using microscopes or smartphone-mounted adapters, enabling low-cost image acquisition.

2. Preprocess
Images are resized, normalized, and converted into tensors so they can be consistently interpreted by the neural network.

3. Inference (ResNet18-CBAM)
The deep learning model analyzes cell morphology and nuclear features to classify the image as Normal or Abnormal.

4. Grad-CAM Explainability
A heatmap highlights regions influencing the prediction, improving trust and interpretability for medical users.

5. Clinician Review
Doctors validate or override AI predictions, ensuring human-in-the-loop safety.

6. Reporting & Logging
All results, confidence scores, and model versions are stored for traceability, auditing, and future improvement.

Features
1. Low-Cost Smartphone Imaging

i) Works with low-resolution microscopic images

ii) Smartphone-like images simulated from SipakMed dataset

iii) Uses JPEG compression Q95 and Q50

iv) Eliminates need for expensive digital microscope cameras

v) Enables affordable large-scale screening

Explanation:
This feature ensures the system can function in rural clinics and community health programs without specialized imaging hardware.

2. Binary AI Screening

i) Classifies cytology images as Normal vs Abnormal

ii) Provides simple, interpretable output

iii) Supports non-expert preliminary triage

Explanation:
A binary decision simplifies deployment and allows health workers with minimal training to identify suspicious samples for referral.

3. Explainable AI with Grad-CAM

i) Highlights nucleus-centered discriminative regions

ii) Aligns with pathological assessment criteria

iii) Improves clinical trust and transparency

Explanation:
Instead of a “black-box” prediction, clinicians can see what the AI is focusing on, making the tool safer and more acceptable in medical workflows.

4. Confidence Scoring

i) Outputs prediction probability

ii) Helps prioritize uncertain or critical cases

Explanation:
Confidence scores allow risk-based triage, ensuring borderline or suspicious samples receive faster expert attention.

Technical Stack

Core Environment

1. Python 3.10

2. PyTorch

3. Torchvision

4. OpenCV

5. NumPy

6. Matplotlib

Model Components

1. ResNet18 backbone → deep feature extraction

2. CBAM attention module → improved spatial & channel focus

3. Grad-CAM explainability → visual reasoning support

Explanation:
This stack balances research-grade deep learning capability with practical deployment feasibility.

Performance & Robustness

1. Evaluated across multiple JPEG compression settings

2. Maintains stable predictions in low-quality conditions

3. Tracks accuracy, misclassification, and confidence metrics

Explanation:
Robustness to image degradation is critical for real-world medical deployment, especially with smartphone imaging variability.

Clinical Impact

This solution enables:

1. Affordable large-scale screening

2. Early detection in underserved populations

3. Reduced specialist workload

4. Explainable AI-assisted clinical decision support

Explanation:
The ultimate goal is improved public health outcomes, not just algorithm performance.

Future Work

1. Pixel-level cell segmentation

2. Multi-cell slide analysis

3. Lightweight mobile deployment

4. Cloud-based screening dashboards

Explanation:
These steps move the system toward clinical-grade deployment and scalability.

Team

Team Name: Cris prAI
Team Leader: Dr. Bhavani
Team Members: Kayalvizhi S, Salim Moula Shaikh
