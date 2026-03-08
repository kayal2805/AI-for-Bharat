# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

base_dir = Path(r"D:\Bhavani\AI_Hackathon\ai_hackathon2\Pap Reduced\Pap Reduced\SingleCellPAP\95+50_50")
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2,3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# --- Wrap BasicBlock ---
from torchvision.models.resnet import BasicBlock

class BasicBlockWithCBAM(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(BasicBlockWithCBAM, self).__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # 🔥 Apply CBAM after residual addition
        out = self.cbam(out)
        return out

def resnet18_cbam(num_classes=2):
    # Load standard ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace layer4 blocks with CBAM-enhanced blocks
    model.layer4[0] = BasicBlockWithCBAM(
        model.layer4[0].conv1.in_channels,
        model.layer4[0].conv2.out_channels,
        stride=model.layer4[0].stride,
        downsample=model.layer4[0].downsample
    )
    model.layer4[1] = BasicBlockWithCBAM(
        model.layer4[1].conv1.in_channels,
        model.layer4[1].conv2.out_channels,
        stride=model.layer4[1].stride,
        downsample=model.layer4[1].downsample
    )

    # Replace final FC for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18_cbam(num_classes=2).to(device)

### Load your trained model
model.load_state_dict(torch.load(base_dir/r"resnet18_pap_cbam.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

### STEP 3 — Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

### STEP 4 — Load one test image
img_path = base_dir / r"Test/Abnormal/im_Dyskeratotic_009_07.jpg"

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

### STEP 5 — Capture feature maps & gradients (CORE PART)
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.layer4[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

### STEP 6 — Forward + backward pass
output = model(input_tensor)
#predicted_class = output.argmax(dim=1)
class_names = ["Abormal", "Normal"]

predicted_class = output.argmax(dim=1).item()
confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

print("Prediction:", class_names[predicted_class])
print("Confidence:", round(confidence * 100, 2), "%")

model.zero_grad()
output[0, predicted_class].backward()

### STEP 7 — Compute Grad-CAM heatmap
grads = gradients[0][0]             # (C, H, W)
fmap = feature_maps[0][0]     # (C, H, W)

weights = grads.mean(dim=(1, 2))  # Global average pooling

cam = torch.sum(weights[:, None, None] * fmap, dim=0)

cam = F.relu(cam)

cam_min = cam.min()
cam_max = cam.max()

if cam_max - cam_min != 0:
    cam = (cam - cam_min) / (cam_max - cam_min)
else:
    cam = torch.zeros_like(cam)

### STEP 8 — Overlay heatmap on image
##cam = cam.cpu().numpy()
cam = cam.detach().cpu().numpy()
cam = cv2.resize(cam, (224, 224))

img_np = np.array(img.resize((224, 224))) / 255.0
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

overlay = heatmap * 0.4 + img_np
overlay = np.uint8(255 * overlay)

### STEP 9 — Save result
cv2.imwrite(str(base_dir/r"gradcam_manual_cbam.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))



