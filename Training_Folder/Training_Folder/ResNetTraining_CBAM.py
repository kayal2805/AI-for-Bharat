# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Resize to 224x224, normalize like ImageNet
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
base_dir = Path(r"D:\ai_hackathon2\Pap Reduced\Pap Reduced\SingleCellPAP\95+50_50")
train_dir = base_dir / r"Training"
test_dir = base_dir / r"Test"


train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Replace final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Normal vs Abnormal

model = model.to(device)

# Define loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20  # adjust for hackathon proof-of-concept

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop_count = 0;
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        loop_count = loop_count+1
        print(loop_count)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

torch.save(model.state_dict(),  base_dir/r"resnet18_pap_cbam.pth")
# Test / Evaluation
model.eval()
correct = 0
total = 0

class_names = test_loader.dataset.classes

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        probs = F.softmax(outputs, dim=1)

        # 🔥 Loop inside batch
        for i in range(images.size(0)):

            global_index = batch_idx * test_loader.batch_size + i

            # Prevent index overflow in last batch
            if global_index >= len(test_loader.dataset.samples):
                continue

            file_path = test_loader.dataset.samples[global_index][0]
            filename = os.path.basename(file_path)

            predicted_class = class_names[preds[i].item()]
            true_class = class_names[labels[i].item()]
            confidence = probs[i][preds[i].item()].item()

            print(f"File: {filename}")
            print(f"True: {true_class}")

            # ✅ If/Else Statement
            if predicted_class.lower() == "normal":
                print("Model says: NORMAL image")
            else:
                print("Model says: ABNORMAL image")

            print(f"Confidence: {confidence:.4f}")
            print("-" * 40)

print("Accuracy:", correct / total)








