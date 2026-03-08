from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from fastapi import Request
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from gradcam import generate_gradcam
import torch.nn as nn
import time

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# ✅ Mount static directory so Grad-CAM results can be served
app.mount("/static", StaticFiles(directory="static"), name="static")

# get clases from ResNetTraining_CBAM.py
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
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18_cbam(num_classes=2)
model.load_state_dict(torch.load("resnet18_pap_cbam.pth", map_location=device))
model.to(device)
model.eval()


# Preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred].item()

    orig_path, gradcam_path = generate_gradcam(model, input_tensor, pred, img)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": "normal" if pred == 1 else "abnormal",
        "confidence": round(confidence * 100, 2),
        "original_url": f"/{orig_path}",
        "gradcam_url": f"/{gradcam_path}",
        "timestamp": int(time.time())   
    })

#    return JSONResponse({
#        "prediction": "normal" if pred == 1 else "abnormal",
#        "confidence": round(confidence * 100, 2),
#        "original_url": f"/{orig_path}",
#        "gradcam_url": f"/{gradcam_path}"
#        })


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Upload an image for prediction</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """
