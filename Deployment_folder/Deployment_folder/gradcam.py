import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

def generate_gradcam(model, input_tensor, predicted_class, original_image, save_dir="static"):
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward + backward
    output = model(input_tensor)
    model.zero_grad()
    output[0, predicted_class].backward()

    grads = gradients[0][0]   # (C, H, W)
    fmap = feature_maps[0][0] # (C, H, W)

    weights = grads.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = F.relu(cam)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    img_np = np.array(original_image.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * 0.4 + img_np
    overlay = np.uint8(255 * overlay)

    Path(save_dir).mkdir(exist_ok=True)
    save_path = Path(save_dir) / "gradcam_result.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    img_resized = np.uint8(original_image.resize((224, 224)))
    orig_path = Path(save_dir) / "original.jpg"
    cv2.imwrite(str(orig_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    return str(orig_path), str(save_path)
