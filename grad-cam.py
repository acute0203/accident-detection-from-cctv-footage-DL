import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torchsummary import summary
from train import get_model
import argparse
import torch.nn as nn

# === CLI 參數 ===
parser = argparse.ArgumentParser(description='Binary Classification with ResNet, ReXNet or customized model')
parser.add_argument('--model', type=str, default='cs', choices=['res', 'rex', 'cs'], help='Model name')
parser.add_argument('--img', type=str, help='IMG Path')

args = parser.parse_args()

# === 載入模型到 CPU 並顯示架構 ===
device = torch.device("cpu")
model = get_model(args.model)
model.load_state_dict(torch.load(f"deep_cctv_model-{args.model}.pth", map_location=device))
model.eval()
summary(model, (3, 224, 224), device="cpu")

# === 自動尋找最後一層 Conv2d 作為 Grad-CAM 的 target layer ===
def find_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model.")

# === Grad-CAM 工具類 ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

# === 圖片前處理 ===
image_path = args.model
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to(device)

# === Grad-CAM 熱點圖產生 ===
target_layer = find_last_conv_layer(model)
grad_cam = GradCAM(model, target_layer)
cam = grad_cam.generate(input_tensor)

# === 圖像與熱點圖顯示（左：原圖，右：Grad-CAM 疊圖） ===
image_np = np.array(image.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image.resize((224, 224)))
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(overlay_rgb)
axs[1].set_title("Grad-CAM")
axs[1].axis("off")

plt.tight_layout()
plt.show()
