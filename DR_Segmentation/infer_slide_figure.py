import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from DR_Segmentation.models.model import DRSegModel
from DR_Segmentation.data_modules.task1_dataset import Task1Dataset



# =========================
# CONFIG
# =========================

DATA_DIR = "/content/drive/MyDrive/A. Segmentation"

CKPT = {
    1: "lightning_logs/version_0/checkpoints/target1.ckpt",
    2: "lightning_logs/version_1/checkpoints/target2.ckpt",
    3: "lightning_logs/version_2/checkpoints/target3.ckpt",
}

IMAGE_INDEX = 17  # mesma imagem para todas as classes
OUT_DIR = "slide_figures"
OUT_NAME = "task1_qualitative_result.png"

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD IMAGE ONCE
# =========================

datasets = {
    t: Task1Dataset(
        split="test",
        data_dir=DATA_DIR,
        target=t,
        transform=None
    )
    for t in [1, 2, 3]
}

img, _ = datasets[1][IMAGE_INDEX]
img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# =========================
# INFERENCE PER CLASS
# =========================

preds = {}

for t in [1, 2, 3]:
    model = DRSegModel.load_from_checkpoint(CKPT[t])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

    preds[t] = pred


# =========================
# PLOT — IGUAL AO SLIDE
# =========================

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(img_np, cmap="gray")
plt.axis("off")

titles = {
    1: "Class 1\nMicrovascular Abnormality",
    2: "Class 2\nNonperfusion Area",
    3: "Class 3\nNeovascularization",
}

for i, t in enumerate([1, 2, 3], start=2):
    plt.subplot(1, 4, i)
    plt.title(titles[t])
    plt.imshow(preds[t] > 0.5, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, OUT_NAME), dpi=300)
plt.close()

print("[OK] Slide-style qualitative figure saved.")
