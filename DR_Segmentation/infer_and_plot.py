import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from DR_Segmentation.models.model import DRSegModel
from DR_Segmentation.data_modules.task1_dataset import Task1Dataset


# ============================
# CONFIGURAÇÕES
# ============================

DATA_DIR = "/content/drive/MyDrive/A. Segmentation"
CKPT_PATH = "lightning_logs/version_0/checkpoints/epoch=0-step=XXX.ckpt"
TARGET = 1               # 1=MA, 2=HE, 3=EX
OUTPUT_DIR = "paper_figures/target_{}".format(TARGET)
SAMPLES = [3, 17, 42]    # índices fixos (reprodutibilidade)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================
# FUNÇÕES AUXILIARES
# ============================

def overlay_mask(image, mask, color=(255, 0, 0)):
    """
    image: HxWx3 uint8
    mask: HxW float [0,1]
    """
    image = image.copy()
    mask = mask > 0.5
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)


# ============================
# LOAD MODEL
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DRSegModel.load_from_checkpoint(CKPT_PATH)
model.to(device)
model.eval()


# ============================
# LOAD DATASET (TEST)
# ============================

test_ds = Task1Dataset(
    split="test",
    data_dir=DATA_DIR,
    target=TARGET,
    transform=None
)


# ============================
# INFERÊNCIA + PLOTS
# ============================

for idx in SAMPLES:
    img, gt = test_ds[idx]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

    # imagem original (C,H,W -> H,W,C)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_np = gt.numpy()

    overlay_pred = overlay_mask(img_np, pred)

    # ============================
    # FIGURA (PADRÃO PAPER)
    # ============================

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay_pred)
    plt.axis("off")

    out_path = os.path.join(
        OUTPUT_DIR, f"sample_{idx}_target_{TARGET}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {out_path}")
