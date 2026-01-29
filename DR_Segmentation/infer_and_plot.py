import os
import torch
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_modules.datasets.task1_dataset import Task1Dataset
from src.lightning_modules.task1_lm import Task1LM


# ===================== CONFIG =====================

# Pode ser:
# "/content/drive/MyDrive"
# ou "/content/drive/MyDrive/A. Segmentation"
DATA_DIR = "/content/drive/MyDrive"

# Caminhos dos checkpoints (ajuste se necessário)
CKPTS = {
    1: "/content/Segmenta-o-sbcas/outputs_drac/task1/debug/class_1/42_class_1_epoch=00-Dice=0.0000.ckpt",
    2: "/content/Segmenta-o-sbcas/outputs_drac/task1/debug/class_2/42_class_2_epoch=00-Dice=0.6453.ckpt",
    3: "/content/Segmenta-o-sbcas/outputs_drac/task1/debug/class_3/42_class_3_epoch=00-Dice=0.0000.ckpt",
}

LABELS = {
    1: "Class 1\nMicrovascular\nAbnormality",
    2: "Class 2\nNonperfusion\nArea",
    3: "Class 3\nNeovascularization",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_FIG = "qualitative_result.png"

# ================================================


def load_model(ckpt_path: str) -> Task1LM:
    """Carrega modelo Lightning a partir de checkpoint"""
    model = Task1LM.load_from_checkpoint(
        ckpt_path,
        map_location=DEVICE
    )
    model.eval()
    model.to(DEVICE)
    return model


def main():

    # -------- Transform (SEM AUGMENTATION) --------
    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2()
    ])

    # -------- Dataset --------
    dataset = Task1Dataset(
        data_dir=DATA_DIR,
        split="test",
        transform=transform
    )

    # Pega UMA imagem (qualitativo)
    img, gt = dataset[0]     # img: (3,H,W), gt: (3,H,W)
    img = img.to(DEVICE)

    # -------- Inferência (uma classe por modelo) --------
    preds = []

    with torch.no_grad():
    for cls in [1, 2, 3]:
        model = load_model(CKPTS[cls])

        out = torch.sigmoid(model(img.unsqueeze(0)))[0]
        # out pode ser (1,H,W) ou (3,H,W)

        if out.shape[0] == 1:
            # modelo binário (um checkpoint por classe)
            preds.append(out[0].cpu())
        else:
            # modelo multiclasse
            preds.append(out[cls - 1].cpu())

    # -------- Plot --------
    img_np = img.cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    for i, cls in enumerate([1, 2, 3]):
        plt.subplot(1, 4, i + 2)
        plt.imshow(preds[i], cmap="gray")
        plt.title(LABELS[cls])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()

    print(f"[OK] Figura salva em: {OUT_FIG}")


if __name__ == "__main__":
    main()
