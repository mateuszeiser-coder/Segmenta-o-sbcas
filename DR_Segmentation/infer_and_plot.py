import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.lightning_modules.task1_lm import Task1LM
from src.data_modules.datasets.task1_dataset import Task1Dataset


# ================= CONFIG =================
DATA_DIR = "/content/drive/MyDrive"

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
# ==========================================


def load_model(ckpt, target):
    backbone = "u2net_lite" if target in [1, 3] else "u2net_full"
    model = Task1LM.load_from_checkpoint(
        ckpt,
        lr=1e-4,
        backbone=backbone,
        target=target,
        reg_type="none",
        reg_weight=0.0,
    )
    model.eval().to(DEVICE)
    return model


def main():

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = Task1Dataset(
        data_dir=DATA_DIR,
        split="test",
        transform=transform
    )

    img, _ = dataset[0]              # (3,H,W)
    img = img.unsqueeze(0).to(DEVICE)

    preds = []

    with torch.no_grad():
        for cls in [1, 2, 3]:
            model = load_model(CKPTS[cls], cls)
            p = torch.sigmoid(model(img))[0, cls - 1]
            preds.append(p.cpu())

    img_np = img[0].cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    for i, cls in enumerate([1, 2, 3]):
        plt.subplot(1, 4, i + 2)
        plt.imshow(preds[i], cmap="gray")
        plt.title(LABELS[cls])
        plt.axis("off")

    plt.suptitle("Qualitative Result", fontsize=16)
    plt.tight_layout()
    plt.savefig("qualitative_result.png", dpi=300)
    plt.show()

    print("✔ qualitative_result.png gerado com sucesso")


if __name__ == "__main__":
    main()
