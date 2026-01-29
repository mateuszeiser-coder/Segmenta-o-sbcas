import torch
import matplotlib.pyplot as plt

from src.lightning_modules.task1_lm import Task1LM
from src.data_modules.task1_dm import Task1DM
from src.utils import set_seed


# ================= CONFIG =================
DATA_DIR = "/content/drive/MyDrive/A. Segmentation"
INPUT_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    model.to(DEVICE)
    model.eval()
    return model


def main():
    set_seed(42)

    # Data (pegamos UMA imagem só)
    dm = Task1DM(
        task="segment",
        data_dir=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=1,
        num_workers=2,
        balanced_sampling=False,
        target=1,  # irrelevante aqui
    )
    dm.setup(stage="test")
    batch = next(iter(dm.test_dataloader()))

    image = batch["image"].to(DEVICE)
    img_np = image[0].permute(1, 2, 0).cpu().numpy()

    preds = {}

    for cls in [1, 2, 3]:
        model = load_model(CKPTS[cls], cls)
        with torch.no_grad():
            p = torch.sigmoid(model(image))
            preds[cls] = (p > 0.5).float()[0, 0].cpu().numpy()

    # ===== Plot =====
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(img_np)
    axs[0].set_title("Original Image", fontsize=12)
    axs[0].axis("off")

    for i, cls in enumerate([1, 2, 3], start=1):
        axs[i].imshow(preds[cls], cmap="gray")
        axs[i].set_title(LABELS[cls], fontsize=12)
        axs[i].axis("off")

    plt.suptitle("Qualitative Result", fontsize=16)
    plt.tight_layout()
    plt.savefig("qualitative_result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
