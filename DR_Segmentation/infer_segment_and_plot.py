import os
import torch
import matplotlib.pyplot as plt

from src.lightning_modules.task1_lm import Task1LM
from src.data_modules.task1_dm import Task1DM
from src.utils import set_seed


# ================= CONFIGURAÇÕES =================
DATA_DIR = "/content/drive/MyDrive/A. Segmentation"
CKPT_PATH = "/content/Segmenta-o-sbcas/outputs_drac/task1/debug/class_3/SEED_class_avg_epoch=XX-Dice=YY.ckpt"
TARGET = 3          # 1, 2 ou 3
BATCH_SIZE = 1
INPUT_SIZE = 1024
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================


def main():

    set_seed(42)

    # 1️⃣ DataModule (mesmo do treino)
    dm = Task1DM(
        task="segment",
        data_dir=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        balanced_sampling=False,
        target=TARGET,
    )
    dm.setup(stage="test")

    # 2️⃣ LightningModule (carregado do checkpoint)
    model = Task1LM.load_from_checkpoint(
        CKPT_PATH,
        lr=1e-4,          # não importa para inferência
        backbone="u2net_lite" if TARGET in [1, 3] else "u2net_full",
        target=TARGET,
        reg_type="none",
        reg_weight=0.0,
    )

    model.to(DEVICE)
    model.eval()

    # 3️⃣ Pegar UMA amostra
    batch = next(iter(dm.test_dataloader()))
    image = batch["image"].to(DEVICE)
    mask_gt = batch["mask"].to(DEVICE)

    # 4️⃣ Inferência
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    # 5️⃣ Plot (estilo artigo)
    img = image[0].permute(1, 2, 0).cpu().numpy()
    gt = mask_gt[0, 0].cpu().numpy()
    pr = pred[0, 0].cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pr, cmap="gray")
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    axs[3].imshow(img)
    axs[3].imshow(pr, alpha=0.4, cmap="jet")
    axs[3].set_title("Overlay")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_example.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
