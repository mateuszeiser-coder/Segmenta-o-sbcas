import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "/content/drive/MyDrive/A. Segmentation"

IMG_DIR = os.path.join(
    DATA_DIR,
    "1. Original Images",
    "a. Training Set"
)

files = sorted([
    os.path.basename(p)
    for p in glob.glob(os.path.join(IMG_DIR, "*.png"))
])

assert len(files) > 0, f"Nenhuma imagem encontrada em {IMG_DIR}"

train_f, val_f = train_test_split(files, test_size=0.1, random_state=42)

df = pd.concat([
    pd.DataFrame({"filename": train_f, "split": "train"}),
    pd.DataFrame({"filename": val_f,   "split": "val"}),
], ignore_index=True)

out_csv = os.path.join(DATA_DIR, "segmentation_split.csv")
df.to_csv(out_csv, index=False)

print("CSV criado em:", out_csv)
print(df["split"].value_counts())
