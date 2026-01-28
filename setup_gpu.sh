#!/usr/bin/env bash
set -e

echo "==> Atualizando pip"
python -m pip install --upgrade pip

echo "==> Instalando PyTorch + CUDA (default: cu121)"
echo "    Se sua máquina for CUDA 11.8, troque cu121 por cu118."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Instalando dependências do projeto"
pip install -r requirements.text

echo "==> Sanity check (CUDA)"
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> OK. Agora rode seu comando de treino."
