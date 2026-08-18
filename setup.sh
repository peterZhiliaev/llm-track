#!/usr/bin/env bash
# Environment setup — run on any fresh machine / rented GPU box.
# Goal: spinning up a cloud GPU costs 5 minutes, not an evening.
set -e

pip install --upgrade pip
pip install torch numpy tiktoken transformers datasets

# quick sanity check
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
