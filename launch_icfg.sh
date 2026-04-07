#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
DATASET_NAME="ICFG-PEDES"
MASTER_PORT="${MASTER_PORT:-29505}"

CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NUM_GPUS}"
  --master-port="${MASTER_PORT}"
  train.py
  --name new_rasa
  --checkpoint ./data/ALBEF/ALBEF.pth
  --dataset_name "${DATASET_NAME}"
  --root_dir ./re_id
  --num_epoch 30
  --config ./configs/PS_icfg_pedes.yaml
)

PYTHONUNBUFFERED=1 "${CMD[@]}"
