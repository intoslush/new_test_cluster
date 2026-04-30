#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
DATASET_NAME="ICFG-PEDES"
EXP_DESC="${1:-ICFG-PEDES no-cluster instance-label test: each sample uses an independent pseudo label}"
ROOT_LOG="${2:-./output_icfg_instance.log}"
MASTER_PORT="${MASTER_PORT:-29506}"

mkdir -p "$(dirname "$ROOT_LOG")"
exec > >(tee -a "$ROOT_LOG") 2>&1

echo "[$(date '+%F %T')] Launching experiment"
echo "Dataset: ${DATASET_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Root log: ${ROOT_LOG}"
echo "Description: ${EXP_DESC}"
echo "Master port: ${MASTER_PORT}"

CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NUM_GPUS}"
  --master-port="${MASTER_PORT}"
  train.py
  --name new_rasa
  --config ./configs/PS_icfg_pedes.yaml
  --checkpoint ./data/ALBEF/ALBEF.pth
  --dataset_name "${DATASET_NAME}"
  --root_dir ./re_id
  --num_epoch 35
  --cluster_id_mode instance
  --massage "${EXP_DESC}"
)

PYTHONUNBUFFERED=1 "${CMD[@]}"
