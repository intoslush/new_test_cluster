#!/bin/bash

set -u

export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DATASET_NAME="CUHK-PEDES"
EXP_DESC="${1:-CUHK-PEDES baseline with cross-modal confidence calibration for pseudo supervision}"
ROOT_LOG="${2:-./output.log}"

mkdir -p "$(dirname "$ROOT_LOG")"
exec > >(tee -a "$ROOT_LOG") 2>&1

echo "[$(date '+%F %T')] Launching experiment"
echo "Dataset: ${DATASET_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Root log: ${ROOT_LOG}"
echo "Description: ${EXP_DESC}"

PYTHONUNBUFFERED=1 torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29505 \
  train.py \
  --name new_rasa \
  --config ./configs/PS_cuhk_pedes.yaml \
  --checkpoint ./data/ALBEF/ALBEF.pth \
  --dataset_name $DATASET_NAME \
  --root_dir ./re_id \
  --num_epoch 35 \
  --cluster_id_mode "cluster" \
  --massage "${EXP_DESC}"
