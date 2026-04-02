import argparse
import os
import random

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description="ICPG Args")
    parser.add_argument("--config", default="./configs/PS_cuhk_pedes.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_mAP", action="store_true", help="whether to evaluate mAP")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--batch_size", default=13, type=int)
    parser.add_argument("--embed_dim", default=577, type=int)
    parser.add_argument("--tokenizer_path", default="./bert-base-uncased", type=str)
    parser.add_argument("--dataset_name", default="CUHK-PEDES", type=str, help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--root_dir", default="./re_id", type=str)
    parser.add_argument("--num_epoch", default=30, type=int)

    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=500)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test", help="use val or test split for evaluation")
    parser.add_argument("--resume_ckpt_file", default="", help="resume from ...")
    parser.add_argument(
        "--cluster_id_mode",
        default="cluster",
        type=str,
        help="choose from ['cluster', 'instance', 'unique_noise']",
    )
    parser.add_argument("--massage", default=" ")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for randomly initialized modules")

    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest="training", default=True, action="store_false")
    parser.add_argument("--swap_epoch", type=int, default=99, help="from which epoch to use pseudo augmentation")

    return parser.parse_args()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
