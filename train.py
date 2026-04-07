import os
import os.path as op
import time
from datetime import timedelta

import torch

from dataset import get_dataloder
from my_model import build_model
from processor.processor import do_train
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.options import get_args, set_seed


if __name__ == "__main__":
    args = get_args()
    seed = args.seed + get_rank()
    set_seed(seed)
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    device = "cuda"
    cur_time = time.strftime("%m%d_%H%M", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{name}")
    logger = setup_logger(args.name, save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=1800))
        synchronize()
        logger.info("Distributed training initialized")

    logger.info("Using %s GPUs", num_gpus)
    if str(args.massage).strip():
        logger.info("Experiment: %s", str(args.massage).strip())
    logger.info(str(args).replace(",", "\n"))
    save_train_configs(args.output_dir, args)
    model = build_model(args)
    logger.info("Total params: %2.fM", sum(p.numel() for p in model.parameters()) / 1000000.0)

    if args.distributed:
        model = model.to(torch.device("cuda", args.local_rank))
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )
        logger.info("Model wrapped with DistributedDataParallel")
    else:
        logger.info("Model initialized")
        model.to(device)

    train_loader, val_loader, test_loader, cluster_lodaer = get_dataloder(args)
    logger.info("Dataloader ready")
    start_epoch = 1

    do_train(start_epoch, args, model, train_loader, None, None, cluster_lodaer, test_loader)

    if args.distributed:
        torch.distributed.destroy_process_group()
    logger.info("Test complete!")
