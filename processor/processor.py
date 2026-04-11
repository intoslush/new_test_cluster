import json
import logging
import os
import pprint
from io import StringIO
from typing import Any, Dict

import torch
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast

import utils.optimizer as utils
from optim import create_optimizer
from scheduler import create_scheduler
from utils.comm import get_rank, is_main_process, synchronize

from .eval_hooks import evaluate_and_checkpoint
from .pseudo import generate_and_broadcast_pseudo_labels
from .weights import compute_dynamic_weights


def _setup_optim_sched(config: Dict[str, Any], model):
    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    scheduler, _ = create_scheduler(arg_sche, optimizer)
    return optimizer, scheduler


def do_train(start_epoch, args, model, train_loader, evaluator, checkpointer, cluster_loader, test_loader):
    device = torch.device("cuda")
    num_epoch = args.num_epoch

    is_distributed = args.distributed
    rank = get_rank()
    is_main = is_main_process()

    logger = logging.getLogger(args.name)
    logger.info("start training (rank %s)", rank)

    yaml = YAML(typ="rt")
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file)

    optimizer, scheduler = _setup_optim_sched(config, model)

    start_epoch = 0
    warmup_epochs = config["schedular"]["warmup_epochs"]
    step_size = 100
    warmup_iterations = warmup_epochs * step_size

    use_amp = getattr(args, "use_amp", True) and device.type == "cuda"
    logger.info("Using AMP: %s", use_amp)
    scaler = GradScaler(enabled=use_amp)

    model_without_ddp = model.module if hasattr(model, "module") else model

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)

    best = 0.0
    best_epoch = 0
    best_log = {}

    for epoch in range(start_epoch, num_epoch + 1):
        dump_epoch_1based = 7
        target_epoch = start_epoch + dump_epoch_1based - 1

        if is_main and epoch == target_epoch:
            logger.info("========== [Dump args & config @ epoch=%d] ==========", epoch)
            try:
                logger.info("args (vars):\n%s", pprint.pformat(vars(args), width=120, sort_dicts=False))
            except Exception:
                logger.info("args:\n%s", str(args))
            try:
                buf = StringIO()
                yaml.dump(config, buf)
                logger.info("config (yaml):\n%s", buf.getvalue())
            except Exception:
                logger.info("config (repr):\n%s", pprint.pformat(config, width=120))
            logger.info("========== [End dump] ==========")

        if epoch < 5 or epoch % 2 == 1:
            pseudo_cache = generate_and_broadcast_pseudo_labels(
                epoch=epoch,
                device=device,
                is_main=is_main,
                is_distributed=is_distributed,
                rank=rank,
                cluster_loader=cluster_loader,
                model=model,
                args=args,
                config=config,
                logger=logger,
                enable_nmi_ari=True,
                cluster_until_epoch=50,
            )
            train_loader.dataset.mode = "train"
            train_loader.dataset.set_pseudo_labels(pseudo_cache["pseudo_labels"].cpu())
            train_loader.dataset.set_sample_confidences(
                pseudo_cache["sample_confidence"].cpu(),
                pseudo_cache["confidence_group"].cpu(),
            )

            if bool(config.get("reset_queue_each_epoch", True)):
                synchronize()
                model.reset_queues(random_init=bool(config.get("queue_random_reinit", False)))
                if is_main:
                    logger.info(
                        "[Rank %s] Reset contrast queue (random_init=%s)",
                        rank,
                        bool(config.get("queue_random_reinit", False)),
                    )
                synchronize()

        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_valid_indices"):
            train_loader.sampler.set_valid_indices(train_loader.dataset.valid_indices)
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if is_distributed:
            synchronize()

        if epoch > 0:
            try:
                scheduler.step(epoch)
            except Exception:
                scheduler.step()

        model.train()
        logger.info("[Rank %s] Start epoch %s mini-batch loop", rank, epoch)

        dynamic_weights = compute_dynamic_weights(
            epoch=epoch,
            num_epoch=num_epoch,
            base_weights=config["weights"],
            schedule=config.get("weights_schedule"),
        )
        if is_main:
            logger.info("[Rank %s] Epoch %s loss weights: %s", rank, epoch, dynamic_weights)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
            if epoch > 0 or not config.get("warm_up", False):
                alpha = config["alpha"]
            else:
                alpha = config["alpha"] * min(1.0, (n_iter + 1) / len(train_loader))

            if n_iter % args.log_period == 0:
                logger.info("Epoch %s batch %s/%s loss calculation", epoch, n_iter, len(train_loader))

            with autocast(device_type=device.type, enabled=use_amp):
                loss_dict: Dict[str, torch.Tensor] = model(batch, alpha, config, epoch)
                loss = 0.0
                for k, v in loss_dict.items():
                    weight = dynamic_weights.get(k, config["weights"].get(k, 0.5))
                    loss = loss + weight * v

            if n_iter % args.log_period == 0:
                relation_stats = getattr(model_without_ddp, "latest_relation_stats", None)
                if relation_stats is not None:
                    logger.info(
                        "[Relation][epoch %s batch %s] active=%s verified_pairs=%d q_aa=%.4f q_ab=%.4f rel_ab=%.4f rel_ba=%.4f self_min=%.4f",
                        epoch,
                        n_iter,
                        int(relation_stats.get("active", 0.0) > 0),
                        int(relation_stats.get("num_verified_pairs", 0.0)),
                        float(relation_stats.get("mean_q_aa", 0.0)),
                        float(relation_stats.get("mean_q_ab", 0.0)),
                        float(relation_stats.get("mean_rel_ab", 0.0)),
                        float(relation_stats.get("mean_rel_ba", 0.0)),
                        float(relation_stats.get("mean_self_min", 0.0)),
                    )

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if epoch == 0 and n_iter % step_size == 0 and n_iter <= warmup_iterations:
                try:
                    scheduler.step(n_iter // step_size)
                except Exception:
                    pass

            del loss_dict, loss

        logger.info("---------- epoch %s training complete ----------", epoch)

        with torch.no_grad():
            if epoch >= config.get("eval_epoch", 0) or args.evaluate or epoch == 0:
                best, best_epoch, best_log_epoch = evaluate_and_checkpoint(
                    model_without_ddp=model_without_ddp,
                    test_loader=test_loader,
                    device=device,
                    config=config,
                    args=args,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best=best,
                    best_epoch=best_epoch,
                    output_dir=args.output_dir,
                    logger=logger,
                )
                if best_log_epoch:
                    best_log = best_log_epoch

        if is_distributed:
            synchronize()

        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(best_log) + "\n")
