import json
import os

import torch

try:
    from .eval import itm_eval, evaluation
except Exception:
    from .eval import itm_eval, evaluation  # type: ignore


def evaluate_and_checkpoint(
    *,
    model_without_ddp,
    test_loader,
    device,
    config,
    args,
    optimizer,
    scheduler,
    epoch: int,
    best: float,
    best_epoch: int,
    output_dir: str,
    logger,
):
    """Run evaluation, update the best checkpoint, and return summary info."""
    best_log = {}

    score_test_t2i = evaluation(
        model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config, args
    )
    test_result = itm_eval(
        score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, args.eval_mAP
    )

    logger.info("Test result: %s", test_result)

    log_stats = {"epoch": epoch, **{f"test_{k}": v for k, v in test_result.items()}}
    with open(os.path.join(output_dir, "log.txt"), "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_stats) + "\n")

    if test_result.get("r1", -float("inf")) > best:
        best = test_result["r1"]
        best_epoch = epoch
        best_log = log_stats
        save_obj = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
        }
        os.makedirs(os.path.join(output_dir, "checkpoint"), exist_ok=True)
        torch.save(save_obj, os.path.join(output_dir, "checkpoint", "checkpoint_best.pth"))

    return best, best_epoch, best_log
