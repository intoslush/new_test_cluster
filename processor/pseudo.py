import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional
from collections import defaultdict
from utils.confidence import compute_sample_confidence, get_conf_calibration_cfg

try:
    from .cluster import cluster_begin_epoch
except Exception:
    # 兼容直接放在同级目录的情形
    from .cluster import cluster_begin_epoch  # type: ignore


def _bucketize_confidence(
    confidence: torch.Tensor,
    *,
    high_thres: float,
    low_thres: float,
) -> torch.Tensor:
    return torch.where(
        confidence >= high_thres,
        torch.full_like(confidence, 2, dtype=torch.long),
        torch.where(confidence > low_thres, torch.full_like(confidence, 1, dtype=torch.long), torch.zeros_like(confidence, dtype=torch.long)),
    )


def _default_confidence_tensor(
    *,
    dataset_size: int,
    device: torch.device,
    raw_pseudo_np: Optional[np.ndarray],
    noise_value: float,
    high_thres: float,
    low_thres: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    confidence = torch.ones((dataset_size,), dtype=torch.float32, device=device)
    if raw_pseudo_np is not None:
        raw_tensor = torch.tensor(raw_pseudo_np, dtype=torch.long, device=device)
        confidence = torch.where(raw_tensor == -1, torch.full_like(confidence, float(noise_value)), confidence)
    groups = _bucketize_confidence(confidence, high_thres=high_thres, low_thres=low_thres)
    return confidence, groups


def _full_dataset_size(dataset) -> int:
    if hasattr(dataset, "pairs"):
        return len(dataset.pairs)
    return len(dataset)

def _replace_noise_with_unique_ids(pseudo_np: np.ndarray) -> np.ndarray:
    """
    将 pseudo_np 中的 -1（噪声）替换为每个样本独立的 id。
    新 id 从 (max_non_noise_id + 1) 开始递增，保证不与已有簇 id 冲突。
    """
    pseudo_np = pseudo_np.astype(np.int64, copy=True)
    noise_mask = (pseudo_np == -1)
    if not noise_mask.any():
        return pseudo_np

    non_noise = pseudo_np[~noise_mask]
    start_id = int(non_noise.max() + 1) if non_noise.size > 0 else 0

    noise_indices = np.where(noise_mask)[0]
    pseudo_np[noise_indices] = np.arange(start_id, start_id + noise_indices.size, dtype=np.int64)
    return pseudo_np


def apply_cluster_id_mode(
    *,
    mode: str,
    dataset_size: int,
    raw_pseudo_np: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    返回: (final_pseudo_np, raw_pseudo_np_for_metrics)
    - final_pseudo_np: 实际用于训练/广播的伪标签
    - raw_pseudo_np_for_metrics: 用于评估监控的原始输出（含-1），instance 模式下为 None
    """
    mode = (mode or "cluster").lower()

    if mode == "instance":
        final_np = np.arange(dataset_size, dtype=np.int64)
        return final_np, None

    if raw_pseudo_np is None:
        raise ValueError(f"mode={mode} 需要 raw_pseudo_np，但传入为 None")

    if mode == "cluster":
        return raw_pseudo_np.astype(np.int64, copy=False), raw_pseudo_np

    if mode == "unique_noise":
        final_np = _replace_noise_with_unique_ids(raw_pseudo_np)
        return final_np, raw_pseudo_np

    raise ValueError(
        f"未知 cluster_id_mode={mode}，支持：cluster / instance / unique_noise"
    )

def compute_pseudo_stats(pseudo_labels: np.ndarray, true_person_ids: np.ndarray):
    """
    pseudo_labels: [N]，聚类结果，含 -1（噪声）
    true_person_ids: [N]，真实 person 索引
    返回：覆盖率、正确数、纯度等
    """
    assert len(pseudo_labels) == len(true_person_ids), (
        f"长度不一致: pseudo={len(pseudo_labels)}, gt={len(true_person_ids)}"
    )

    mask = (pseudo_labels != -1)
    num_total = len(pseudo_labels)
    num_assigned = int(mask.sum())
    num_noise = num_total - num_assigned

    # 按簇聚合 -> 多数表决
    correct = 0
    if num_assigned > 0:
        cluster2idx = defaultdict(list)
        for i, cid in enumerate(pseudo_labels):
            if cid != -1:
                cluster2idx[cid].append(i)
        for _, idxs in cluster2idx.items():
            gt = true_person_ids[idxs]
            values, counts = np.unique(gt, return_counts=True)
            correct += int(counts.max())

    return {
        "num_total": num_total,
        "num_assigned": num_assigned,
        "num_noise": num_noise,
        "coverage": num_assigned / num_total if num_total else 0.0,
        "correct": correct,
        "purity_non_noise": correct / num_assigned if num_assigned else 0.0,
        "purity_overall": correct / num_total if num_total else 0.0,
    }


def generate_and_broadcast_pseudo_labels(
    *,
    epoch: int,
    device: torch.device,
    is_main: bool,
    is_distributed: bool,
    rank: int,
    cluster_loader,
    model,
    args,
    config,
    logger,
    tb_writer=None,
    enable_nmi_ari: bool = True,
    cluster_until_epoch: int = 40,
):
    """
    仅在 epoch < cluster_until_epoch 时进行聚类，并将结果广播到所有 rank。
    返回：dict，包含 pseudo_labels / sample_confidence / confidence_group
    """
    conf_cfg = get_conf_calibration_cfg(config)
    cluster_loader.dataset.mode = "cluster"
    dataset_size = _full_dataset_size(cluster_loader.dataset)
    if epoch >= cluster_until_epoch:
        # 直接创建占位（-1）张量
        pseudo_labels = torch.full((dataset_size,), -1, dtype=torch.long, device=device)
        sample_confidence, confidence_group = _default_confidence_tensor(
            dataset_size=dataset_size,
            device=device,
            raw_pseudo_np=np.full((dataset_size,), -1, dtype=np.int64),
            noise_value=float(conf_cfg["noise_value"]),
            high_thres=float(conf_cfg["high_thres"]),
            low_thres=float(conf_cfg["low_thres"]),
        )
        return {
            "pseudo_labels": pseudo_labels,
            "sample_confidence": sample_confidence,
            "confidence_group": confidence_group,
        }

    # 新增：从 args 获取聚类 id 策略
    cluster_id_mode = getattr(args, "cluster_id_mode", "cluster")
    # cluster_id_mode="cluster"

    if is_main:
        with torch.no_grad():
            raw_pseudo_np: Optional[np.ndarray] = None
            confidence_feature_cache = None

            if str(cluster_id_mode).lower() == "instance":
                # 不聚类：每个样本独立 id
                final_pseudo_np, raw_for_metrics = apply_cluster_id_mode(
                    mode=cluster_id_mode,
                    dataset_size=dataset_size,
                    raw_pseudo_np=None,
                )
                # instance 模式下无 raw_for_metrics
            else:
                # 正常聚类
                cluster_result = cluster_begin_epoch(
                    cluster_loader, model, args, config, None, logger
                )
                if isinstance(cluster_result, dict):
                    raw_pseudo_np = cluster_result.get("pseudo_labels")
                    confidence_feature_cache = cluster_result.get("feature_cache")
                else:
                    raw_pseudo_np = cluster_result
                final_pseudo_np, raw_for_metrics = apply_cluster_id_mode(
                    mode=cluster_id_mode,
                    dataset_size=dataset_size,
                    raw_pseudo_np=raw_pseudo_np,
                )

            if len(final_pseudo_np) != dataset_size:
                raise ValueError(
                    f"伪标签长度与数据集长度不一致: pseudo={len(final_pseudo_np)}, dataset={dataset_size}"
                )
            if raw_for_metrics is not None and len(raw_for_metrics) != dataset_size:
                raise ValueError(
                    f"原始聚类结果长度与数据集长度不一致: raw={len(raw_for_metrics)}, dataset={dataset_size}"
                )

            # 统计簇数：建议用 raw（含 -1），避免 unique_noise 把噪声当成大量新簇
            if raw_for_metrics is not None:
                image_num_cluster = len(set(raw_for_metrics)) - (
                    1 if -1 in raw_for_metrics else 0
                )
                noise_frac = float((raw_for_metrics == -1).mean())
            else:
                # instance：每个样本一个簇
                image_num_cluster = dataset_size
                noise_frac = 0.0

            logger.info(
                "==> [epoch %d] cluster_id_mode=%s, clusters=%d, total=%d",
                epoch, str(cluster_id_mode), image_num_cluster, dataset_size
            )

            # 用 final_pseudo_np 作为实际返回/训练用伪标签
            image_pseudo_labels = torch.tensor(
                final_pseudo_np, dtype=torch.long
            ).to(device, non_blocking=True)
            sample_confidence, confidence_group = _default_confidence_tensor(
                dataset_size=dataset_size,
                device=device,
                raw_pseudo_np=raw_for_metrics,
                noise_value=float(conf_cfg["noise_value"]),
                high_thres=float(conf_cfg["high_thres"]),
                low_thres=float(conf_cfg["low_thres"]),
            )

            if bool(conf_cfg["enabled"]):
                can_compute_conf = (
                    raw_for_metrics is not None
                    and confidence_feature_cache is not None
                    and confidence_feature_cache.get("visual_features") is not None
                    and confidence_feature_cache.get("text_features") is not None
                    and len(confidence_feature_cache["visual_features"]) == dataset_size
                    and len(confidence_feature_cache["text_features"]) == dataset_size
                )
                if can_compute_conf:
                    confidence_result = compute_sample_confidence(
                        visual_features=confidence_feature_cache["visual_features"].to(device, non_blocking=True),
                        text_features=confidence_feature_cache["text_features"].to(device, non_blocking=True),
                        pseudo_labels=torch.tensor(raw_for_metrics, dtype=torch.long, device=device),
                        alpha=float(conf_cfg["alpha"]),
                        beta=float(conf_cfg["beta"]),
                        temp=float(conf_cfg["temp"]),
                        noise_value=float(conf_cfg["noise_value"]),
                        high_thres=float(conf_cfg["high_thres"]),
                        low_thres=float(conf_cfg["low_thres"]),
                        use_text_prototype=bool(conf_cfg["use_text_prototype"]),
                        chunk_size=int(conf_cfg["chunk_size"]),
                        min_cluster_size=int(conf_cfg["min_cluster_size"]),
                        eps=float(conf_cfg["eps"]),
                    )
                    sample_confidence = confidence_result["confidence"].to(device, non_blocking=True)
                    confidence_group = confidence_result["group"].to(device, non_blocking=True)

                    if bool(conf_cfg["debug_log"]):
                        logger.info(
                            "[ConfCalib][epoch %d] mean=%.4f, min=%.4f, max=%.4f, core=%d, boundary=%d, unreliable=%d",
                            epoch,
                            float(sample_confidence.mean().item()),
                            float(sample_confidence.min().item()),
                            float(sample_confidence.max().item()),
                            int((confidence_group == 2).sum().item()),
                            int((confidence_group == 1).sum().item()),
                            int((confidence_group == 0).sum().item()),
                        )
                else:
                    logger.warning(
                        "[ConfCalib][epoch %d] 跳过 confidence 计算：缺少原始聚类结果或 visual/text feature cache，回退为默认权重。",
                        epoch,
                    )

            # 可选：伪标签质量监控（建议用 raw_for_metrics）
            if hasattr(cluster_loader.dataset, 'pairs'):
                gt_persons = np.array([p for _, _, p in cluster_loader.dataset.pairs], dtype=np.int64)

                # 注意：instance 模式下 raw_for_metrics=None，此时“聚类质量”指标意义不同，
                # 这里选择跳过（你也可以改为对 final_pseudo_np 计算，但指标会退化且难解释）
                if raw_for_metrics is not None:
                    pseudo_np_metrics = raw_for_metrics.astype(np.int64, copy=False)
                    stats = compute_pseudo_stats(pseudo_np_metrics, gt_persons)

                    logger.info(
                        "[ClusterEval][epoch %d] assigned=%d/%d (coverage=%.4f), noise=%d, "
                        "correct=%d, purity_non_noise=%.4f, overall=%.4f",
                        epoch, stats["num_assigned"], stats["num_total"], stats["coverage"],
                        stats["num_noise"], stats["correct"], stats["purity_non_noise"], stats["purity_overall"],
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("Cluster/coverage", stats["coverage"], epoch)
                        tb_writer.add_scalar("Cluster/correct_count", stats["correct"], epoch)
                        tb_writer.add_scalar("Cluster/purity_non_noise", stats["purity_non_noise"], epoch)
                        tb_writer.add_scalar("Cluster/purity_overall", stats["purity_overall"], epoch)
                        tb_writer.add_scalar("Cluster/num_clusters", image_num_cluster, epoch)
                        tb_writer.add_scalar("Cluster/noise_frac", noise_frac, epoch)

                    if enable_nmi_ari:
                        try:
                            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
                            m = (pseudo_np_metrics != -1)
                            if m.any():
                                nmi = normalized_mutual_info_score(gt_persons[m], pseudo_np_metrics[m])
                                ari = adjusted_rand_score(gt_persons[m], pseudo_np_metrics[m])
                                logger.info("[ClusterEval][epoch %d] NMI=%.4f, ARI=%.4f", epoch, nmi, ari)
                                if tb_writer is not None:
                                    tb_writer.add_scalar("Cluster/NMI_non_noise", nmi, epoch)
                                    tb_writer.add_scalar("Cluster/ARI_non_noise", ari, epoch)
                        except Exception as e:
                            logger.warning(f"计算 NMI/ARI 失败：{e}")
                else:
                    logger.info(
                        "[ClusterEval][epoch %d] cluster_id_mode=instance，跳过聚类质量评估指标（无 raw clustering 输出）",
                        epoch
                    )

            # 提前释放 numpy 内存
            if raw_pseudo_np is not None:
                del raw_pseudo_np
            del final_pseudo_np
    else:
        print(f"[Rank {rank}] 等待主进程生成伪标签")
        image_pseudo_labels = torch.empty(dataset_size, dtype=torch.long, device=device)
        sample_confidence = torch.empty(dataset_size, dtype=torch.float32, device=device)
        confidence_group = torch.empty(dataset_size, dtype=torch.long, device=device)

    if is_distributed:
        dist.broadcast(image_pseudo_labels, src=0)
        dist.broadcast(sample_confidence, src=0)
        dist.broadcast(confidence_group, src=0)
        dist.barrier()

    # 还原为 train 模式交由上层设置
    cluster_loader.dataset.mode = 'cluster'

    return {
        "pseudo_labels": image_pseudo_labels,
        "sample_confidence": sample_confidence,
        "confidence_group": confidence_group,
    }
