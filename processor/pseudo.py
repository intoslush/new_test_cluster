import csv
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


_CLUSTER_METRIC_FIELDNAMES = [
    "run_name",
    "config_path",
    "output_dir",
    "train_cluster_mode",
    "cluster_id_mode",
    "epoch",
    "cluster_round",
    "num_samples",
    "gt_num_classes",
    "image_num_clusters",
    "image_num_assigned",
    "image_num_noise",
    "image_coverage",
    "image_nmi_non_noise",
    "image_ari_non_noise",
    "image_assignment_shift",
    "text_num_clusters",
    "text_num_assigned",
    "text_num_noise",
    "text_coverage",
    "text_nmi_non_noise",
    "text_ari_non_noise",
    "text_assignment_shift",
]


def _cluster_metrics_paths():
    os.makedirs("tmp", exist_ok=True)
    return {
        "csv": os.path.join("tmp", "cluster_metrics.csv"),
        "readme": os.path.join("tmp", "cluster_metrics_readme.md"),
        "state": os.path.join("tmp", "cluster_metrics_state.pt"),
    }


def _write_cluster_metrics_readme(path: str):
    readme = """# Cluster Metrics

本文件说明 `tmp/cluster_metrics.csv` 中各字段的含义与计算口径，方便后续直接画图。

## 基本口径

- 每一行对应一次真正执行的聚类事件。
- 所有统计都基于训练集 `pairs` 粒度，即 `(image, caption, person_idx)` 样本，而不是去重后的 image 粒度。
- `image_*` 表示使用图像特征聚类得到的结果。
- `text_*` 表示使用文本特征聚类得到的结果。
- `gt_num_classes` 表示当前训练集真实 `person_idx` 的类别数。
- `*_num_clusters` 不包含噪声标签 `-1`。
- `*_num_assigned` 表示非噪声样本数。
- `*_num_noise` 表示被 DBSCAN 判为 `-1` 的样本数。
- `*_coverage = *_num_assigned / num_samples`。
- `*_nmi_non_noise` 与 `*_ari_non_noise` 仅在非噪声样本子集上计算，即先过滤掉当前模态聚类结果中的 `-1`。

## Assignment Shift 定义

- `*_assignment_shift` 用来衡量“同一模态当前轮相对上一轮”的分配变化率。
- 先把当前轮每个非噪声簇，映射到上一轮中与它样本重叠数最多的簇 id，目的是消除 DBSCAN 纯粹重编号带来的假变化。
- 如果某个当前簇与上一轮所有非噪声簇都没有重叠，则给它分配一个新的临时 id。
- 映射完成后，逐样本比较“上一轮标签”和“当前轮对齐后的标签”是否一致。
- 分母使用全部样本数 `num_samples`。
- `noise -> non-noise`、`non-noise -> noise` 都算作变化；`noise -> noise` 不算变化。
- 第一轮聚类没有上一轮可比，因此 `*_assignment_shift` 会记为 `NaN`。

## 字段说明

- `run_name`: 当前实验名，对应 `args.name`。
- `config_path`: 配置文件绝对路径。
- `output_dir`: 当前训练输出目录绝对路径。
- `train_cluster_mode`: 真正用于训练伪标签的聚类模态，取自 `config['cluster_feature_mode']`。
- `cluster_id_mode`: 训练伪标签 id 策略，取自 `args.cluster_id_mode`。
- `epoch`: 当前 epoch。
- `cluster_round`: 当前运行内第几次触发聚类。
- `num_samples`: 当前训练集 pair 总数。
- `gt_num_classes`: 真实 person 类别数。
- `image_num_clusters`, `text_num_clusters`: 各模态非噪声簇数。
- `image_num_assigned`, `text_num_assigned`: 各模态非噪声样本数。
- `image_num_noise`, `text_num_noise`: 各模态噪声样本数。
- `image_coverage`, `text_coverage`: 各模态非噪声覆盖率。
- `image_nmi_non_noise`, `text_nmi_non_noise`: 各模态非噪声 NMI。
- `image_ari_non_noise`, `text_ari_non_noise`: 各模态非噪声 ARI。
- `image_assignment_shift`, `text_assignment_shift`: 各模态跨轮分配变动率。
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(readme)


def _build_monitor_run_signature(*, args, config, dataset_size: int):
    return {
        "run_name": str(getattr(args, "name", "")),
        "config_path": os.path.abspath(str(getattr(args, "config", ""))),
        "output_dir": os.path.abspath(str(getattr(args, "output_dir", ""))),
        "train_cluster_mode": str(config.get("cluster_feature_mode", "image")).lower(),
        "cluster_id_mode": str(getattr(args, "cluster_id_mode", "cluster")).lower(),
        "dataset_size": int(dataset_size),
    }


def _load_cluster_monitor_state(path: str, signature: dict):
    default_state = {
        "signature": signature,
        "cluster_round": 0,
        "prev_labels": {},
    }
    if not os.path.exists(path):
        return default_state

    try:
        state = torch.load(path, weights_only=False)
    except Exception:
        return default_state

    if not isinstance(state, dict) or state.get("signature") != signature:
        return default_state

    prev_labels = {}
    raw_prev_labels = state.get("prev_labels", {})
    if isinstance(raw_prev_labels, dict):
        for mode, labels in raw_prev_labels.items():
            if labels is None:
                continue
            prev_labels[str(mode)] = np.asarray(labels, dtype=np.int64)

    return {
        "signature": signature,
        "cluster_round": int(state.get("cluster_round", 0)),
        "prev_labels": prev_labels,
    }


def _save_cluster_monitor_state(path: str, state: dict):
    torch.save(state, path)


def _append_cluster_metrics_row(path: str, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CLUSTER_METRIC_FIELDNAMES)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def _count_non_noise_clusters(labels: np.ndarray) -> int:
    unique_labels = np.unique(labels)
    return int(unique_labels.size - np.count_nonzero(unique_labels == -1))


def _align_current_labels_to_previous(prev_labels: np.ndarray, curr_labels: np.ndarray) -> np.ndarray:
    aligned = np.full(curr_labels.shape, -1, dtype=np.int64)

    overlap_by_curr = defaultdict(lambda: defaultdict(int))
    valid_mask = (prev_labels != -1) & (curr_labels != -1)
    for prev_id, curr_id in zip(prev_labels[valid_mask], curr_labels[valid_mask]):
        overlap_by_curr[int(curr_id)][int(prev_id)] += 1

    prev_non_noise = prev_labels[prev_labels != -1]
    next_new_id = int(prev_non_noise.max() + 1) if prev_non_noise.size > 0 else 0

    for curr_id in np.unique(curr_labels[curr_labels != -1]):
        prev_overlap = overlap_by_curr.get(int(curr_id), {})
        if prev_overlap:
            mapped_prev_id = max(
                prev_overlap.items(),
                key=lambda item: (item[1], -item[0]),
            )[0]
        else:
            mapped_prev_id = next_new_id
            next_new_id += 1
        aligned[curr_labels == curr_id] = int(mapped_prev_id)

    return aligned


def _compute_assignment_shift(prev_labels: Optional[np.ndarray], curr_labels: np.ndarray) -> float:
    if prev_labels is None:
        return float("nan")
    if len(prev_labels) != len(curr_labels):
        raise ValueError(
            f"Assignment Shift 长度不一致: prev={len(prev_labels)}, curr={len(curr_labels)}"
        )

    prev_np = np.asarray(prev_labels, dtype=np.int64)
    curr_np = np.asarray(curr_labels, dtype=np.int64)
    aligned_curr = _align_current_labels_to_previous(prev_np, curr_np)
    return float(np.mean(aligned_curr != prev_np))


def _compute_cluster_eval_metrics(
    *,
    pseudo_labels: np.ndarray,
    true_person_ids: np.ndarray,
    enable_nmi_ari: bool,
    logger,
    epoch: int,
    mode: str,
):
    metrics = compute_pseudo_stats(pseudo_labels, true_person_ids)
    metrics["num_clusters"] = _count_non_noise_clusters(pseudo_labels)
    metrics["nmi_non_noise"] = float("nan")
    metrics["ari_non_noise"] = float("nan")

    logger.info(
        "[ClusterEval][epoch %d][%s] assigned=%d/%d (coverage=%.4f), noise=%d, "
        "correct=%d, purity_non_noise=%.4f, overall=%.4f, clusters=%d",
        epoch,
        mode,
        metrics["num_assigned"],
        metrics["num_total"],
        metrics["coverage"],
        metrics["num_noise"],
        metrics["correct"],
        metrics["purity_non_noise"],
        metrics["purity_overall"],
        metrics["num_clusters"],
    )

    if enable_nmi_ari:
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            valid_mask = (pseudo_labels != -1)
            if valid_mask.any():
                metrics["nmi_non_noise"] = float(
                    normalized_mutual_info_score(true_person_ids[valid_mask], pseudo_labels[valid_mask])
                )
                metrics["ari_non_noise"] = float(
                    adjusted_rand_score(true_person_ids[valid_mask], pseudo_labels[valid_mask])
                )
                logger.info(
                    "[ClusterEval][epoch %d][%s] NMI=%.4f, ARI=%.4f",
                    epoch,
                    mode,
                    metrics["nmi_non_noise"],
                    metrics["ari_non_noise"],
                )
        except Exception as exc:
            logger.warning("[ClusterEval][epoch %d][%s] 计算 NMI/ARI 失败：%s", epoch, mode, exc)

    return metrics


def _record_cluster_metrics(
    *,
    epoch: int,
    args,
    config,
    dataset_size: int,
    gt_persons: np.ndarray,
    monitor_labels: dict,
    enable_nmi_ari: bool,
    logger,
):
    paths = _cluster_metrics_paths()
    _write_cluster_metrics_readme(paths["readme"])

    signature = _build_monitor_run_signature(args=args, config=config, dataset_size=dataset_size)
    if int(epoch) == 0:
        state = {
            "signature": signature,
            "cluster_round": 0,
            "prev_labels": {},
        }
    else:
        state = _load_cluster_monitor_state(paths["state"], signature)
    cluster_round = int(state["cluster_round"]) + 1

    row = {
        "run_name": signature["run_name"],
        "config_path": signature["config_path"],
        "output_dir": signature["output_dir"],
        "train_cluster_mode": signature["train_cluster_mode"],
        "cluster_id_mode": signature["cluster_id_mode"],
        "epoch": int(epoch),
        "cluster_round": cluster_round,
        "num_samples": int(dataset_size),
        "gt_num_classes": int(np.unique(gt_persons).size),
    }

    prev_labels_state = state.get("prev_labels", {})
    next_prev_labels = {}
    for mode in ("image", "text"):
        mode_labels = monitor_labels.get(mode)
        if mode_labels is None:
            row[f"{mode}_num_clusters"] = ""
            row[f"{mode}_num_assigned"] = ""
            row[f"{mode}_num_noise"] = ""
            row[f"{mode}_coverage"] = ""
            row[f"{mode}_nmi_non_noise"] = ""
            row[f"{mode}_ari_non_noise"] = ""
            row[f"{mode}_assignment_shift"] = ""
            continue

        mode_np = np.asarray(mode_labels, dtype=np.int64)
        metrics = _compute_cluster_eval_metrics(
            pseudo_labels=mode_np,
            true_person_ids=gt_persons,
            enable_nmi_ari=enable_nmi_ari,
            logger=logger,
            epoch=epoch,
            mode=mode,
        )
        assignment_shift = _compute_assignment_shift(prev_labels_state.get(mode), mode_np)
        logger.info("[ClusterEval][epoch %d][%s] Assignment Shift=%.4f", epoch, mode, assignment_shift)

        row[f"{mode}_num_clusters"] = metrics["num_clusters"]
        row[f"{mode}_num_assigned"] = metrics["num_assigned"]
        row[f"{mode}_num_noise"] = metrics["num_noise"]
        row[f"{mode}_coverage"] = metrics["coverage"]
        row[f"{mode}_nmi_non_noise"] = metrics["nmi_non_noise"]
        row[f"{mode}_ari_non_noise"] = metrics["ari_non_noise"]
        row[f"{mode}_assignment_shift"] = assignment_shift
        next_prev_labels[mode] = mode_np.copy()

    _append_cluster_metrics_row(paths["csv"], row)
    _save_cluster_monitor_state(
        paths["state"],
        {
            "signature": signature,
            "cluster_round": cluster_round,
            "prev_labels": next_prev_labels,
        },
    )
    logger.info(
        "[ClusterEval][epoch %d] 聚类指标已追加写入 %s，字段说明见 %s",
        epoch,
        paths["csv"],
        paths["readme"],
    )


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
            monitor_labels = {}
            train_cluster_mode = str(config.get("cluster_feature_mode", "image")).lower()

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
                    monitor_labels = cluster_result.get("monitor_labels") or {}
                    train_cluster_mode = str(cluster_result.get("train_mode", train_cluster_mode)).lower()
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
                train_num_cluster = len(set(raw_for_metrics)) - (
                    1 if -1 in raw_for_metrics else 0
                )
            else:
                # instance：每个样本一个簇
                train_num_cluster = dataset_size

            logger.info(
                "==> [epoch %d] cluster_id_mode=%s, train_cluster_mode=%s, clusters=%d, total=%d",
                epoch, str(cluster_id_mode), train_cluster_mode, train_num_cluster, dataset_size
            )

            # 用 final_pseudo_np 作为实际返回/训练用伪标签
            train_pseudo_labels = torch.tensor(
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

            if hasattr(cluster_loader.dataset, 'pairs'):
                gt_persons = np.array([p for _, _, p in cluster_loader.dataset.pairs], dtype=np.int64)
                if raw_for_metrics is not None:
                    if train_cluster_mode not in monitor_labels:
                        monitor_labels[train_cluster_mode] = raw_for_metrics.astype(np.int64, copy=False)
                    _record_cluster_metrics(
                        epoch=epoch,
                        args=args,
                        config=config,
                        dataset_size=dataset_size,
                        gt_persons=gt_persons,
                        monitor_labels=monitor_labels,
                        enable_nmi_ari=enable_nmi_ari,
                        logger=logger,
                    )
                else:
                    logger.info(
                        "[ClusterEval][epoch %d] cluster_id_mode=instance，跳过 image/text 聚类监控（无 raw clustering 输出）",
                        epoch
                    )

            # 提前释放 numpy 内存
            if raw_pseudo_np is not None:
                del raw_pseudo_np
            del final_pseudo_np
    else:
        print(f"[Rank {rank}] 等待主进程生成伪标签")
        train_pseudo_labels = torch.empty(dataset_size, dtype=torch.long, device=device)
        sample_confidence = torch.empty(dataset_size, dtype=torch.float32, device=device)
        confidence_group = torch.empty(dataset_size, dtype=torch.long, device=device)

    if is_distributed:
        dist.broadcast(train_pseudo_labels, src=0)
        dist.broadcast(sample_confidence, src=0)
        dist.broadcast(confidence_group, src=0)
        dist.barrier()

    # 还原为 train 模式交由上层设置
    cluster_loader.dataset.mode = 'cluster'

    return {
        "pseudo_labels": train_pseudo_labels,
        "sample_confidence": sample_confidence,
        "confidence_group": confidence_group,
    }
