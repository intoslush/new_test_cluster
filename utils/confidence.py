from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F


def _get_model_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_cfg = config.get("MODEL", {})
    return model_cfg if isinstance(model_cfg, Mapping) else {}


def get_conf_calibration_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    model_cfg = _get_model_cfg(config)

    def pick(flat_key: str, model_key: str, default: Any) -> Any:
        if model_key in model_cfg:
            return model_cfg[model_key]
        return config.get(flat_key, default)

    return {
        "enabled": bool(pick("use_conf_calibration", "USE_CONF_CALIBRATION", False)),
        "alpha": float(pick("conf_alpha", "CONF_ALPHA", 1.0)),
        "beta": float(pick("conf_beta", "CONF_BETA", 1.0)),
        "temp": float(pick("conf_temp", "CONF_TEMP", 1.0)),
        "noise_value": float(pick("conf_noise_value", "CONF_NOISE_VALUE", 0.0)),
        "high_thres": float(pick("conf_high_thres", "CONF_HIGH_THRES", 0.7)),
        "low_thres": float(pick("conf_low_thres", "CONF_LOW_THRES", 0.3)),
        "weight_id_loss": bool(pick("conf_weight_id_loss", "CONF_WEIGHT_ID_LOSS", True)),
        "weight_metric_loss": bool(pick("conf_weight_metric_loss", "CONF_WEIGHT_METRIC_LOSS", False)),
        "use_text_prototype": bool(pick("conf_use_text_prototype", "CONF_USE_TEXT_PROTOTYPE", True)),
        "debug_log": bool(pick("conf_debug_log", "CONF_DEBUG_LOG", False)),
        "chunk_size": int(pick("conf_chunk_size", "CONF_CHUNK_SIZE", 4096)),
        "min_cluster_size": int(pick("conf_min_cluster_size", "CONF_MIN_CLUSTER_SIZE", 1)),
        "eps": float(pick("conf_eps", "CONF_EPS", 1e-12)),
        "text_key": str(pick("conf_text_key", "CONF_TEXT_KEY", config.get("cluster_text_key", "caption2"))),
    }


def normalize_weighted_mean(
    losses: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if losses.ndim != 1 or weights.ndim != 1:
        raise ValueError(f"normalize_weighted_mean expects 1D tensors, got {losses.shape} and {weights.shape}")
    if losses.shape[0] != weights.shape[0]:
        raise ValueError(f"length mismatch: losses={losses.shape[0]}, weights={weights.shape[0]}")
    weights = weights.to(device=losses.device, dtype=losses.dtype)
    return torch.sum(losses * weights) / (weights.sum() + eps)


@torch.no_grad()
def build_cluster_prototypes(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    *,
    min_cluster_size: int = 1,
) -> Dict[str, torch.Tensor]:
    if visual_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("visual/text features must be 2D tensors")
    if visual_features.shape != text_features.shape:
        raise ValueError(
            f"visual/text feature shape mismatch: {visual_features.shape} vs {text_features.shape}"
        )

    labels = pseudo_labels.to(device=visual_features.device, dtype=torch.long)
    valid_mask = labels >= 0
    unique_labels = torch.unique(labels[valid_mask], sorted=True)

    proto_labels = []
    visual_protos = []
    text_protos = []
    cluster_sizes = []

    for cluster_id in unique_labels.tolist():
        members = labels == int(cluster_id)
        cluster_size = int(members.sum().item())
        if cluster_size < min_cluster_size:
            continue

        visual_proto = F.normalize(visual_features[members].mean(dim=0), dim=0)
        text_proto = F.normalize(text_features[members].mean(dim=0), dim=0)

        proto_labels.append(int(cluster_id))
        visual_protos.append(visual_proto)
        text_protos.append(text_proto)
        cluster_sizes.append(cluster_size)

    if not proto_labels:
        feature_dim = visual_features.shape[1]
        empty_proto = visual_features.new_zeros((0, feature_dim))
        return {
            "cluster_labels": torch.empty((0,), dtype=torch.long, device=visual_features.device),
            "visual_prototypes": empty_proto,
            "text_prototypes": empty_proto.clone(),
            "cluster_sizes": torch.empty((0,), dtype=torch.long, device=visual_features.device),
        }

    return {
        "cluster_labels": torch.tensor(proto_labels, dtype=torch.long, device=visual_features.device),
        "visual_prototypes": torch.stack(visual_protos, dim=0),
        "text_prototypes": torch.stack(text_protos, dim=0),
        "cluster_sizes": torch.tensor(cluster_sizes, dtype=torch.long, device=visual_features.device),
    }


@torch.no_grad()
def compute_sample_confidence(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    temp: float,
    noise_value: float,
    high_thres: float,
    low_thres: float,
    use_text_prototype: bool = True,
    chunk_size: int = 4096,
    min_cluster_size: int = 1,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    device = visual_features.device
    labels = pseudo_labels.to(device=device, dtype=torch.long)
    visual_features = F.normalize(visual_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    proto_info = build_cluster_prototypes(
        visual_features,
        text_features,
        labels,
        min_cluster_size=min_cluster_size,
    )
    cluster_labels = proto_info["cluster_labels"]
    visual_prototypes = proto_info["visual_prototypes"]
    text_prototypes = proto_info["text_prototypes"]

    confidences = torch.full(
        (labels.shape[0],),
        float(noise_value),
        dtype=visual_features.dtype,
        device=device,
    )
    groups = torch.zeros((labels.shape[0],), dtype=torch.long, device=device)
    visual_margin = torch.zeros_like(confidences)
    cross_modal_margin = torch.zeros_like(confidences)

    if cluster_labels.numel() == 0:
        return {
            "confidence": confidences,
            "group": groups,
            "visual_margin": visual_margin,
            "cross_modal_margin": cross_modal_margin,
            "cluster_labels": cluster_labels,
            "cluster_sizes": proto_info["cluster_sizes"],
        }

    valid_mask = labels >= 0
    assigned_mask = valid_mask.clone()
    if cluster_labels.numel() < 2:
        if cluster_labels.numel() == 1:
            assigned_mask = labels == int(cluster_labels[0].item())
            confidences[assigned_mask] = 1.0
    else:
        max_label = int(torch.max(torch.cat([cluster_labels, labels[valid_mask]], dim=0)).item())
        label_to_proto = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        label_to_proto[cluster_labels] = torch.arange(cluster_labels.numel(), device=device)
        assigned_proto = torch.full_like(labels, -1)
        assigned_proto[valid_mask] = label_to_proto[labels[valid_mask]]
        assigned_mask = assigned_proto >= 0

        for start in range(0, labels.shape[0], max(1, int(chunk_size))):
            end = min(start + max(1, int(chunk_size)), labels.shape[0])
            chunk_mask = assigned_mask[start:end]
            if not torch.any(chunk_mask):
                continue

            v_chunk = visual_features[start:end][chunk_mask]
            proto_chunk = assigned_proto[start:end][chunk_mask]

            sim_visual = torch.matmul(v_chunk, visual_prototypes.t())
            assigned_visual = sim_visual.gather(1, proto_chunk.unsqueeze(1)).squeeze(1)
            sim_visual.scatter_(1, proto_chunk.unsqueeze(1), float("-inf"))
            other_visual = sim_visual.max(dim=1).values
            visual_margin[start:end][chunk_mask] = assigned_visual - other_visual

            if use_text_prototype:
                sim_text = torch.matmul(v_chunk, text_prototypes.t())
                assigned_text = sim_text.gather(1, proto_chunk.unsqueeze(1)).squeeze(1)
                sim_text.scatter_(1, proto_chunk.unsqueeze(1), float("-inf"))
                other_text = sim_text.max(dim=1).values
                cross_modal_margin[start:end][chunk_mask] = assigned_text - other_text

        logits = alpha * visual_margin + beta * cross_modal_margin
        confidences[assigned_mask] = torch.sigmoid(logits[assigned_mask] / max(float(temp), eps))

    groups = torch.where(
        confidences >= high_thres,
        torch.full_like(groups, 2),
        torch.where(confidences > low_thres, torch.full_like(groups, 1), torch.zeros_like(groups)),
    )

    return {
        "confidence": confidences,
        "group": groups,
        "visual_margin": visual_margin,
        "cross_modal_margin": cross_modal_margin,
        "cluster_labels": cluster_labels,
        "cluster_sizes": proto_info["cluster_sizes"],
    }
