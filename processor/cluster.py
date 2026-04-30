import os
import gc
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import DBSCAN  # 可选：如果内存允许、想用更快的 C 实现，可以开启下面的 sklearn 分支
from utils.confidence import get_conf_calibration_cfg

@torch.no_grad()
def _infer_cluster_feat_dim(model_no_ddp, config, mode: str) -> int:
    mode = mode.lower()
    if mode == "image":
        return int(model_no_ddp.vision_proj.out_features)  # embed_dim
    elif mode == "text":
        use_proj = bool(config.get("cluster_text_use_proj", False))
        if use_proj:
            return int(model_no_ddp.text_proj.out_features)  # embed_dim
        return int(model_no_ddp.text_encoder.config.hidden_size)  # text_width
    elif mode == "fusion":
        use_proj = bool(config.get("cluster_fusion_use_proj", False))
        if use_proj:
            return int(model_no_ddp.text_proj.out_features)  # embed_dim
        return int(model_no_ddp.text_encoder.config.hidden_size)  # text_width
    else:
        raise ValueError(f"Unknown cluster_feature_mode: {mode}")


@torch.no_grad()
def _tokenize_texts(texts, tokenizer, config, device: torch.device):
    return tokenizer(
        texts,
        padding="longest",
        max_length=int(config["max_words"]),
        return_tensors="pt",
        truncation=True,
    ).to(device)


@torch.no_grad()
def _encode_text_hidden(model_no_ddp, text_inputs):
    return model_no_ddp.text_encoder.bert(
        text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        return_dict=True,
        mode="text",
    )


@torch.no_grad()
def extract_cluster_feature_bundle(
    batch,
    model_no_ddp,
    tokenizer,
    config,
    device: torch.device,
    *,
    need_modes,
    confidence_text_key=None,
):
    """
    一次 batch 前向同时提取多路聚类特征，避免 image / text 监控重复跑编码器。
    返回:
      {
        "cluster_features": {"image": ..., "text": ..., "fusion": ...},
        "confidence_features": {"visual_features": ..., "text_features": ...} | None,
      }
    """
    normalized_modes = {str(mode).lower() for mode in need_modes}
    need_text_branch = bool({"text", "fusion"} & normalized_modes)
    if tokenizer is None and (need_text_branch or confidence_text_key is not None):
        raise ValueError("tokenizer is required for text/fusion clustering or confidence calibration.")

    cluster_features = {}
    cluster_text_key = str(config.get("cluster_text_key", "caption2"))

    need_image_branch = ("image" in normalized_modes) or ("fusion" in normalized_modes) or (confidence_text_key is not None)
    image_embeds = None
    image_proj = None
    if need_image_branch:
        image = batch["image1"].to(device, non_blocking=True)
        image_embeds = model_no_ddp.visual_encoder(image)
        if ("image" in normalized_modes) or (confidence_text_key is not None):
            image_proj = F.normalize(model_no_ddp.vision_proj(image_embeds[:, 0, :]), dim=-1)
        if "image" in normalized_modes:
            cluster_features["image"] = image_proj

    cluster_text_inputs = None
    cluster_text_out = None
    cluster_text_cls = None
    if need_text_branch:
        cluster_text_inputs = _tokenize_texts(batch[cluster_text_key], tokenizer, config, device)
        cluster_text_out = _encode_text_hidden(model_no_ddp, cluster_text_inputs)
        cluster_text_cls = cluster_text_out.last_hidden_state[:, 0, :]

        if "text" in normalized_modes:
            text_feat = cluster_text_cls
            if bool(config.get("cluster_text_use_proj", False)):
                text_feat = model_no_ddp.text_proj(text_feat)
            cluster_features["text"] = F.normalize(text_feat, dim=-1)

        if "fusion" in normalized_modes:
            if image_embeds is None:
                raise ValueError("image features are required when cluster_feature_mode='fusion'.")
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            fusion_out = model_no_ddp.text_encoder.bert(
                encoder_embeds=cluster_text_out.last_hidden_state,
                attention_mask=cluster_text_inputs["attention_mask"],
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode="fusion",
            )
            fusion_cls = fusion_out.last_hidden_state[:, 0, :]
            if bool(config.get("cluster_fusion_use_proj", False)):
                fusion_cls = model_no_ddp.text_proj(fusion_cls)
            cluster_features["fusion"] = F.normalize(fusion_cls, dim=-1)

    confidence_features = None
    if confidence_text_key is not None:
        if image_proj is None:
            raise ValueError("confidence calibration requires visual projection features.")

        if confidence_text_key == cluster_text_key and cluster_text_cls is not None:
            confidence_text_cls = cluster_text_cls
        else:
            confidence_text_inputs = _tokenize_texts(batch[confidence_text_key], tokenizer, config, device)
            confidence_text_out = _encode_text_hidden(model_no_ddp, confidence_text_inputs)
            confidence_text_cls = confidence_text_out.last_hidden_state[:, 0, :]

        confidence_features = {
            "visual_features": image_proj,
            "text_features": F.normalize(model_no_ddp.text_proj(confidence_text_cls), dim=-1),
        }

    return {
        "cluster_features": cluster_features,
        "confidence_features": confidence_features,
    }


def compute_jaccard_to_memmap(
    features: torch.Tensor,
    out_path: str,
    k1: int = 30,
    k2: int = 6,
    use_float16: bool = True,
    row_chunk: int = 1024,
    search_option=0
) -> str:
    """
    逐行计算 re-ranking 的 jaccard distance 并写入 disk-backed memmap，避免在 CPU 一次性占用 O(N^2) 内存。
    返回 memmap 文件路径（out_path）。
    features: Tensor[N, D] (will be moved to GPU inside)
    """
    device = torch.device("cuda")
    dtype = torch.float16 if use_float16 else torch.float32

    feats = features.to(device=device, dtype=dtype)  # [N, D]
    N, D = feats.shape
    feats_t = feats.t()  # reuse for similarity

    # 1) initial rank (k1 + 1 neighbors)
    initial_rank = torch.empty((N, k1 + 1), dtype=torch.long, device=device)
    for start in range(0, N, row_chunk):
        end = min(start + row_chunk, N)
        sim = torch.matmul(feats[start:end], feats_t)  # [b, N]
        _, idx = torch.topk(sim, k=k1 + 1, dim=1, largest=True, sorted=True)
        initial_rank[start:end] = idx
        del sim, idx

    # 2) reciprocal neighbors (在 CPU 上做 set 运算)
    init_cpu = initial_rank.cpu()
    nn_k1 = []
    nn_k1_half = []
    half_k1 = (k1 + 1) // 2
    for i in range(N):
        neigh = init_cpu[i]  # (k1+1,)
        back = init_cpu[neigh][:, : k1 + 1]  # [k1+1, k1+1]
        mask = (back == i).any(dim=1)
        k_recip = neigh[mask]
        nn_k1.append(k_recip)

        neigh2 = init_cpu[i][: half_k1 + 1]
        back2 = init_cpu[neigh2][:, : half_k1 + 1]
        mask2 = (back2 == i).any(dim=1)
        nn_k1_half.append(neigh2[mask2])

    # 3) 构造 V（dense，放在 GPU 上），然后逐行计算 jaccard 并 flush 到 disk
    V = torch.zeros((N, N), dtype=dtype, device=device)  # 可能 ~9GB float16 for N=68000

    for i in range(N):
        k_recip_cpu = nn_k1[i]
        if k_recip_cpu.numel() == 0:
            continue
        exp_set = set(int(x) for x in k_recip_cpu.tolist())

        for cand in list(exp_set):
            neigh2_cpu = nn_k1_half[cand]
            if neigh2_cpu.numel() == 0:
                continue
            common = len(set(neigh2_cpu.tolist()) & set(k_recip_cpu.tolist()))
            if common > (2 / 3) * neigh2_cpu.numel():
                exp_set.update(int(x) for x in neigh2_cpu.tolist())

        exp = torch.tensor(sorted(exp_set), device=device, dtype=torch.long)
        if exp.numel() == 0:
            continue
        dist = 2.0 - 2.0 * torch.matmul(feats[i].unsqueeze(0), feats[exp].t())  # [1, M]
        weights = F.softmax(-dist, dim=1).view(-1)  # [M]
        V[i, exp] = weights  # sparse fill

    # 4) query expansion
    if k2 != 1:
        for i in range(N):
            idx = initial_rank[i, :k2]
            V[i] = V[idx].mean(dim=0)

    # 5) 准备 memmap 输出
    dtype_np = np.float16 if use_float16 else np.float32
    dirname = os.path.dirname(out_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    jaccard_mmap = np.memmap(out_path, dtype=dtype_np, mode="w+", shape=(N, N))

    # 6) 逐行计算 jaccard 并写入 disk
    for i in range(N):
        v_i = V[i]  # [N] on GPU
        nz = torch.nonzero(v_i, as_tuple=False).view(-1)
        if nz.numel() == 0:
            row_gpu = torch.ones((N,), dtype=dtype, device=device)
        else:
            V_sub = V[:, nz]  # [N, K]
            v_i_sub = v_i[nz].unsqueeze(0)  # [1, K]
            tmp_min = torch.min(V_sub, v_i_sub).sum(dim=1)  # [N]
            row_gpu = 1.0 - tmp_min / (2.0 - tmp_min)  # [N]
        row_cpu = row_gpu.to("cpu").to(dtype if use_float16 else dtype).numpy()
        jaccard_mmap[i, :] = row_cpu  # 写一行

    jaccard_mmap.flush()

    # 清理 GPU 资源
    del V, feats, feats_t, initial_rank
    torch.cuda.empty_cache()
    gc.collect()

    return out_path  # memmap 路径


def dbscan_memmap(jaccard_path: str, eps: float = 0.6, min_samples: int = 4):
    """
    在磁盘-backed 的 jaccard memmap 上做 DBSCAN（不一次性读入整个矩阵）。
    返回 numpy labels。
    """
    jaccard = np.memmap(jaccard_path, dtype=np.float16, mode="r")
    N = int(np.sqrt(jaccard.size))
    jaccard = jaccard.reshape((N, N))

    core_mask = np.zeros(N, dtype=bool)
    # 1. core 判断（按行读）
    for i in range(N):
        row = jaccard[i]
        if np.count_nonzero(row <= eps) >= min_samples:
            core_mask[i] = True

    labels = -1 * np.ones(N, dtype=int)
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0

    # 2. 聚类扩展（经典 DBSCAN BFS）
    for i in range(N):
        if visited[i] or not core_mask[i]:
            continue
        labels[i] = cluster_id
        visited[i] = True
        queue = [i]
        while queue:
            curr = queue.pop()
            curr_row = jaccard[curr]
            nbrs = np.nonzero(curr_row <= eps)[0]
            for nb in nbrs:
                if not visited[nb]:
                    visited[nb] = True
                    if core_mask[nb]:
                        queue.append(nb)
                if labels[nb] == -1:
                    labels[nb] = cluster_id
        cluster_id += 1

    return labels  # numpy array


def _cluster_single_mode(
    features: torch.Tensor,
    *,
    mode: str,
    config,
    args,
    logger,
):
    save_path = f"./logs/pseudo_labels_{mode}.pt"
    jaccard_path = f"./tmp/{mode}_rerank_jaccard.memmap"

    if args.distributed:
        search_option = 2
        logger.info("[Cluster][%s] Rank %s | 开始计算距离", mode, torch.distributed.get_rank())
    else:
        search_option = 3
        logger.info("[Cluster][%s] 单卡 | 开始计算距离", mode)

    compute_jaccard_to_memmap(
        features,
        out_path=jaccard_path,
        k1=int(config.get("cluster_k1", 30)),
        k2=int(config.get("cluster_k2", 6)),
        use_float16=True,
        row_chunk=int(config.get("cluster_row_chunk", 1024)),
        search_option=search_option,
    )

    logger.info("[Cluster][%s] 开始基于 memmap 的 DBSCAN 聚类", mode)
    pseudo_labels = dbscan_memmap(
        jaccard_path,
        eps=float(config.get("cluster_eps", 0.6)),
        min_samples=int(config.get("cluster_min_samples", 4)),
    )
    if not os.path.exists(save_path):
        torch.save(pseudo_labels, save_path)
        logger.info("[Cluster][%s] 首轮伪标签快照已保存至 %s", mode, save_path)

    num_noise = int((pseudo_labels == -1).sum())
    unique_labels = set(pseudo_labels.tolist())
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(
        "[Cluster][%s] 聚类完成 | total=%d, clusters=%d, noise=%d",
        mode,
        len(pseudo_labels),
        num_clusters,
        num_noise,
    )

    return {
        "pseudo_labels": pseudo_labels,
        "num_clusters": num_clusters,
        "num_noise": num_noise,
    }


@torch.no_grad()
def cluster_begin_epoch(train_loader, model, args, config, tokenizer=None, logger=None):
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()
    if args.distributed and hasattr(model, "module"):
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    max_size = len(train_loader.dataset)
    train_mode = str(config.get("cluster_feature_mode", "image")).lower()
    modes_to_cluster = []
    for mode in ("image", "text", train_mode):
        if mode not in modes_to_cluster:
            modes_to_cluster.append(mode)

    feature_banks = {
        mode: torch.empty(
            (max_size, _infer_cluster_feat_dim(model_no_ddp, config, mode=mode)),
            device=device,
            dtype=torch.float16,
        )
        for mode in modes_to_cluster
    }
    conf_cfg = get_conf_calibration_cfg(config)
    need_confidence_cache = bool(conf_cfg["enabled"])
    visual_bank = None
    text_bank = None
    if need_confidence_cache:
        embed_dim = int(model_no_ddp.vision_proj.out_features)
        visual_bank = torch.empty((max_size, embed_dim), device=device, dtype=torch.float16)
        text_bank = torch.empty((max_size, embed_dim), device=device, dtype=torch.float16)

    index = 0
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    logger.info(
        "开始计算伪标签 | train_cluster_feature_mode=%s | monitor_modes=%s",
        train_mode,
        ",".join([mode for mode in ("image", "text") if mode in modes_to_cluster]),
    )

    for _, batch in enumerate(train_loader):
        feature_bundle = extract_cluster_feature_bundle(
            batch,
            model_no_ddp,
            model_no_ddp.tokenizer,
            config,
            device,
            need_modes=modes_to_cluster,
            confidence_text_key=str(conf_cfg["text_key"]) if need_confidence_cache else None,
        )
        cluster_features = feature_bundle["cluster_features"]
        bs = batch["image1"].size(0)
        for mode in modes_to_cluster:
            feature_banks[mode][index:index + bs] = cluster_features[mode].to(torch.float16)
        if need_confidence_cache:
            confidence_features = feature_bundle["confidence_features"]
            visual_bank[index:index + bs] = confidence_features["visual_features"].to(torch.float16)
            text_bank[index:index + bs] = confidence_features["text_features"].to(torch.float16)
        index += bs

    if index != max_size:
        # 重要：如果你在 DDP 下用 DistributedSampler，这里通常会触发（每张卡只看一部分数据）
        logger.warning(f"注意：feature bank 填充长度 index={index} != dataset_len={max_size}。"
                       f"如果你在分布式聚类，请确保用全量数据/或先 gather 特征。")

    confidence_feature_cache = None
    if need_confidence_cache:
        confidence_feature_cache = {
            "visual_features": visual_bank[:index].cpu(),
            "text_features": text_bank[:index].cpu(),
        }
        del visual_bank, text_bank

    cluster_outputs = {}
    for mode in modes_to_cluster:
        feat_bank_used = feature_banks[mode][:index].to(torch.float32)
        cluster_outputs[mode] = _cluster_single_mode(
            feat_bank_used,
            mode=mode,
            config=config,
            args=args,
            logger=logger,
        )
        del feature_banks[mode], feat_bank_used
        torch.cuda.empty_cache()
        gc.collect()

    train_cluster_output = cluster_outputs[train_mode]
    logger.info(
        "Dataset 总长度: %d, 训练伪标签长度: %d, train_cluster_feature_mode=%s\n",
        len(train_loader.dataset),
        len(train_cluster_output["pseudo_labels"]),
        train_mode,
    )

    return {
        "pseudo_labels": train_cluster_output["pseudo_labels"],
        "feature_cache": confidence_feature_cache,
        "monitor_labels": {
            mode: cluster_outputs[mode]["pseudo_labels"]
            for mode in ("image", "text")
            if mode in cluster_outputs
        },
        "train_mode": train_mode,
    }
