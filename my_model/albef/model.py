# ──────────────────────────────────────────────────────────────────────────────
# Project layout (proposed)
# ──────────────────────────────────────────────────────────────────────────────
# my_model/
#   vit.py
#   xbert.py
#   albef/
#     __init__.py
#     model.py                  # ALBEF main module (forward intact)
#     mixins/
#       __init__.py
#       vision.py               # _build_vit
#       momentum.py             # copy_params, _momentum_update
#       queues.py               # _init_queues, _dequeue_and_enqueue, reset_queues, concat_all_gather
#       mlm.py                  # standard MLM mask
#       infmask.py              # compute_infmask_loss
# ──────────────────────────────────────────────────────────────────────────────
import math
from collections import defaultdict
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch import nn
from my_model.xbert import BertConfig, BertForMaskedLM
from utils.confidence import get_conf_calibration_cfg, normalize_weighted_mean
from .mixins import (
    VisionBuilderMixin,
    MomentumMixin,
    QueueMixin,
    MLMMixin,
    concat_all_gather,
)
from .mixins.infmask import InfMaskMixin


def get_relation_rectification_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": bool(config.get("use_relation_rectification", True)),
        "warmup_epochs": int(config.get("relation_warmup_epochs", 1)),
        "mu": float(config.get("relation_mu", 0.5)),
        "gamma": float(config.get("relation_gamma", 0.1)),
        "lambda": float(config.get("relation_lambda", 0.5)),
        "hard_neg_ratio": float(config.get("hard_neg_ratio", 0.25)),
        "max_rel_pairs_per_id": int(config.get("max_rel_pairs_per_id", 1)),
        "detach_relation_weight": bool(config.get("detach_relation_weight", True)),
        "eps": float(config.get("relation_eps", 1e-6)),
    }


class ALBEF(VisionBuilderMixin, MomentumMixin, QueueMixin, MLMMixin, InfMaskMixin, nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None, config: Dict[str, Any] = None):
        super().__init__()
        if config is None:
            config = {}

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        self.mrtd_mask_probability = config['mrtd_mask_probability']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        image_res = config['image_res']

        # Vision Encoder
        self.visual_encoder = self._build_vit(image_res)
        self.vision_proj = nn.Linear(vision_width, embed_dim)

        # Text Encoder
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(self.text_width, embed_dim)

        # Heads
        self.itm_head = nn.Linear(self.text_width, 2)
        # ✱ 新增：ITM 动量头
        self.itm_head_m = nn.Linear(self.text_width, 2)

        # Temperature parameter
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        # Momentum models
        self.visual_encoder_m = self._build_vit(image_res)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(self.text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
            # ✱ 新增：ITM 线性头也做 EMA
            [self.itm_head, self.itm_head_m],
        ]
        self.copy_params()
        # Queues
        self._init_queues(embed_dim)
        self.latest_relation_stats = self._init_relation_stats(active=False)
        self.latest_relation_debug = {}

    def _init_relation_stats(self, active: bool) -> Dict[str, float]:
        return {
            "active": float(active),
            "mean_q_aa": 0.0,
            "mean_q_ab": 0.0,
            "mean_r_ab": 0.0,
            "mean_g_ab": 0.0,
            "num_verified_pairs": 0.0,
            "num_hard_negatives": 0.0,
        }

    def _build_relation_pairs(
        self,
        pseudo_labels: torch.Tensor,
        max_pairs_per_id: int,
    ) -> Optional[torch.Tensor]:
        labels = pseudo_labels.view(-1)
        groups = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            if int(label) < 0:
                continue
            groups[int(label)].append(idx)

        pairs = []
        for members in groups.values():
            if len(members) == 2:
                pairs.append((members[0], members[1]))
                continue
            if len(members) < 2:
                continue

            order = torch.randperm(len(members), device=labels.device).tolist()
            shuffled = [members[i] for i in order]
            num_pairs = min(len(shuffled) // 2, max(0, int(max_pairs_per_id)))
            for pair_idx in range(num_pairs):
                base = 2 * pair_idx
                pairs.append((shuffled[base], shuffled[base + 1]))

        if not pairs:
            return None
        return torch.tensor(pairs, dtype=torch.long, device=labels.device)

    def _compute_itm_positive_prob(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        text_embeds: torch.Tensor,
        text_atts: torch.Tensor,
    ) -> torch.Tensor:
        output = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion',
            output_hidden_states=False,
        )
        logits = self.itm_head(output.last_hidden_state[:, 0, :].float())
        return F.softmax(logits, dim=-1)[:, 1]

    def _compute_relation_scores(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        text_embeds: torch.Tensor,
        text_atts: torch.Tensor,
        relation_pairs: torch.Tensor,
        relation_cfg: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        a_idx = relation_pairs[:, 0]
        b_idx = relation_pairs[:, 1]
        image_idx = torch.cat([a_idx, a_idx, b_idx, b_idx], dim=0)
        text_idx = torch.cat([a_idx, b_idx, a_idx, b_idx], dim=0)

        q_all = self._compute_itm_positive_prob(
            image_embeds=image_embeds[image_idx],
            image_atts=image_atts[image_idx],
            text_embeds=text_embeds[text_idx],
            text_atts=text_atts[text_idx],
        )

        num_pairs = relation_pairs.shape[0]
        q_aa = q_all[:num_pairs]
        q_ab = q_all[num_pairs: 2 * num_pairs]
        q_ba = q_all[2 * num_pairs: 3 * num_pairs]
        q_bb = q_all[3 * num_pairs:]

        eps = float(relation_cfg["eps"])
        gamma = max(float(relation_cfg["gamma"]), eps)
        r_ab = 0.5 * (q_ab / (q_aa + eps) + q_ba / (q_bb + eps))
        g_ab = torch.sigmoid((r_ab - float(relation_cfg["mu"])) / gamma)
        g_targets = g_ab.detach() if bool(relation_cfg["detach_relation_weight"]) else g_ab

        return {
            "pairs": relation_pairs,
            "q_aa": q_aa,
            "q_ab": q_ab,
            "q_ba": q_ba,
            "q_bb": q_bb,
            "r_ab": r_ab,
            "g_ab": g_ab,
            "g_targets": g_targets,
        }

    def _build_relation_soft_targets(
        self,
        batch_size: int,
        total_candidates: int,
        relation_pairs: torch.Tensor,
        relation_weights: torch.Tensor,
        relation_lambda: float,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> torch.Tensor:
        targets = torch.zeros((batch_size, total_candidates), device=device, dtype=dtype)
        diag_idx = torch.arange(batch_size, device=device)
        targets[diag_idx, diag_idx] = 1.0

        if relation_pairs.numel() > 0:
            a_idx = relation_pairs[:, 0]
            b_idx = relation_pairs[:, 1]
            targets[a_idx, b_idx] += relation_lambda * relation_weights.to(device=device, dtype=dtype)
            targets[b_idx, a_idx] += relation_lambda * relation_weights.to(device=device, dtype=dtype)

        row_sums = targets.sum(dim=1, keepdim=True)
        zero_rows = row_sums.squeeze(1) <= eps
        if zero_rows.any():
            targets[zero_rows] = 0.0
            fallback_idx = torch.arange(batch_size, device=device)[zero_rows]
            targets[zero_rows, fallback_idx] = 1.0
            row_sums = targets.sum(dim=1, keepdim=True)
        return targets / (row_sums + eps)

    def _mine_disagreement_hard_negatives(
        self,
        relation_pairs: torch.Tensor,
        relation_scores: torch.Tensor,
        hard_neg_ratio: float,
    ) -> Optional[torch.Tensor]:
        if relation_pairs is None or relation_pairs.numel() == 0 or hard_neg_ratio <= 0:
            return None

        num_pairs = relation_pairs.shape[0]
        select_k = min(num_pairs, max(1, int(math.ceil(num_pairs * hard_neg_ratio))))
        if select_k <= 0:
            return None

        _, indices = torch.topk(relation_scores.detach(), k=select_k, largest=False)
        return relation_pairs.index_select(0, indices)

    def forward(self, batch, alpha, config, epoch):  # text2 是概率同一个 id 的其他图片描述, img1/img2 同一图不同增广
        loss_dict = {}
        conf_cfg = get_conf_calibration_cfg(config)
        relation_cfg = get_relation_rectification_cfg(config)
        relation_active = bool(relation_cfg["enabled"]) and int(epoch) >= int(relation_cfg["warmup_epochs"])
        self.latest_relation_stats = self._init_relation_stats(active=relation_active)
        self.latest_relation_debug = {}
        image1 = batch['image1']
        image2 = batch['image2']
        text1 = self.tokenizer(batch['caption1'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text2 = self.tokenizer(batch['caption2'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text_atts = text2['attention_mask']
        idx = batch['person_id']
        idx = batch['pseudo_label']  # 覆盖为伪标签，保持原逻辑
        confidence = batch.get('confidence')
        if confidence is not None:
            confidence = confidence.to(image1.device, dtype=torch.float32).view(-1)

        # extract image features
        image_embeds = self.visual_encoder(image1,register_blk=-1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        relation_info = None
        relation_pairs = None
        if relation_active:
            relation_pairs = self._build_relation_pairs(
                idx.view(-1),
                max_pairs_per_id=int(relation_cfg["max_rel_pairs_per_id"]),
            )
            if relation_pairs is not None and relation_pairs.numel() > 0:
                relation_info = self._compute_relation_scores(
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    text_embeds=text_embeds,
                    text_atts=text_atts,
                    relation_pairs=relation_pairs,
                    relation_cfg=relation_cfg,
                )
                self.latest_relation_stats.update(
                    {
                        "mean_q_aa": float(relation_info["q_aa"].detach().mean().item()),
                        "mean_q_ab": float(relation_info["q_ab"].detach().mean().item()),
                        "mean_r_ab": float(relation_info["r_ab"].detach().mean().item()),
                        "mean_g_ab": float(relation_info["g_ab"].detach().mean().item()),
                        "num_verified_pairs": float(relation_pairs.shape[0]),
                    }
                )
                self.latest_relation_debug = {
                    "pairs": relation_info["pairs"].detach(),
                    "q_aa": relation_info["q_aa"].detach(),
                    "q_ab": relation_info["q_ab"].detach(),
                    "q_ba": relation_info["q_ba"].detach(),
                    "q_bb": relation_info["q_bb"].detach(),
                    "r_ab": relation_info["r_ab"].detach(),
                    "g_ab": relation_info["g_ab"].detach(),
                }
        
        # ===== Contrastive loss =====
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        use_momentum   = bool(config.get('use_momentum', True))   # 是否用动量分支
        use_queue      = bool(config.get('use_queue', True))      # <-- 新增：是否使用队列做对比

        if enable_cl_loss:
            idx = idx.view(-1, 1)

            # -------- 1) 构造对比的候选集合：batch-only 或 batch+queue --------
            if use_queue:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
            else:
                idx_all = idx.t()  # [1, B]

            pos_idx = torch.eq(idx, idx_all).float()  # [B, B(+Q)]
            sim_targets = pos_idx / (pos_idx.sum(1, keepdim=True) + 1e-8)
            if relation_info is not None:
                sim_targets = self._build_relation_soft_targets(
                    batch_size=image1.size(0),
                    total_candidates=idx_all.shape[1],
                    relation_pairs=relation_info["pairs"],
                    relation_weights=relation_info["g_targets"],
                    relation_lambda=float(relation_cfg["lambda"]),
                    device=image1.device,
                    dtype=image_feat.dtype,
                    eps=float(relation_cfg["eps"]),
                )

            # -------- 2) 计算 soft targets（可选动量） --------
            sim_i2t_targets = sim_targets
            sim_t2i_targets = sim_targets
            with torch.no_grad():
                if use_momentum:
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text2['attention_mask'],
                        return_dict=True,
                        mode='text'
                    )
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

                    if use_queue:
                        image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D, B+Q]
                        text_feat_all  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)  # [D, B+Q]
                    else:
                        image_feat_all = image_feat_m.t()  # [D, B]
                        text_feat_all  = text_feat_m.t()   # [D, B]

                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # [B, B(+Q)]
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # [B, B(+Q)]

                    sim_i2t_m_targets = F.softmax(sim_i2t_m, dim=1)
                    sim_t2i_m_targets = F.softmax(sim_t2i_m, dim=1)

                else:
                    # 非动量：targets 直接用 sim_targets（你当前就是这么做的）
                    if use_queue:
                        image_feat_all = torch.cat([image_feat.detach().t(), self.image_queue.clone().detach()], dim=1)
                        text_feat_all  = torch.cat([text_feat.detach().t(),  self.text_queue.clone().detach()],  dim=1)
                    else:
                        image_feat_all = image_feat.detach().t()
                        text_feat_all  = text_feat.detach().t()
                    sim_i2t_m_targets = None
                    sim_t2i_m_targets = None

            if sim_i2t_m_targets is not None and sim_t2i_m_targets is not None:
                sim_i2t_targets = alpha * sim_i2t_m_targets + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * sim_t2i_m_targets + (1 - alpha) * sim_targets

            # -------- 3) 用在线特征算 logits（注意：这里也要跟 use_queue 同步） --------
            if use_queue:
                # 注意：若 use_momentum=False，上面 image_feat_all/text_feat_all 已经是 detach 版本+queue
                # 若 use_momentum=True，上面 image_feat_all/text_feat_all 是 m 版本+queue（detached）
                # 这里对齐：logits 永远用 online feat 对 candidate 集合
                # candidate 集合应是：text_feat_all / image_feat_all（同维度）
                pass
            else:
                # batch-only: candidate 集合就是当前 batch 的 text_feat/image_feat
                # 为了避免混淆，直接重置为 online 的 batch-only（不依赖上面 no_grad 分支）
                image_feat_all = image_feat.t()
                text_feat_all  = text_feat.t()

            # logits
            sim_i2t = image_feat @ text_feat_all / self.temp  # [B, B(+Q)]
            sim_t2i = text_feat @ image_feat_all / self.temp  # [B, B(+Q)]

            loss_i2t_per = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1)
            loss_t2i_per = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1)
            if (
                bool(conf_cfg["enabled"])
                and bool(conf_cfg["weight_metric_loss"])
                and confidence is not None
                and confidence.shape[0] == loss_i2t_per.shape[0]
            ):
                loss_i2t = normalize_weighted_mean(loss_i2t_per, confidence, eps=float(conf_cfg["eps"]))
                loss_t2i = normalize_weighted_mean(loss_t2i_per, confidence, eps=float(conf_cfg["eps"]))
            else:
                loss_i2t = loss_i2t_per.mean()
                loss_t2i = loss_t2i_per.mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            # -------- 4) 队列更新：只有 use_queue=True 才更新 --------
            if use_queue:
                if use_momentum:
                    self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
                else:
                    self._dequeue_and_enqueue(image_feat.detach(), text_feat.detach(), idx)

        # ===== Masked Language Modeling =====
        enable_mlm_loss = bool(config.get('enable_mlm_loss', False))
        enable_soft_label = bool(config.get('mlm_soft_label', False))
        image_embeds_m = None      # ensure defined if used below
        if enable_mlm_loss:
            input_ids = text1.input_ids.clone()
            labels = input_ids.clone()
            input_ids, labels = self.mask(
                input_ids,
                self.text_encoder.config.vocab_size,
                targets=labels,
            ) 
            if enable_soft_label:
                with torch.no_grad():
                    # ensure image_embeds_m is available if CL disabled
                    if image_embeds_m is None:
                        image_embeds_m = self.visual_encoder_m(image2) if enable_cl_loss else image_embeds
                    logits_m = self.text_encoder_m(
                        input_ids,
                        attention_mask=text1.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        return_logits=True,
                    )
                mlm_output = self.text_encoder(
                    input_ids,
                    attention_mask=text1.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    labels=labels,
                    soft_labels=F.softmax(logits_m, dim=-1),
                    alpha=alpha,
                )
            else:
                mlm_output = self.text_encoder(
                    input_ids,
                    attention_mask=text1.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    labels=labels,
                )
            loss_dict['loss_mlm'] = mlm_output.loss

        # ===== ITM (matched/unmatched) =====
        enable_itm_loss = bool(config.get('enable_itm_loss', False))
        if enable_itm_loss:
            # --- 学生：正样本 (text2, image1) ---
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
                output_hidden_states=False,
            )
            with torch.no_grad():
                bs = image1.size(0)
                itm_neg_sampling = str(config.get("itm_neg_sampling", "cl")).lower()
                if itm_neg_sampling not in ("cl", "random"):
                    raise ValueError(f"config['itm_neg_sampling'] must be 'cl' or 'random', got {itm_neg_sampling}")

                idx_1d = idx.view(-1)  # [B]
                mask_same = torch.eq(idx_1d.view(bs, 1), idx_1d.view(1, bs))  # [B,B]

                if itm_neg_sampling == "cl":
                    # 沿用原本：按 CL 相似度分布采样
                    # 如果 enable_cl_loss=False，你原代码这里会现算 sim；保持不变
                    if not enable_cl_loss:
                        sim_i2t = image_feat @ text_feat.t()  # [B,B]
                        sim_t2i = text_feat @ image_feat.t()  # [B,B]

                    weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)  # [B,B]
                    weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)  # [B,B]
                    weights_i2t.masked_fill_(mask_same, 0.0)
                    weights_t2i.masked_fill_(mask_same, 0.0)

                    # 兜底：若某行全 0（极端情况下同 id 太多），退化为随机（排除自身）
                    if (weights_i2t.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_i2t = w / (w.sum(dim=1, keepdim=True) + 1e-12)
                    else:
                        weights_i2t = weights_i2t / (weights_i2t.sum(dim=1, keepdim=True) + 1e-12)

                    if (weights_t2i.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_t2i = w / (w.sum(dim=1, keepdim=True) + 1e-12)
                    else:
                        weights_t2i = weights_t2i / (weights_t2i.sum(dim=1, keepdim=True) + 1e-12)

                    image_neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)  # [B]
                    text_neg_idx  = torch.multinomial(weights_i2t, 1).squeeze(1)  # [B]

                else:
                    # random：均匀随机采样，排除同 id（含自身）
                    valid = (~mask_same).float()  # [B,B]
                    row_sum = valid.sum(dim=1, keepdim=True)

                    # 若某行无可选（整批同 id），退化为排除自身
                    if (row_sum == 0).any():
                        valid_fallback = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        valid = torch.where(row_sum > 0, valid, valid_fallback)
                        row_sum = valid.sum(dim=1, keepdim=True)

                    probs = valid / (row_sum + 1e-12)

                    image_neg_idx = torch.multinomial(probs, 1).squeeze(1)  # [B]
                    text_neg_idx  = torch.multinomial(probs, 1).squeeze(1)  # [B]



            image_embeds_neg = image_embeds[image_neg_idx]
            text_embeds_neg = text_embeds[text_neg_idx]
            text_atts_neg = text_atts[text_neg_idx]

            relation_hard_pairs = None
            if relation_info is not None:
                relation_hard_pairs = self._mine_disagreement_hard_negatives(
                    relation_pairs=relation_info["pairs"],
                    relation_scores=relation_info["g_ab"],
                    hard_neg_ratio=float(relation_cfg["hard_neg_ratio"]),
                )

            text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
            text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)

            if relation_hard_pairs is not None and relation_hard_pairs.numel() > 0:
                a_idx = relation_hard_pairs[:, 0]
                b_idx = relation_hard_pairs[:, 1]
                hard_image_idx = torch.cat([a_idx, b_idx], dim=0)
                hard_text_idx = torch.cat([b_idx, a_idx], dim=0)
                text_embeds_all = torch.cat([text_embeds_all, text_embeds[hard_text_idx]], dim=0)
                text_atts_all = torch.cat([text_atts_all, text_atts[hard_text_idx]], dim=0)
                image_embeds_all = torch.cat([image_embeds_all, image_embeds[hard_image_idx]], dim=0)
                image_atts_all = torch.cat([image_atts_all, image_atts[hard_image_idx]], dim=0)
                self.latest_relation_stats["num_hard_negatives"] = float(2 * relation_hard_pairs.shape[0])

            output_neg_cross = self.text_encoder.bert(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                mode='fusion',
            )

            vl_embeddings = torch.cat([
                output_pos.last_hidden_state[:, 0, :],              # [bs, D]
                output_neg_cross.last_hidden_state[:, 0, :],        # [2*bs, D]
            ], dim=0)                                              # [3*bs, D]
            vl_output = self.itm_head(vl_embeddings)               # [3*bs, 2]

            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(vl_output.shape[0] - bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            # 纯硬标签 CE
            loss_itm = F.cross_entropy(vl_output, itm_labels)
            loss_dict['loss_itm'] = loss_itm

        return loss_dict
