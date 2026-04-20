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
from collections import defaultdict
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch import nn
from my_model.xbert import BertConfig, BertForMaskedLM
from utils.confidence import get_conf_calibration_cfg
from .mixins import (
    VisionBuilderMixin,
    MomentumMixin,
    QueueMixin,
    MLMMixin,
    concat_all_gather,
)
from .mixins.infmask import InfMaskMixin


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
        self.latest_pair_release_stats = self._init_pair_release_stats(active=False)

    def _init_pair_release_stats(self, active: bool) -> Dict[str, float]:
        return {
            "active": float(active),
            "num_candidates": 0.0,
            "num_released": 0.0,
        }

    def _sample_low_conf_pair_candidates(
        self,
        pseudo_labels: torch.Tensor,
        confidence_groups: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        labels = pseudo_labels.view(-1)
        confidence_groups = confidence_groups.view(-1)
        cluster_members = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            if int(label) < 0:
                continue
            cluster_members[int(label)].append(idx)

        pairs = []
        for members in cluster_members.values():
            if len(members) < 2:
                continue

            # Conservative release only inspects one-to-one low-vs-non-low pairs inside a pseudo-ID cluster.
            low_members = [member for member in members if int(confidence_groups[member].item()) == 0]
            non_low_members = [member for member in members if int(confidence_groups[member].item()) > 0]
            if not low_members or not non_low_members:
                continue

            low_order = torch.randperm(len(low_members), device=labels.device).tolist()
            non_low_order = torch.randperm(len(non_low_members), device=labels.device).tolist()
            paired_count = min(len(low_members), len(non_low_members))
            for offset in range(paired_count):
                pairs.append([low_members[low_order[offset]], non_low_members[non_low_order[offset]]])

        if not pairs:
            return None
        return torch.tensor(pairs, dtype=torch.long, device=labels.device)

    def _select_releasable_low_conf_pairs(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        text_embeds: torch.Tensor,
        text_atts: torch.Tensor,
        positive_match_scores: torch.Tensor,
        candidate_pairs: Optional[torch.Tensor],
        fusion_encoder,
        itm_head,
    ) -> Optional[torch.Tensor]:
        if candidate_pairs is None or candidate_pairs.numel() == 0:
            return None

        a_idx = candidate_pairs[:, 0]
        b_idx = candidate_pairs[:, 1]
        image_idx = torch.cat([a_idx, b_idx], dim=0)
        text_idx = torch.cat([b_idx, a_idx], dim=0)

        output = fusion_encoder.bert(
            encoder_embeds=text_embeds[text_idx],
            attention_mask=text_atts[text_idx],
            encoder_hidden_states=image_embeds[image_idx],
            encoder_attention_mask=image_atts[image_idx],
            return_dict=True,
            mode='fusion',
            output_hidden_states=False,
        )
        matched_probs = F.softmax(itm_head(output.last_hidden_state[:, 0, :].float()), dim=-1)[:, 1]

        num_pairs = candidate_pairs.shape[0]
        s_ij = matched_probs[:num_pairs]
        s_ji = matched_probs[num_pairs:]
        # Release only when both cross-pair scores stay below both original matched scores.
        self_min = torch.minimum(positive_match_scores[a_idx], positive_match_scores[b_idx])
        cross_max = torch.maximum(s_ij, s_ji)
        release_mask = cross_max < self_min
        if not torch.any(release_mask):
            return None
        return candidate_pairs[release_mask]

    def forward(self, batch, alpha, config, epoch):  # text2 是概率同一个 id 的其他图片描述, img1/img2 同一图不同增广
        loss_dict = {}
        conf_cfg = get_conf_calibration_cfg(config)
        image1 = batch['image1']
        image2 = batch['image2']
        text1 = self.tokenizer(batch['caption1'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text2 = self.tokenizer(batch['caption2'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text_atts = text2['attention_mask']
        idx = batch['person_id']
        idx = batch['pseudo_label']  # 覆盖为伪标签，保持原逻辑
        enable_low_conf_pair_release = bool(config.get("enable_low_conf_pair_release", True))
        pair_release_warmup_epochs = int(config.get("pair_release_warmup_epochs", 6))
        pair_release_active = enable_low_conf_pair_release and int(epoch) >= pair_release_warmup_epochs
        self.latest_pair_release_stats = self._init_pair_release_stats(active=pair_release_active)
        confidence = batch.get('confidence')
        if confidence is None:
            confidence = torch.ones(image1.size(0), device=image1.device, dtype=torch.float32)
        else:
            confidence = confidence.to(image1.device, dtype=torch.float32).view(-1)
        confidence_group = batch.get('confidence_group')
        if confidence_group is not None:
            confidence_group = confidence_group.to(image1.device, dtype=torch.long).view(-1)

        # extract image features
        image_embeds = self.visual_encoder(image1,register_blk=-1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # ===== Contrastive loss =====
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        use_momentum   = bool(config.get('use_momentum', True))   # 是否用动量分支
        use_queue      = bool(config.get('use_queue', True))      # <-- 新增：是否使用队列做对比

        if enable_cl_loss:
            idx = idx.view(-1, 1)

            # -------- 1) 构造对比的候选集合：batch-only 或 batch+queue --------
            if use_queue:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
                confidence_all = torch.cat(
                    [confidence.view(1, -1), self.confidence_queue.clone().detach()],
                    dim=1,
                )  # [1, B+Q]
            else:
                idx_all = idx.t()  # [1, B]
                confidence_all = confidence.view(1, -1)  # [1, B]

            pos_idx = torch.eq(idx, idx_all).float()  # [B, B(+Q)]
            # Pair-aware pseudo-positive target: positives are reweighted by target-side reliability w_j.
            sim_targets = pos_idx * confidence_all.to(dtype=pos_idx.dtype)
            sim_targets = sim_targets / (sim_targets.sum(1, keepdim=True) + 1e-8)

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
            else:
                sim_i2t_targets = sim_targets
                sim_t2i_targets = sim_targets

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
            loss_i2t = loss_i2t_per.mean()
            loss_t2i = loss_t2i_per.mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            # -------- 4) 队列更新：只有 use_queue=True 才更新 --------
            if use_queue:
                if use_momentum:
                    self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx, confidence)
                else:
                    self._dequeue_and_enqueue(image_feat.detach(), text_feat.detach(), idx, confidence)

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
            pos_vl_output = self.itm_head(output_pos.last_hidden_state[:, 0, :].float())
            with torch.no_grad():
                bs = image1.size(0)
                itm_neg_sampling = str(config.get("itm_neg_sampling", "cl")).lower()
                if itm_neg_sampling not in ("cl", "random"):
                    raise ValueError(f"config['itm_neg_sampling'] must be 'cl' or 'random', got {itm_neg_sampling}")

                idx_1d = idx.view(-1)  # [B]
                mask_exclude = torch.eq(idx_1d.view(bs, 1), idx_1d.view(1, bs))  # [B,B]
                candidate_pairs = None
                released_pairs = None
                if pair_release_active and confidence_group is not None and confidence_group.shape[0] == bs:
                    candidate_pairs = self._sample_low_conf_pair_candidates(
                        pseudo_labels=idx_1d,
                        confidence_groups=confidence_group,
                    )
                    released_pairs = self._select_releasable_low_conf_pairs(
                        image_embeds=image_embeds,
                        image_atts=image_atts,
                        text_embeds=text_embeds,
                        text_atts=text_atts,
                        positive_match_scores=F.softmax(pos_vl_output.detach(), dim=-1)[:, 1],
                        candidate_pairs=candidate_pairs,
                        fusion_encoder=self.text_encoder,
                        itm_head=self.itm_head,
                    )
                    if released_pairs is not None and released_pairs.numel() > 0:
                        release_a = released_pairs[:, 0]
                        release_b = released_pairs[:, 1]
                        mask_exclude[release_a, release_b] = False
                        mask_exclude[release_b, release_a] = False

                self.latest_pair_release_stats.update(
                    {
                        "active": float(pair_release_active),
                        "num_candidates": float(0 if candidate_pairs is None else candidate_pairs.shape[0]),
                        "num_released": float(0 if released_pairs is None else released_pairs.shape[0]),
                    }
                )

                # Same pseudo-ID pairs stay masked by default.
                # Only low-confidence one-to-one candidates that pass the strict bottom-two ITM check are released.
                neg_anchor_idx = (~mask_exclude).any(dim=1).nonzero(as_tuple=False).view(-1)

                if itm_neg_sampling == "cl":
                    # 沿用原本：按 CL 相似度分布采样
                    # 如果 enable_cl_loss=False，你原代码这里会现算 sim；保持不变
                    if not enable_cl_loss:
                        sim_i2t = image_feat @ text_feat.t()  # [B,B]
                        sim_t2i = text_feat @ image_feat.t()  # [B,B]

                    weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)  # [B,B]
                    weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)  # [B,B]
                    weights_i2t.masked_fill_(mask_exclude, 0.0)
                    weights_t2i.masked_fill_(mask_exclude, 0.0)

                    if neg_anchor_idx.numel() > 0:
                        weights_i2t = weights_i2t[neg_anchor_idx]
                        weights_t2i = weights_t2i[neg_anchor_idx]
                        weights_i2t = weights_i2t / (weights_i2t.sum(dim=1, keepdim=True) + 1e-12)
                        weights_t2i = weights_t2i / (weights_t2i.sum(dim=1, keepdim=True) + 1e-12)
                        image_neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)  # [N]
                        text_neg_idx  = torch.multinomial(weights_i2t, 1).squeeze(1)  # [N]
                    else:
                        image_neg_idx = torch.empty(0, dtype=torch.long, device=image1.device)
                        text_neg_idx  = torch.empty(0, dtype=torch.long, device=image1.device)

                else:
                    # random：均匀随机采样，排除同 id（含自身）
                    valid = (~mask_exclude).float()  # [B,B]
                    if neg_anchor_idx.numel() > 0:
                        valid = valid[neg_anchor_idx]
                        probs = valid / (valid.sum(dim=1, keepdim=True) + 1e-12)
                        image_neg_idx = torch.multinomial(probs, 1).squeeze(1)  # [N]
                        text_neg_idx  = torch.multinomial(probs, 1).squeeze(1)  # [N]
                    else:
                        image_neg_idx = torch.empty(0, dtype=torch.long, device=image1.device)
                        text_neg_idx  = torch.empty(0, dtype=torch.long, device=image1.device)

            if neg_anchor_idx.numel() > 0:
                image_embeds_neg = image_embeds[image_neg_idx]
                image_atts_neg = image_atts[image_neg_idx]
                text_embeds_neg = text_embeds[text_neg_idx]
                text_atts_neg = text_atts[text_neg_idx]

                text_embeds_all = torch.cat([text_embeds[neg_anchor_idx], text_embeds_neg], dim=0)
                text_atts_all = torch.cat([text_atts[neg_anchor_idx], text_atts_neg], dim=0)
                image_embeds_all = torch.cat([image_embeds_neg, image_embeds[neg_anchor_idx]], dim=0)
                image_atts_all = torch.cat([image_atts_neg, image_atts[neg_anchor_idx]], dim=0)

                output_neg_cross = self.text_encoder.bert(
                    encoder_embeds=text_embeds_all,
                    attention_mask=text_atts_all,
                    encoder_hidden_states=image_embeds_all,
                    encoder_attention_mask=image_atts_all,
                    return_dict=True,
                    mode='fusion',
                )
                neg_vl_output = self.itm_head(output_neg_cross.last_hidden_state[:, 0, :].float())
            else:
                neg_vl_output = pos_vl_output.new_empty((0, pos_vl_output.shape[1]))
            vl_output = torch.cat([pos_vl_output, neg_vl_output], dim=0)  # [3*bs, 2]

            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(vl_output.shape[0] - bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            loss_itm = F.cross_entropy(vl_output, itm_labels)
            loss_dict['loss_itm'] = loss_itm

        return loss_dict
