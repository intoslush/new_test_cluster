import torch
import torch.nn.functional as F
from torch import nn

class QueueMixin:
    def _init_queues(self, embed_dim):
        self.register_buffer("image_queue", nn.functional.normalize(torch.randn(embed_dim, self.queue_size), dim=0))
        self.register_buffer("text_queue", nn.functional.normalize(torch.randn(embed_dim, self.queue_size), dim=0))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("confidence_queue", torch.zeros((1, self.queue_size), dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch._dynamo.disable
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx, confidence=None):
        # gather keys before updating queue
        if confidence is None:
            confidence = torch.ones_like(idx, dtype=self.confidence_queue.dtype, device=idx.device)
        if confidence.ndim == 1:
            confidence = confidence.view(-1, 1)
        confidence = confidence.to(device=idx.device, dtype=self.confidence_queue.dtype)

        if torch.distributed.is_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
            idxs = concat_all_gather(idx)
            confidences = concat_all_gather(confidence)
        else:
            image_feats = image_feat
            text_feats = text_feat
            idxs = idx
            confidences = confidence
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
            self.confidence_queue[:, ptr:ptr + batch_size] = confidences.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.confidence_queue[:, ptr:] = confidences[:empty].T
            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
            self.confidence_queue[:, :batch_size - empty] = confidences[empty:].T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def reset_queues(self, random_init: bool = True):
        """
        每个 epoch 重新初始化对比队列：
        - random_init=True：随机单位化向量初始化（推荐）
        - random_init=False：全部置零
        并把 idx_queue 设为 -100、指针清零；DDP 时从 rank0 广播.
        """
        is_ddp = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_ddp else 0

        if random_init:
            if rank == 0:
                img = F.normalize(torch.randn_like(self.image_queue), dim=0)
                txt = F.normalize(torch.randn_like(self.text_queue), dim=0)
                self.image_queue.copy_(img)
                self.text_queue.copy_(txt)
                self.idx_queue.fill_(-100)
                self.confidence_queue.zero_()
                self.queue_ptr.zero_()
        else:
            self.image_queue.zero_()
            self.text_queue.zero_()
            self.idx_queue.fill_(-100)
            self.confidence_queue.zero_()
            self.queue_ptr.zero_()

        if is_ddp:
            torch.distributed.broadcast(self.image_queue, src=0)
            torch.distributed.broadcast(self.text_queue, src=0)
            torch.distributed.broadcast(self.idx_queue, src=0)
            torch.distributed.broadcast(self.confidence_queue, src=0)
            torch.distributed.broadcast(self.queue_ptr, src=0)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
