import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

class ValidIndexDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=True,
        batch_size=1,
        num_instances=2,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.epoch = 0
        self.batch_size = max(1, int(batch_size))
        self.num_instances = max(2, int(num_instances))
        self.valid_indices = self._sync_valid_indices()

    def _sync_valid_indices(self):
        """
        获取并同步 valid_indices，避免多进程不一致。
        """
        if not dist.is_available() or not dist.is_initialized():
            if hasattr(self.dataset, 'valid_indices'):
                return list(self.dataset.valid_indices)
            return list(range(len(self.dataset)))

        if self.rank == 0:
            if hasattr(self.dataset, 'valid_indices'):
                valid_indices = list(self.dataset.valid_indices)
            else:
                valid_indices = list(range(len(self.dataset)))
        else:
            valid_indices = None

        object_list = [valid_indices]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    def set_valid_indices(self, valid_indices):
        self.valid_indices = list(valid_indices)

    def _build_grouped_positions(self):
        pseudo_labels = getattr(self.dataset, 'pseudo_labels', None)
        if pseudo_labels is None or len(self.valid_indices) <= 1:
            return None

        groups = defaultdict(list)
        for logical_pos, real_idx in enumerate(self.valid_indices):
            label = pseudo_labels[real_idx]
            if hasattr(label, "item"):
                label = label.item()
            groups[int(label)].append(logical_pos)

        if not groups:
            return None

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        pair_chunks = []
        singletons = []
        for label, positions in groups.items():
            if label == -1:
                singletons.extend(positions)
                continue

            if self.shuffle and len(positions) > 1:
                order = torch.randperm(len(positions), generator=generator).tolist()
                positions = [positions[i] for i in order]

            while len(positions) >= self.num_instances:
                pair_chunks.append(positions[:self.num_instances])
                positions = positions[self.num_instances:]
            if positions:
                singletons.extend(positions)

        if self.shuffle and len(pair_chunks) > 1:
            order = torch.randperm(len(pair_chunks), generator=generator).tolist()
            pair_chunks = [pair_chunks[i] for i in order]

        if self.shuffle and len(singletons) > 1:
            order = torch.randperm(len(singletons), generator=generator).tolist()
            singletons = [singletons[i] for i in order]

        ordered = [idx for chunk in pair_chunks for idx in chunk]
        ordered.extend(singletons)
        return ordered if len(ordered) == len(self.valid_indices) else None

    def _build_positions(self):
        grouped_positions = self._build_grouped_positions()
        if grouped_positions is not None:
            return grouped_positions

        if not self.shuffle:
            return list(range(len(self.valid_indices)))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.valid_indices), generator=generator).tolist()

    def __iter__(self):
        positions = self._build_positions()
        global_batch = self.batch_size * self.num_replicas

        if self.drop_last:
            total_size = (len(positions) // global_batch) * global_batch
            positions = positions[:total_size]
        else:
            total_size = int(math.ceil(len(positions) / global_batch)) * global_batch
            padding_size = total_size - len(positions)
            if padding_size > 0:
                if not positions:
                    positions = [0] * total_size
                else:
                    repeat = (padding_size + len(positions) - 1) // len(positions)
                    positions += (positions * repeat)[:padding_size]

        num_global_batches = total_size // global_batch if global_batch > 0 else 0
        rank_positions = []
        for batch_idx in range(num_global_batches):
            start = batch_idx * global_batch + self.rank * self.batch_size
            end = start + self.batch_size
            rank_positions.extend(positions[start:end])

        return iter(rank_positions)

    def __len__(self):
        global_batch = self.batch_size * self.num_replicas
        if self.drop_last:
            return (len(self.valid_indices) // global_batch) * self.batch_size
        return int(math.ceil(len(self.valid_indices) / global_batch)) * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
