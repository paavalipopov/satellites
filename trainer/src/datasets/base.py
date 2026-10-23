"""The dataset contract every loader here shares: sampler and collate.

Concrete datasets live beside this file; ``src/registry.py`` picks between them.
"""

import numpy as np
import torch


class EpochBatchSampler:
    """Consecutive batches of indices, each tagged with the epoch. Adapted from
    ``DBBatchSampler`` in mindfultensors:
    https://github.com/neuroneural/mindfultensors/blob/main/mindfultensors/utils.py

    Passed as ``sampler=`` (not ``batch_sampler=``), which is what makes each yielded
    item arrive at ``__getitem__`` as a single "index". The item is ``(epoch, indices)``:
    workers see only what the sampler yields, so this is how the epoch reaches them.
    No shuffling: every scene is random by its seed, so order carries no information.
    """

    def __init__(self, data_source, batch_size=1):
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.epoch = 0

    def set_epoch(self, epoch):  # called by accelerate before every epoch, and on resume
        self.epoch = epoch

    def __iter__(self):
        return iter([(self.epoch, batch) for batch in self._batches(np.arange(self.data_size))])

    def _batches(self, order):
        """Split an ordering into batches, leaving no batch of one."""
        step = self.batch_size
        batches = [order[i : i + step] for i in range(0, self.data_size, step)]
        # Fix for trailing batch of size 1
        if len(batches) > 1 and len(batches[-1]) == 1:
            tail = np.concatenate(batches[-2:])
            cut = len(tail) // 2
            batches[-2:] = [tail[:cut], tail[cut:]] if cut > 1 else [tail]
        return batches

    def __len__(self):
        return len(self._batches(np.arange(self.data_size)))


def collate_volumes(results):
    """``[{index: record}]`` -> ``(inputs [B,C,H,W], masks [B,H,W], indices [B])``.

    ``indices`` are positions within the loader's split; with the split's seeds they
    identify the scene (see ``HSISplit.seed``).
    """
    results = results[0]
    inputs = torch.stack([record["input"] for record in results.values()])
    labels = torch.stack([record["label"] for record in results.values()])
    indices = torch.tensor([record["index"] for record in results.values()])
    return inputs, labels.long(), indices.long()

