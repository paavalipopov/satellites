"""The dataset contract every loader here shares: sampler and collate.

Concrete datasets live beside this file; ``src/registry.py`` picks between them.
"""

import numpy as np
import torch


class SeededBatchSampler:
    """Vendored from ``DBBatchSampler`` in mindfultensors:
    https://github.com/neuroneural/mindfultensors/blob/main/mindfultensors/utils.py


    Passed as ``sampler=`` (not ``batch_sampler=``), which is what makes each yielded
    item arrive at ``__getitem__`` as a single "index". The item is ``(epoch, indices)``:
    workers see only what the sampler yields, so this is how the epoch reaches them.
    """

    def __init__(self, data_source, batch_size=1, seed=None):
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch): #TODO why do we need to set and return epoch again?
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(None if self.seed is None else self.seed + self.epoch)
        order = rng.permutation(self.data_size)
        return iter([(self.epoch, batch) for batch in self._batches(order)])

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
        # sizes depend only on data_size and batch_size, not on the ordering
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

