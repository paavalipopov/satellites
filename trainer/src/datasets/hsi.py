"""Synthetic HSI scenes from ``src.data_generation``, generated on the fly by the loader workers.

``data.name=hsi``. One ``HSIDataset`` holds the generator settings and the material
library; ``split()`` gives each loader its own dataset. Every scene is fixed by its seed:

    train   scene_seed(seed_train, fold, epoch, index)   fresh every epoch and fold
    valid   scene_seed(seed_valid, index)                 the same set every epoch and rerun
    test    scene_seed(seed_test, index)
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data_generation import N_CLASSES, SimConfig, SpectralLibrary, generate_sample

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # holds assets/materials


def default_HPs(cfg):
    return OmegaConf.create(
        {
            "n_train": 10000,       # fresh scenes per epoch
            "n_valid": 1000,
            "n_test": 1000,
            "seed_train": 0,
            "seed_valid": 1,
            "seed_test": 2,
            "scene": "random",      # random (cfg.scene_probs mix) | satellite | shapes | patches
            "material_root": None,  # where sim.material_dir resolves from; null = the project root
            "sim": asdict(SimConfig()),  # every generator setting, so config.yaml records them all
        }
    )


def scene_seed(*keys):
    """One generator seed from a tuple of ints; distinct tuples give unrelated scenes."""
    return int(np.random.SeedSequence(keys).generate_state(1, np.uint64)[0])


class HSIDataset:
    """Generator config and material library, shared by the three splits."""

    def __init__(self, params):
        self.params = params
        self.sim = SimConfig(**params.sim) # TODO its better to call it cfg I think?
        self.library = SpectralLibrary(self.sim, root=str(params.material_root or PROJECT_ROOT))
        self.n_channels = self.sim.n_bands  # main() sizes the model from these two
        self.n_classes = N_CLASSES

    def split(self, name, fold=0):
        """The dataset for loader ``name``: train, valid or test."""
        size, seed = self.params[f"n_{name}"], self.params[f"seed_{name}"]
        if name == "train":
            return HSISplit(self, size, keys=(seed, fold), fresh=True)
        return HSISplit(self, size, keys=(seed,), fresh=False)


class HSISplit(Dataset):
    """``size`` scenes; ``fresh`` adds the epoch to every seed."""

    def __init__(self, parent, size, keys, fresh):
        self.sim, self.library, self.scene = parent.sim, parent.library, parent.params.scene
        self.size, self.keys, self.fresh = size, tuple(keys), fresh

    def __len__(self):
        return self.size

    def seed(self, epoch, index):
        return scene_seed(*self.keys, *([epoch] if self.fresh else []), index)

    def __getitem__(self, item):
        epoch, batch = item  # what SeededBatchSampler yields
        results = {}
        for index in map(int, batch):
            sample = generate_sample(self.sim, self.seed(epoch, index), self.library, scene=self.scene)
            results[index] = {
                "input": torch.from_numpy(sample.cube).float(),  # [bands, H, W]
                "label": torch.from_numpy(sample.mask).long(),   # [H, W], 0 = background
                "index": index,
            }
        return results


DATASET = HSIDataset
