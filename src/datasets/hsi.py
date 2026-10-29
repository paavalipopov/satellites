"""Synthetic HSI scenes from ``src.data_generation``, generated on the fly by the loader workers.

``data.name=hsi``. One ``HSIDataset`` holds the generator settings and the material
library; ``split()`` gives each loader its own dataset. Every scene is fixed by its seed:

    train   scene_seed(seed_train, run, epoch, index)    fresh every epoch and run
    valid   scene_seed(seed_valid, index)                 the same set every epoch and rerun
    test*   scene_seed(seed_test, index)                  one loader per entry of ``tests``
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data_generation import N_CLASSES, SimConfig, SpectralLibrary, generate_sample

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # holds assets/materials
SCENE_KINDS = ("satellite", "shapes", "patches")  # a scene's "kind" is its index here


def default_HPs(cfg):
    return OmegaConf.create(
        {
            "n_train": 10000,       # fresh scenes per epoch
            "n_valid": 1000,
            "n_test": 1000,
            "seed_train": 0,
            "seed_valid": 1,
            "seed_test": 2,
            "scene": "random",      # train and valid: random (the generator's mix) | satellite
                                    # | shapes | patches
            "tests": {"test": "random", "test_satellite": "satellite"},  # test loader -> its scene
            "material_root": None,  # where sim.material_dir resolves from; null = the project root
            "sim": asdict(SimConfig()),  # every generator setting, so config.yaml records them all
        }
    )


def scene_seed(*keys):
    """One generator seed from a tuple of ints; distinct tuples give unrelated scenes."""
    return int(np.random.SeedSequence(keys).generate_state(1, np.uint64)[0])


class HSIDataset:
    """Generator config and material library, shared by every split."""

    def __init__(self, params):
        self.params = params
        self.cfg = SimConfig(**params.sim)
        self.library = SpectralLibrary(self.cfg, root=str(params.material_root or PROJECT_ROOT))
        self.n_channels = self.cfg.n_bands  # main() sizes the model from these two
        self.n_classes = N_CLASSES
        self.loaders = ["train", "valid", *params.tests]  # train.py builds one loader each
        if not all(name.startswith("test") for name in params.tests):
            raise ValueError(f"tests: loader names must start with 'test', got {list(params.tests)}")

    def split(self, name, run=0):
        """The dataset for loader ``name``: train, valid or one of ``tests``."""
        p = self.params
        if name == "train":
            return HSISplit(self, p.n_train, (p.seed_train, run), p.scene, fresh=True)
        if name == "valid":
            return HSISplit(self, p.n_valid, (p.seed_valid,), p.scene, fresh=False)
        return HSISplit(self, p.n_test, (p.seed_test,), p.tests[name], fresh=False)


class HSISplit(Dataset):
    """``size`` scenes; ``fresh`` adds the epoch to every seed."""

    def __init__(self, parent, size, keys, scene, fresh):
        self.cfg, self.library = parent.cfg, parent.library
        self.size, self.keys, self.scene, self.fresh = size, tuple(keys), scene, fresh

    def __len__(self):
        return self.size

    def seed(self, epoch, index):
        return scene_seed(*self.keys, *([epoch] if self.fresh else []), index)

    def __getitem__(self, item):
        epoch, batch = item  # what SeededBatchSampler yields
        results = {}
        for index in map(int, batch):
            sample = generate_sample(self.cfg, self.seed(epoch, index), self.library, scene=self.scene)
            results[index] = {
                "input": torch.from_numpy(sample.cube).float(),  # [bands, H, W]
                "label": torch.from_numpy(sample.mask).long(),   # [H, W], 0 = background
                "index": index,
                "kind": SCENE_KINDS.index(sample.meta["kind"]),
            }
        return results


DATASET = HSIDataset
