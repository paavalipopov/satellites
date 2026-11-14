"""Synthetic HSI scenes from ``src.data_generation``, generated on the fly by the loader workers.

``data.name=hsi``. One ``HSIDataset`` holds the generator settings and the material
library; ``split()`` gives each loader its own dataset. Every scene is fixed by its seed:

    train   scene_seed(seed_train, run, epoch, index)    fresh every epoch and run
            (with pair_copies / max_one_copies: 2 x n_train items; items 2k and 2k+1 are scene k as generated and
            divided by one number: its max, or the mean / 99th percentile of its object pixels)
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
SCENE_KINDS = ("satellite", "shapes", "patches", "empty")  # a scene's "kind" is its index here


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
            "max_one_copies": False,  # train only: every scene appears twice, as generated and again divided by its
                                      # brightest value (brightest = 1). The label is the same; valid/test are untouched.
            "pair_copies": None,      # the same with another divisor: "max" (= max_one_copies), "mean" or "p99" = the mean or
                                      # the 99th percentile of the values of the scene's OBJECT pixels (all bands)
            "sim": asdict(SimConfig()),  # every generator setting, so config.yaml records them all
        }
    )


PAIR_STATS = ("max", "mean", "p99")


def copy_scale(cube, mask, stat):
    """The number the second copy of a scene is divided by. cube [bands, H, W] tensor, mask [H, W] (0 = background).
    max: the brightest value of the whole scene. mean / p99: the mean / 99th percentile of the values of the object
    pixels (all bands), which is what scripts/auto_scale.py takes from the object at test time (single_mean / single_p99);
    a scene without object pixels uses all of its pixels."""
    if stat == "max":
        value = cube.max()
    else:
        obj = mask > 0
        values = cube[:, obj] if bool(obj.any()) else cube.reshape(cube.shape[0], -1)
        value = values.mean() if stat == "mean" else torch.quantile(values.flatten().float(), 0.99)
    return value.clamp_min(1e-12)


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
            pairs = p.get("pair_copies") or ("max" if p.get("max_one_copies") else None)
            if pairs not in (None, *PAIR_STATS):
                raise ValueError(f"pair_copies: one of {', '.join(PAIR_STATS)} (or null), got {pairs!r}")
            return HSISplit(self, p.n_train, (p.seed_train, run), p.scene, fresh=True, pairs=pairs)
        if name == "valid":
            return HSISplit(self, p.n_valid, (p.seed_valid,), p.scene, fresh=False)
        return HSISplit(self, p.n_test, (p.seed_test,), p.tests[name], fresh=False)


class HSISplit(Dataset):
    """``size`` scenes; ``fresh`` adds the epoch to every seed.

    ``pairs`` ("max" | "mean" | "p99" | None): the split holds 2 x ``size`` items. Item 2k is scene k as generated,
    item 2k+1 is the very same scene (same seed, so the same pixels, noise and label) divided by one number (copy_scale):
    with "max" its maximum becomes 1, with "mean" / "p99" the mean / 99th percentile of its object pixels becomes 1.
    The label does not change. The model sees both brightness levels of every scene."""

    def __init__(self, parent, size, keys, scene, fresh, pairs=None):
        self.cfg, self.library = parent.cfg, parent.library
        self.size, self.keys, self.scene, self.fresh, self.pairs = size, tuple(keys), scene, fresh, pairs

    def __len__(self):
        return 2 * self.size if self.pairs else self.size

    def seed(self, epoch, index):
        return scene_seed(*self.keys, *([epoch] if self.fresh else []), index)

    def __getitem__(self, item):
        epoch, batch = item  # what SeededBatchSampler yields
        results = {}
        for item in map(int, batch):
            index, rescale = (item // 2, item % 2 == 1) if self.pairs else (item, False)
            sample = generate_sample(self.cfg, self.seed(epoch, index), self.library, scene=self.scene)
            cube = torch.from_numpy(sample.cube).float()          # [bands, H, W]
            label = torch.from_numpy(sample.mask).long()
            if rescale:
                cube = cube / copy_scale(cube, label, self.pairs)
            results[item] = {
                "input": cube,
                "label": torch.from_numpy(sample.mask).long(),   # [H, W], 0 = background
                "index": index,
                "kind": SCENE_KINDS.index(sample.meta["kind"]),
            }
        return results


DATASET = HSIDataset
