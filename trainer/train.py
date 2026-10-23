import warnings

warnings.filterwarnings("ignore")

import csv
import gc
import glob
import json
import os
import time
from functools import partial

import hydra
import numpy as np
import torch
import yaml
from catalyst import dl, metrics
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import get_optimizer_momentum, load_checkpoint
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.datasets.base import EpochBatchSampler, collate_volumes
from src.registry import build_dataset, build_model, resolve_params
from src.utils import (
    ModelCheckpoint,
    SegmentationMeter,
    clear_rundir,
    plot_curves,
    setup_distributed_port,
    tee_stdout,
    world_size_from_env,
)

METRICS = {"loss", *SegmentationMeter.names}  # every loader metric; selection picks from these

setup_distributed_port()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CustomRunner(dl.Runner):
    """Per-pixel material classifier trainer, one run per runner.

    A run has its own stream of fresh training scenes and its own model init; every run
    shares the same valid and test sets.
    """

    def __init__(self, logdir: str, dataset, cfg):
        super().__init__()
        self._logdir = logdir
        self._moredir = os.path.join(logdir, "more")  # detail logs, out of the main view
        os.makedirs(self._moredir, exist_ok=True)
        self.dataset = dataset
        self.cfg = cfg
        # what catalyst writes to hparams.json and hands to wandb's run config
        self._hparams = OmegaConf.to_container(cfg, resolve=True)

        self.n_classes = cfg.runtime.n_classes

        self._started = time.time()
        self._split_sizes = {}

    def get_engine(self):
        # Pin the spawn/world size: Catalyst otherwise spawns one process per *physical*
        # GPU via device_count(). See world_size_from_env().
        n_gpus = world_size_from_env()
        if n_gpus > 1:
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": "nccl"},
                num_node_workers=n_gpus,
                world_size=n_gpus,
            )
        return dl.GPUEngine() if n_gpus == 1 else dl.CPUEngine()

    def get_loggers(self):
        loggers = {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

        wandb_cfg = self._hparams["wandb"]
        # wandb.project=null (default config) keeps it off
        if wandb_cfg["project"]:
            loggers["wandb"] = dl.WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"],
                name=wandb_cfg["name"] or self._hparams["experiment"]["name"],
                log_batch_metrics=False,
            )
        return loggers

    @property
    def num_epochs(self) -> int:
        return self.cfg.experiment.epochs

    @property
    def seed(self) -> int:
        """Base for Catalyst's own reseeding.

        Catalyst calls ``set_global_seed(self.seed + rank + epoch_step)`` in five places:
        setting up the loaders, the components and the callbacks, and at the start of
        every epoch and every loader. The rank term is what stops ranks drawing identical
        augmentations. Within one epoch the value does not change, so the per-loader call
        just rewinds the stream.
        """
        return self.cfg.experiment.global_seed

    def on_experiment_start(self, runner):
        torch.backends.cudnn.benchmark = self.cfg.experiment.cudnn_benchmark
        super().on_experiment_start(runner)
        # engine.prepare has sharded it; test runs once at the end, or also every epoch
        every_epoch = self.cfg.experiment.test_every_epoch
        self.test_loader = self.loaders["test"] if every_epoch else self.loaders.pop("test")
        if not self.cfg.runtime.resume_runner:
            return

        # === Resume routines ===
        # reopen CSVLoggers manually to avoid writing headers again
        if "csv" in self.loggers:  # only the main process has loggers
            for key in [*self.loaders, "_epoch_"]:
                path = os.path.join(self._logdir, f"{key}.csv")
                if os.path.isfile(path):
                    self.loggers["csv"].loggers[key] = open(path, "a")

    def get_loaders(self):
        experiment, runtime = self.cfg.experiment, self.cfg.runtime
        # fresh train scenes per epoch and run; valid and test fixed by their seeds
        splits = {
            key: self.dataset.split(key, run=runtime.run_idx) for key in ("train", "valid", "test")
        }
        self._split_sizes = {name: len(split) for name, split in splits.items()}
        print(
            f"[Run {runtime.run_idx}] "
            + " ".join(f"{k}={v}" for k, v in self._split_sizes.items())
        )

        if self.engine.is_main_process:
            # everything needed to reproduce this run, seeds included
            run_cfg = {
                "run_idx": runtime.run_idx,
                "seeds": self._seeds(),
                "world_size": self.engine.num_processes,
                "max_lr": runtime.max_lr,
                "batch_size_per_rank": experiment.batch_size,
                "epochs": experiment.epochs,
                "split_sizes": self._split_sizes,
                "config": self._hparams,
            }
            with open(os.path.join(self._moredir, "run_config.yaml"), "w") as handle:
                yaml.safe_dump(run_cfg, handle, indent=4, sort_keys=False)

        loaders = {}
        for key, subset in splits.items():
            loader_opts = self.cfg.loader["train" if key == "train" else "eval"]
            num_workers = loader_opts.num_workers
            # Plain sampler with the same seed on every rank. accelerate's
            # engine.prepare() already shards the DataLoader, so pre-sharding here would
            # shard twice and drop ~(1 - 1/world_size) of the data each epoch.
            kwargs = {
                "sampler": EpochBatchSampler(subset, experiment.batch_size),
                "collate_fn": collate_volumes,
                "pin_memory": True,
                "num_workers": num_workers,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = loader_opts.persistent_workers
                kwargs["prefetch_factor"] = loader_opts.prefetch_factor
            loader = DataLoader(subset, **kwargs)
            # the prefetch wrapper needs CUDA streams; prefetches=0 disables it on CPU
            prefetches = loader_opts.prefetches
            loaders[key] = (
                BatchPrefetchLoaderWrapper(loader, num_prefetches=prefetches)
                if prefetches > 0
                else loader
            )
        return loaders

    def _seeds(self):
        data = self.cfg.data.params
        return {
            "seed_train": data.seed_train,
            "seed_valid": data.seed_valid,
            "seed_test": data.seed_test,
            "global_seed": self.cfg.experiment.global_seed,
            "model_init_seed": self._init_seed(),
        }

    def _init_seed(self):
        init_seed = self.cfg.model.init_seed
        return None if init_seed is None else init_seed + self.cfg.runtime.run_idx

    def get_model(self):
        # fixed init, with the global RNG stream restored so it is not perturbed
        init_seed = self._init_seed()
        if init_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(init_seed)

        model = build_model(self.cfg)
        if self.cfg.experiment.paths.init_weights:  # start from saved weights (not a resume)
            model.load_state_dict(load_checkpoint(self.cfg.experiment.paths.init_weights))
        if self.cfg.experiment.channels_last:  # before DDP wraps it
            model = model.to(memory_format=torch.channels_last)
        if self.cfg.experiment.compile:  # in place, keeps state_dict keys
            model.compile()

        if init_seed is not None:
            torch.set_rng_state(rng_state)
            print(f"Model initialized with fixed seed {init_seed}")
        return model

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, model):
        # Note: OneCycleLR overwrites the LR defined here
        return torch.optim.Adam(model.parameters(), lr=self.cfg.optimizer.base_lr)

    def get_scheduler(self, optimizer):
        return OneCycleLR(
            optimizer,
            max_lr=self.cfg.runtime.max_lr,
            div_factor=100,
            pct_start=0.1,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )

    def get_callbacks(self):
        return {
            # best weights per selection metric, and the state for a mid-run resume
            "checkpoint": ModelCheckpoint(
                self._logdir,
                self.cfg.experiment.selection,
                save_state=self.cfg.experiment.save_state,
                resume_from=self.cfg.runtime.resume_runner or None,
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)

        # initialize metrics
        self.meters = {
            "loss": metrics.AdditiveMetric(),
            "pixels": SegmentationMeter(self.n_classes, self.engine.device),
        }
        # (index, predicted mask, target mask) per batch, saved for test only
        self.test_preds = [] if self.loader_key == "test" else None

    def on_loader_end(self, runner):
        # compute the loader's metrics
        self.loader_metrics["loss"] = self.meters["loss"].compute()[0]
        # counts summed over ranks first, so the metrics see every pixel
        reduce = partial(self.engine.reduce, reduction="sum")
        self.loader_metrics.update(self.meters["pixels"].compute(reduce))

        # predicted masks, merged across ranks in on_experiment_end
        if self.test_preds is not None:
            index, pred, target = (torch.cat(column).numpy() for column in zip(*self.test_preds))
            path = os.path.join(self._moredir, f"test_preds_rank{self.engine.process_index}.npz")
            np.savez_compressed(path, index=index, pred=pred, target=target)
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        self.epoch_metrics["_epoch_"]["lr"] = self.optimizer.param_groups[0]["lr"]
        self.epoch_metrics["_epoch_"]["momentum"] = get_optimizer_momentum(self.optimizer)
        super().on_epoch_end(runner)

    def on_experiment_end(self, runner):
        # load the main selection's best weights and run them on test
        checkpoint, main = self.callbacks["checkpoint"], self.cfg.experiment.selection[0]
        best = checkpoint.best[main]
        self.engine.wait_for_everyone()  # the main process has finished writing them
        self.engine.unwrap_model(self.model).load_state_dict(load_checkpoint(checkpoint.path(main)))

        # prepare for test
        # epoch_step stamps the test logs, so point it at the epoch these
        # weights come from, then put the true epoch count back for epochs_run
        epoch_buffer, self.epoch_step = self.epoch_step, best["epoch"]
        self.loader_key, self.loader = "test", self.test_loader

        # run test
        self._run_event("on_loader_start")
        self._run_loader()
        self._run_event("on_loader_end")

        # save logs and restore things
        self.engine.wait_for_everyone()  # every rank has saved its test predictions
        self.epoch_step = epoch_buffer
        best["metrics"]["test"] = {
            k: float(v) for k, v in sorted(self.loader_metrics.items())
        }

        # final logging and cleanup
        if self.engine.is_main_process:
            # combine the ranks' test predictions into one file, one row per test scene
            parts = [np.load(path) for path in
                     sorted(glob.glob(os.path.join(self._moredir, "test_preds_rank*.npz")))]
            preds = {key: np.concatenate([part[key] for part in parts])
                     for key in ("index", "pred", "target")}
            _, keep = np.unique(preds["index"], return_index=True)  # DDP pads with repeats
            np.savez_compressed(os.path.join(self._logdir, "test_predictions.npz"),
                                epoch=best["epoch"], **{k: v[keep] for k, v in preds.items()})

            # save run summary
            summary = {
                "run_idx": self.cfg.runtime.run_idx,
                "selected_on": f"{'min' if main == 'loss' else 'max'} valid/{main}",
                "best_epoch": best["epoch"],
                "epochs_run": int(self.epoch_step),
                "split_sizes": self._split_sizes,
                "metrics": best["metrics"],
                "bests": checkpoint.best,  # every selection; test scores only with test_every_epoch
                "seeds": self._seeds(),
                "world_size": self.engine.num_processes,
                "batch_size_per_rank": self.cfg.experiment.batch_size,
                "max_lr": self.cfg.runtime.max_lr,
                "minutes": round((time.time() - self._started) / 60, 2),
            }
            with open(os.path.join(self._logdir, "run_summary.json"), "w") as handle:
                json.dump(summary, handle, indent=4)

            # training curves and run info
            plot_curves([self._logdir], self._hparams, os.path.join(self._logdir, "curves.png"))
            print(
                f"[Run {self.cfg.runtime.run_idx}] best epoch {best['epoch']} "
                f"(valid/{main}={best['score']:.4f})"
            )
        super().on_experiment_end(runner)

    def handle_batch(self, batch):
        # target: [B,H,W] masks; index: each scene's position in its split, which with
        # the split's seed gives the scene's generator seed
        sample, target, index = batch
        if self.cfg.experiment.channels_last:
            sample = sample.contiguous(memory_format=torch.channels_last)
        autocast = torch.autocast(
            self.engine.device.type, dtype=torch.bfloat16, enabled=self.cfg.experiment.bf16
        )

        if self.model.training:
            with autocast:
                y_hat = self.model(sample)  # .forward() would bypass compile
                loss = self.criterion(y_hat, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            with torch.no_grad(), autocast:
                y_hat = self.model(sample)
                loss = self.criterion(y_hat, target)
        y_hat = y_hat.float()  # bf16 logits would coarsen metrics and preds

        with torch.no_grad():
            pred = y_hat.argmax(1)
            accuracy = (pred == target).float().mean()
            self.meters["pixels"].update(pred, target)
            if self.test_preds is not None:
                self.test_preds.append((index.cpu(), pred.byte().cpu(), target.byte().cpu()))

        # batch_metrics feeds the tqdm bar; the loader values are saved in the meters
        self.batch_metrics.update({"loss": loss, "accuracy": accuracy})
        self.meters["loss"].update(loss.item(), self.batch_size)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """ Configure experiment, train the requested runs and roll up every finished one."""
    # ===== Initial setup =====
    logdir = os.path.join(cfg.experiment.paths.logdir, cfg.experiment.name)
    os.makedirs(logdir, exist_ok=True)
    experiment = cfg.experiment
    run_idx = experiment.run_idx
    run_ids = list(run_idx) if OmegaConf.is_list(run_idx) else [run_idx]
    if cfg.stream_log:  # one log per job, so parallel jobs don't interleave
        tee_stdout(os.path.join(logdir, "stdout_run" + "_".join(map(str, run_ids)) + ".log"))

    # validate config
    for ok, message in (
        (experiment.epochs >= 1, "epochs must be at least 1"),
        (len(experiment.selection) > 0 and set(experiment.selection) <= METRICS,
         f"selection: a list of {', '.join(sorted(METRICS))}"),
        (run_ids and all(isinstance(i, int) and i >= 0 for i in run_ids),
         "run_idx must be a non-negative int or a non-empty list of them"),
    ):
        if not ok:
            raise ValueError(f"experiment.{message}")

    # ===== Data setup =====
    # build dataset and get data params from it
    dataset = build_dataset(cfg)
    cfg.runtime.n_channels = int(dataset.n_channels)
    cfg.runtime.n_classes = int(dataset.n_classes)
    data = cfg.data.params
    print(
        f"[Data] {cfg.data.name}: {data.n_train} fresh train scenes per epoch, "
        f"valid {data.n_valid}, test {data.n_test}, {cfg.runtime.n_channels} channel(s), "
        f"{cfg.runtime.n_classes} classes"
    )

    resolve_params("model", cfg)

    # ===== Env-dependent setup =====
    # get the world size (number of GPUs for DDP)
    cfg.runtime.world_size = max(1, world_size_from_env())

    # max_lr is the peak OneCycleLR climbs to, scaled to world size according to Malladi, 2022
    # https://doi.org/10.48550/arXiv.2205.10287
    cfg.runtime.max_lr = cfg.optimizer.base_lr
    if cfg.optimizer.scale_lr:
        cfg.runtime.max_lr *= cfg.runtime.world_size**0.5
    print(
        f"[LR] max_lr={cfg.runtime.max_lr:g} (base={cfg.optimizer.base_lr}, "
        f"world_size={cfg.runtime.world_size})"
    )

    # ===== Resumed experiment routines =====
    # recovers the completed runs and continues from there
    resume = cfg.experiment.resume
    config_path = os.path.join(logdir, "config.yaml")
    if resume:
        done = [
            i for i in run_ids
            if os.path.isfile(os.path.join(logdir, f"run_{i}", "run_summary.json"))
        ]
        print(f"[Resume] on; runs already complete: {done}")

        # a resumed run keeps the old schedule, splits and optimizer state
        for key in ("experiment.epochs", "experiment.batch_size", "data.params.n_train",
                    "data.params.seed_train", "data.params.seed_valid", "data.params.seed_test",
                    "experiment.global_seed", "runtime.world_size"):
            was = OmegaConf.select(OmegaConf.load(config_path), key)
            if was is not None and was != OmegaConf.select(cfg, key):
                raise ValueError(
                    f"resume: {key} was {was}, now {OmegaConf.select(cfg, key)}; "
                    "this will mess with OneCycleLR's schedule; to continue from "
                    "this model instead, point experiment.paths.init_weights at its checkpoint"
                )

    # pre-flight maintenance: save the config
    with open(config_path, "w") as handle:
        yaml.safe_dump(
            OmegaConf.to_container(cfg, resolve=True), handle, indent=4, sort_keys=False
        )

    # ===== Run loop =====
    for position, run_idx in enumerate(run_ids, start=1):
        rundir = os.path.join(logdir, f"run_{run_idx}")
        if resume and os.path.isfile(os.path.join(rundir, "run_summary.json")):
            print(f"[Resume] run {run_idx} ({position}/{len(run_ids)}) done, skipping")
            continue
        print(f"Starting run {run_idx} ({position}/{len(run_ids)})")
        os.makedirs(rundir, exist_ok=True)

        # pick an interrupted run up where it stopped, or start it fresh
        state = os.path.join(rundir, "state.last.pth")
        resume_runner = state if resume and os.path.isfile(state) else ""
        if resume_runner:
            print(f"[Resume] continuing run {run_idx} from state.last.pth")
        else:  # drop what an earlier run left here
            clear_rundir(rundir)

        # TRAIN THE RUN
        cfg.runtime.run_idx = run_idx
        cfg.runtime.resume_runner = resume_runner
        runner = CustomRunner(logdir=rundir, dataset=dataset, cfg=cfg)
        runner.run()

        # GC step; the run is done, so its resume state can go
        del runner
        torch.cuda.empty_cache()
        gc.collect()
        if os.path.isfile(state):
            os.remove(state)
        if not os.path.isfile(os.path.join(rundir, "run_summary.json")):
            print(f"[Warn] run {run_idx} produced no run_summary.json")

    # ===== Summary over every finished run, from this job or others =====
    rows = []
    for best_path in glob.glob(os.path.join(logdir, "run_*", "run_summary.json")):
        with open(best_path) as handle:
            best = json.load(handle)

        valid, test = best["metrics"].get("valid", {}), best["metrics"].get("test", {})
        rows.append(
            {
                "run": best["run_idx"],
                **{f"n_{name}": size for name, size in best["split_sizes"].items()},
                "epochs_run": best["epochs_run"],
                "best_epoch": best["best_epoch"],
                **{f"best_epoch@{m}": b["epoch"] for m, b in best.get("bests", {}).items()},
                **{f"valid_{key}": value for key, value in valid.items()},
                **{f"test_{key}": value for key, value in test.items()},
                **{f"test_{k}@{m}": v for m, b in best.get("bests", {}).items()
                   for k, v in b["metrics"].get("test", {}).items()},
                **best["seeds"],
                "world_size": best["world_size"],
                "batch_size_per_rank": best["batch_size_per_rank"],
                "max_lr": best["max_lr"],
                "minutes": best["minutes"],
            }
        )
    if not rows:
        print("[Runs] no finished run yet; no summary")
        return
    rows.sort(key=lambda row: row["run"])
    with open(os.path.join(logdir, "runs.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list({k: 0 for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"n_runs": len(rows), "runs": [row["run"] for row in rows]}
    for key in [c for c in rows[0] if c.startswith(("valid_", "test_"))]:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        # 95% t interval on the mean across runs
        mean, n = float(np.mean(values)), len(values)
        half = (
            stats.t.ppf(0.975, n - 1) * float(np.std(values, ddof=1)) / np.sqrt(n)
            if n > 1
            else float("nan")
        )
        summary[key] = {
            "mean": mean,
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "ci95": [mean - half, mean + half],
        }
        if key.startswith("test"):
            print(f"[Runs] {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")

    with open(os.path.join(logdir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=4)

    # every run's curves in one figure
    rundirs = [os.path.join(logdir, f"run_{row['run']}") for row in rows]
    plot_curves(rundirs, OmegaConf.to_container(cfg, resolve=True),
                os.path.join(logdir, "curves.png"))


if __name__ == "__main__":
    main()
