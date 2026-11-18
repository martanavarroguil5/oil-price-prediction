"""Training-time benchmark for the comparison table.

Run sequentially with nothing else on the device, so times are comparable: each
(architecture, horizon) is trained once with its tuned hyperparameters and fixed epochs
on the largest pre-test training set (everything realised before the test start).
"""

from __future__ import annotations

import logging

import pandas as pd

from oilcast.config import Config
from oilcast.features.windows import Panel
from oilcast.models.recurrent import count_parameters
from oilcast.paths import Paths
from oilcast.training.splits import purged_train
from oilcast.training.trainer import fit, prepare_data
from oilcast.training.tune import load_best
from oilcast.utils import get_device

log = logging.getLogger(__name__)


def run_timing(cfg: Config, paths: Paths) -> pd.DataFrame:
    device = get_device()
    panel = Panel.load(paths, cfg)
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    rows = []
    warmed = False
    for h in cfg["targets"]["horizons"]:
        train = purged_train(panel, h, test_start)
        data = prepare_data(panel, h, train, cfg, device)
        for arch in cfg["architectures"]:
            hp, epochs = load_best(paths, arch, h)
            if not warmed:  # first CUDA/cuDNN call pays one-off initialisation costs
                fit(arch, hp, data, train, cfg, epochs=1, seed=cfg["seed"])
                warmed = True
            res = fit(arch, hp, data, train, cfg, epochs=epochs, seed=cfg["seed"])
            rows.append(
                {
                    "arch": arch,
                    "horizon": h,
                    "train_seconds": res.seconds,
                    "epochs": epochs,
                    "seconds_per_epoch": res.seconds / epochs,
                    "n_train": len(train),
                    "n_params": count_parameters(res.model),
                    "device": str(device),
                    "device_name": "GPU" if device.type != "cpu" else "CPU",
                    **hp.to_dict(),
                }
            )
            log.info(
                "timing %s h=%d: %.1fs (%d epochs, %d params)",
                arch,
                h,
                res.seconds,
                epochs,
                rows[-1]["n_params"],
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results / "timing.csv", index=False)
    return out
