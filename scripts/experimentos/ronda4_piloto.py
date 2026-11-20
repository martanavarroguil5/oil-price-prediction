"""Ronda 4, piloto solo con desarrollo: W1 (hiperparámetros publicados) + oro_crudo [+ atajo].

Uso: python scripts/experimentos/ronda4_piloto.py <variante> <red> <h>
Variantes: P1 = W1 + oro_crudo, P2 = P1 + atajo, P3 = solo oro_crudo, P4 = P3 + atajo.
Escribe reports/experiments/ronda4/piloto/<variant>/predictions/<arch>_h<h>.csv con las
predicciones de los reentrenos de calibración (2008-2020). No genera nada del test.
"""

import copy
import sys

import pandas as pd
import torch

from oilcast.config import load_config
from oilcast.features.windows import Panel
from oilcast.paths import get_paths
from oilcast.training.evaluate import _frame, quantile_columns
from oilcast.training.multitask import schedule
from oilcast.training.splits import RefitBlock, origins_between, purged_train
from oilcast.training.trainer import fit, predict_scaled, prepare_data
from oilcast.training.tune import load_best
from oilcast.utils import get_device

variant, arch, h = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = copy.deepcopy(load_config("configs/ronda3_w1.yaml"))
cfg["features"]["valuation"] = True
cfg.setdefault("model", {})["highway"] = variant in ("P2", "P4")
if variant in ("P3", "P4"):
    cfg["features"]["only"] = ["oro_crudo"]
torch.set_num_threads(2)
device = get_device()
paths = get_paths(cfg)
panel = Panel.load(paths, cfg)
hp, epochs = load_best(get_paths({"paths": {"results_dir": "reports/results"}}), arch, h)
qcols = quantile_columns(cfg["targets"]["quantiles"])
frames = []
for start, end, phase in schedule(panel, cfg):
    if phase != "calibration":
        continue
    train = purged_train(panel, h, start)
    block = RefitBlock(start, end, phase, train, origins_between(panel, h, start, end))
    data = prepare_data(panel, h, train, cfg, device)
    for seed in cfg["evaluation"]["seeds"]:
        res = fit(arch, hp, data, train, cfg, epochs=epochs, seed=seed)
        q = predict_scaled(res.model, data, block.predict) * data.y_scale
        df = _frame(panel, h, block, q, qcols)
        df.insert(4, "seed", seed)
        frames.append(df)
    print(variant, arch, h, start.date(), len(train), flush=True)
out = get_paths({"paths": {"results_dir": f"reports/experiments/ronda4/piloto/{variant}"}})
path = out.predictions_csv(arch, h)
path.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.6g")
