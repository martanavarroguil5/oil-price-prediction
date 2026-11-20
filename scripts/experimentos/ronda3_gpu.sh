#!/usr/bin/env bash
# Ronda 3 (docs/experimento-ronda3.md) en la GPU: cuatro carriles en paralelo.
# Uso: PY=/ruta/al/python scripts/experimentos/ronda3_gpu.sh   (logs en reports/experiments/ronda3/logs)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=src
LOG=reports/experiments/ronda3/logs
mkdir -p "$LOG"
oc() { "$PY" -m oilcast -c "configs/$1.yaml" "${@:2}"; }

lane_single() {  # S1: una red por horizonte, selección por la mediana
  oc ronda3_single tune -a "$1" && oc ronda3_single evaluate -a "$1"
}
lane_multi() {   # M0, M1 o M2 para las tres redes
  oc "$1" multi-tune && oc "$1" multi-evaluate
}

( lane_single rnn; oc ronda3_w1 extend-published ) > "$LOG/single_rnn.log" 2>&1 &
( lane_single lstm ) > "$LOG/single_lstm.log" 2>&1 &
( lane_single gru ) > "$LOG/single_gru.log" 2>&1 &
( lane_multi ronda3_pinball; lane_multi ronda3; lane_multi ronda3_productos ) > "$LOG/multi.log" 2>&1 &
wait
echo "ronda3 terminada $(date)" > "$LOG/DONE"
