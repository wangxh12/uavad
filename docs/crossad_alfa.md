# CrossAD / ALFA 18-feature baseline

This is an independent implementation of Li et al., *CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling*, NeurIPS 2025. The model and score follow the authors' published architecture; their source code is not vendored. This is a new ALFA experiment, not a reproduction of the paper's seven-dataset numeric table.

The input is `data/raw/alfa_20hz_l_18f/*.csv`: 18 fixed numeric channels, `time_s`, and `label`. The file with only `-1` labels is excluded. A complete timestamp group belongs to exactly one split. The manifest and training-normal-only scaler are stored under `data/processed/alfa_crossad/`; this directory and result directories are ignored by Git. Raw data are never modified. The fixed split is in `data/alfa.py` and recorded in the manifest. Normal prefixes of training fault flights are allowed. Validation normal prefixes select the checkpoint and calibrate a 99th-percentile point-score threshold; no test labels are used for model selection, scaling, or thresholding.

From `/home/wrj/workspace/uavad` with the local `ltsm` environment:

```bash
python -m scripts.crossad_alfa prepare
python -m pytest tests/test_alfa.py tests/test_crossad.py -q
python -m scripts.crossad_alfa train --run-dir results/crossad_alfa/smoke --batch-size 1 --epochs 1 --max-batches 2 --device cuda
python -m scripts.crossad_alfa train --run-dir results/crossad_alfa/smoke --batch-size 1 --epochs 2 --max-batches 2 --device cuda --resume
```

After the small-scale check, on `h100` in `/workspace/code/uavad` and its independent `ltsm` environment:

```bash
python -m scripts.crossad_alfa train --run-dir results/crossad_alfa/main --epochs 20 --batch-size 128 --device cuda:1 --evaluate-after
```

`latest.pt` stores optimizer, context bank, random states, and epoch for `--resume`; `best.pt` is the validation-selected checkpoint. `evaluate` can be rerun with `--run-dir ... --device cuda`; it writes `metrics.json`, `flights.csv`, and per-flight `.npz` files with timestamps, labels, per-feature scores, aggregate scores, and decisions. The 192-point windows use future samples within each window and are **offline**, not an online early-warning protocol. End-anchored windows cover the tails of flights; duplicate point scores are averaged. The evaluation has no point adjustment and reports unadjusted point metrics, event delays, and normal-flight false alarms. These results must not be compared numerically with the paper's unrelated datasets.

## Completed run (2026-10-25)

One seed (2025), all 20 epochs, H100 `cuda:1`. The validation-selected threshold is 3.2974. On 16 held-out flights (30,901 points), unadjusted point AUROC is 0.5810, AUPRC 0.2959, precision 0.3665, recall 0.1863, and F1 0.2470. Seven of 13 fault flights have an alarm after onset; two of three normal flights have a false alarm. Rudder faults are much easier under this split, while engine and aileron recall is low. The three normal test flights are too few to estimate a deployment false-alarm rate. This run is a baseline, not evidence of cross-platform or online performance.

The full report and scores are in `results/crossad_alfa/main/`. `train_log.csv` records the console loss values to six decimals; `provenance.json` records source/data hashes and the reference commit because the working tree was not committed during training. The test was rerun from `best.pt` and produced identical metrics.
