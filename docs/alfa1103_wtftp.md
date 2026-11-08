# ALFA 1103 roll-injection WTFTP experiment

Historical 18-input/18-output run only. The current 17-input/roll-only runner is documented in [alfa1103_cross_roll_wtftp.md](alfa1103_cross_roll_wtftp.md). The commands below describe the historical setting and are not compatible with the current `run.py` defaults.

This experiment uses the original 9-to-1 forecasting `models/WTFTP.py` interface, not the separate 96-point reconstruction adaptation in `models/wtftp_reconstruction.py`.

- `datasets/alfa1103/train.csv`: 6,679 normal training rows.
- `datasets/alfa1103/val.csv`: 2,227 validation rows, assumed normal for threshold calibration.
- `datasets/alfa1103/偏差为8-test.csv`: 2,225 test rows, with roll shifted by +8 for the last 1,200 rows. `原始test.csv` is the unmodified comparison file and is not used for training or scoring.
- `datasets/alfa1103/label.csv`: 1,025 zeros then 1,200 ones. Labels are used only for final evaluation.

The 18 channels are standardized with per-channel mean and population standard deviation from **train only**. A window of 9 observations forecasts the next point; the 10-point training target supervises the Haar low/high coefficients. The best model is selected by validation next-step MSE across all channels. For anomaly detection, the primary score is squared prediction error on standardized `field.measured.roll`. The threshold is the 99th percentile of validation roll scores, with `score > threshold` classified as anomalous. No test labels, test scores or point adjustment are used in calibration. Scores for original row indices 0 through 8 are undefined and excluded from metrics; `scores.npz` contains `row_index` for the 2,216 scored points.

Small local CPU check (not a meaningful model evaluation):

```bash
/home/wrj/miniconda3/envs/ltsm/bin/python run.py --device cpu --setting WTFTP_smoke --d_model 16 --e_layers 1 --train_epochs 1 --max_train_batches 2
```

For the full run, execute on `h100` in `/workspace/code/uavad` using its separate `ltsm` environment:

```bash
/opt/conda/envs/ltsm/bin/python run.py --device cuda --gpu 1 --setting WTFTP_alfa1103_roll8_seed2025
/opt/conda/envs/ltsm/bin/python run.py --is_training 0 --device cuda --gpu 1 --setting WTFTP_alfa1103_roll8_seed2025
```

The first command writes `checkpoints/<setting>/checkpoint.pth` and `metadata.json` (normalization, validation threshold, training-file hashes), then `results/<setting>/scores.npz` and `metrics.json`. Test-only mode verifies the checkpoint configuration, train/validation files, feature order and train normalization before scoring. The second command re-evaluates without retraining. `scores.npz` also includes the all-channel mean error, forecast, observed target, labels and binary decisions.

This is a single time-ordered split from one series, not a cross-flight generalization benchmark. A constant roll offset can become predictable from injected history; report threshold-free AUROC/AUPRC alongside recall at the validation-calibrated threshold, and do not interpret a one-epoch smoke run as a result.

## Default WTFTP run (2026-11-03)

Ran on H100 GPU 1 with seed 2025, embedding 64, four encoder layers, one decoder layer, batch size 64, Adam learning rate 0.001, 20 epochs maximum and patience 5. The best validation next-step MSE was 0.113131 at epoch 18. Test-only checkpoint reload reproduced the same metrics.

| Metric | Value |
| --- | ---: |
| Validation roll-score 99th percentile threshold | 0.594992 |
| Test AUROC / AUPRC | 0.758411 / 0.747029 |
| Precision / Recall / F1 at threshold | 0.481481 / 0.010833 / 0.021190 |
| TP / FP / FN / TN | 13 / 14 / 1187 / 1002 |
| Scored points / unscored prefix | 2216 / 9 |

The training roll standard deviation is 25.585 degrees. The injected +8 degrees equals 0.313 training standard deviations, whereas the calibrated squared-error threshold corresponds to an absolute forecast error of about 19.74 degrees. The anomaly median roll score is 0.0646; the mean score over the last 20 anomalous points is 0.0062. Thus the model ranks many anomalies above normal points but almost never crosses this validation-calibrated threshold. These are results from this one split and seed, not a robustness or generalization estimate.
