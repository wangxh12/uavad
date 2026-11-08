# ALFA 1103 WTFTP: non-roll sensors to roll

## Protocol

Use `datasets/alfa1103` (train 6,679, validation 2,227, injected test 2,225 rows). The model input is the previous 9 rows of the 17 columns **excluding** `field.measured.roll`. It predicts the next row's roll only. The train/validation target window contains 10 roll observations for the WTFTP Haar low/high coefficient loss; this window is passed to the loss, **never** to the model encoder. Target roll normalization and the 17 input normalizations use training-only statistics. Validation selects the checkpoint by next-step roll MSE; its 99th percentile of squared standardized roll errors sets the threshold. Test labels are used only for evaluation. The initial 9 test rows are unscored; scores align to original zero-based row indices 9 through 2224. No point adjustment or score smoothing is used for metrics (the figure's 11-point trailing median is display-only).

This differs from the earlier 18-input/18-output experiment in [alfa1103_wtftp.md](alfa1103_wtftp.md). The old checkpoint and scores remain under `WTFTP_alfa1103_roll8_seed2025`.

## Run

Local and remote `ltsm` environments are independent. After the small local and H100 smoke tests passed, the full run was executed on `h100` GPU 1 in `/workspace/code/uavad` (2026-11-03):

```bash
/opt/conda/envs/ltsm/bin/python run.py --device cuda --gpu 1 --setting WTFTP_alfa1103_cross_roll_seed2025
/opt/conda/envs/ltsm/bin/python run.py --is_training 0 --device cuda --gpu 1 --setting WTFTP_alfa1103_cross_roll_seed2025
/opt/conda/envs/ltsm/bin/python -m scripts.visualize_wtftp_alfa1103
```

Default WTFTP hyperparameters: embedding 64, four encoder layers, one decoder layer, batch size 64, Adam learning rate 0.001, seed 2025, maximum 20 epochs and patience 5. Training stopped after epoch 18; best checkpoint was epoch 13, validation roll MSE 0.031673. Test-only reload reproduced all saved metrics.

| Metric | 17-to-roll run | Historical 18-to-18 run |
| --- | ---: | ---: |
| Validation threshold | 0.192910 | 0.594992 |
| AUROC | 0.862880 | 0.758411 |
| AUPRC | 0.826674 | 0.747029 |
| Precision | 0.860417 | 0.481481 |
| Recall | 0.344167 | 0.010833 |
| F1 | 0.491667 | 0.021190 |
| TP / FP / FN / TN | 413 / 67 / 787 / 949 | 13 / 14 / 1187 / 1002 |

The new model catches substantially more injected points but raises more false alarms. This one-split, one-seed comparison changes both the input/target definition and the training objective; it is not an isolated architectural ablation. The normal test segment's 67/1016 alarm rate exceeds the nominal 1% validation calibration rate, suggesting validation-to-test score shift. The roll target is never an encoder input, including after injection onset.

Artifacts: `checkpoints/WTFTP_alfa1103_cross_roll_seed2025/` contains the model and metadata (including input column list and training file hashes); `results/WTFTP_alfa1103_cross_roll_seed2025/` contains `scores.npz`, `metrics.json` and [threshold.png](../results/WTFTP_alfa1103_cross_roll_seed2025/threshold.png).

## Min-max trial (2026-11-03)

Only the normalization changed: each input channel and the roll target use their own training-set minimum and range, with a unit divisor for constant channels. Validation/test values outside the training range are **not clipped**. The 17-to-1 model, seed, batch size, learning rate, threshold quantile and epoch limit are unchanged. The run used H100 GPU 1:

```bash
/opt/conda/envs/ltsm/bin/python run.py --device cuda --gpu 1 --normalization minmax --setting WTFTP_alfa1103_cross_roll_minmax_seed2025
/opt/conda/envs/ltsm/bin/python run.py --is_training 0 --device cuda --gpu 1 --normalization minmax --setting WTFTP_alfa1103_cross_roll_minmax_seed2025
/opt/conda/envs/ltsm/bin/python -m scripts.visualize_wtftp_alfa1103 --run-dir results/WTFTP_alfa1103_cross_roll_minmax_seed2025
```

Training stopped after epoch 19, with the best checkpoint at epoch 14 (validation roll MSE 0.003211 in min-max units). Checkpoint reload reproduced the saved metrics. The raw squared-score thresholds are not directly comparable across normalizations.

| Metric | Standard | Min-max |
| --- | ---: | ---: |
| Validation threshold | 0.192910 | 0.021689 |
| AUROC / AUPRC | 0.862880 / 0.826674 | 0.686593 / 0.620064 |
| Precision / Recall / F1 | 0.860417 / 0.344167 / 0.491667 | 0.355932 / 0.035000 / 0.063733 |
| TP / FP / FN / TN | 413 / 67 / 787 / 949 | 42 / 76 / 1158 / 940 |

The min-max training roll range is 103.71 degrees, so the injected +8 degrees is 0.077 of that range. Its squared-error threshold corresponds to about 15.27 degrees of absolute roll prediction error. The min-max run performed worse by both ranking and fixed-threshold detection in this one-split, one-seed trial. Its [threshold plot](../results/WTFTP_alfa1103_cross_roll_minmax_seed2025/threshold.png), [scores](../results/WTFTP_alfa1103_cross_roll_minmax_seed2025/scores.npz) and checkpoint are kept separately from the standard run.
