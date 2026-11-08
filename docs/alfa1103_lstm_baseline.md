# ALFA 1103 plain LSTM baseline

## Protocol

The model in `models/LSTM.py` receives 9 previous rows of the 17 non-roll sensors and predicts the next row's roll. It is a four-layer, 64-hidden-unit LSTM with a linear output head (121,153 parameters), trained with next-roll MSE. Neither historical nor current roll enters the model. The shared data provider scales inputs and target using **train only**; validation selects the best checkpoint by next-roll MSE and supplies the 99th-percentile threshold on squared normalized roll errors. The first 9 test rows have no forecast. Labels are used only for evaluation; no score smoothing or point adjustment is applied to metrics. See [WTFTP protocol](alfa1103_cross_roll_wtftp.md) for the dataset and injection details.

Local and H100 smoke checks passed first. Full runs on `h100` GPU 1 in `/workspace/code/uavad`, seed 2025, batch size 64, Adam learning rate 0.001, at most 20 epochs and patience 5:

```bash
/opt/conda/envs/ltsm/bin/python run.py --model LSTM --device cuda --gpu 1
/opt/conda/envs/ltsm/bin/python run.py --model LSTM --normalization minmax --device cuda --gpu 1
```

Use the same commands with `--is_training 0` to reload checkpoints without training. Both reloads reproduced the saved metrics. The standard run stopped after epoch 11, best epoch 6 (validation MSE 0.034367); min-max completed 20 epochs, best epoch 17 (validation MSE 0.002169). Raw threshold and MSE magnitudes cannot be compared across normalization modes.

| Metric | LSTM standard | WTFTP standard | LSTM min-max | WTFTP min-max |
| --- | ---: | ---: | ---: | ---: |
| Threshold | 0.254464 | 0.192910 | 0.018660 | 0.021689 |
| AUROC | **0.907897** | 0.862880 | **0.911406** | 0.686593 |
| AUPRC | **0.874609** | 0.826674 | **0.889574** | 0.620064 |
| Precision | 0.846154 | 0.860417 | 0.897059 | 0.355932 |
| Recall | 0.110000 | **0.344167** | **0.101667** | 0.035000 |
| F1 | 0.194690 | **0.491667** | **0.182635** | 0.063733 |
| TP / FP / FN / TN | 132 / 24 / 1068 / 992 | 413 / 67 / 787 / 949 | 122 / 14 / 1078 / 1002 | 42 / 76 / 1158 / 940 |

LSTM ranks anomalous points better on this split, but the validation-calibrated 99th-percentile threshold gives low recall. Standard WTFTP detects more abnormal points at this fixed threshold, at the cost of more false alarms. WTFTP uses a Haar coefficient objective and 220,274 parameters; therefore this comparison is a practical baseline, not a parameter-matched architectural ablation. Only one split and one seed were evaluated.

Artifacts: `results/LSTM_alfa1103_cross_roll_seed2025/` and `results/LSTM_alfa1103_cross_roll_minmax_seed2025/` each contain `metrics.json`, `scores.npz` and a threshold plot. The corresponding checkpoints and scaler metadata are under `checkpoints/`. The plots are [standard](../results/LSTM_alfa1103_cross_roll_seed2025/threshold.png) and [min-max](../results/LSTM_alfa1103_cross_roll_minmax_seed2025/threshold.png); the trailing-median line is display-only.
