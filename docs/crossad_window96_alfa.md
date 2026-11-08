# CrossAD 96-point ALFA comparison (2026-10-27)

The original channel-independent CrossAD was retrained on the same z-score
processed ALFA split with `--window-size 96`, seed 2025, batch size 128, up to
20 epochs on `h100` GPU 1. All 20 epochs completed; the last epoch had the
lowest normal-validation loss (0.070205). The six time scales became
`[3, 6, 12, 24, 48, 96]` with patch counts `[1, 1, 2, 4, 8, 16]`.
Train windows still have stride 1; evaluation uses non-overlapping 96-point
windows with a tail-covering final window. It is still offline detection.

```bash
python -m scripts.crossad_alfa train --model crossad --window-size 96 \
  --processed-dir data/processed/alfa_crossad \
  --run-dir results/crossad_alfa/original_w96 \
  --epochs 20 --batch-size 128 --seed 2025 --device cuda:1 --evaluate-after
python -m scripts.visualize_crossad --run-dir results/crossad_alfa/original_w96
python -m scripts.compare_crossad_windows --baseline results/crossad_alfa/main \
  --candidate results/crossad_alfa/original_w96 \
  --output results/crossad_alfa/original_w96/comparison_192_vs_96.csv
```

| Window | AUROC | AUPRC | Precision | Recall | F1 | Fault flights detected | Normal flights with alarms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 0.5810 | 0.2959 | 0.3665 | 0.1863 | 0.2470 | 7/13 | 2/3 |
| 96 | 0.5776 | 0.2976 | 0.3543 | 0.1675 | 0.2275 | 7/13 | 2/3 |

The 96-point validation threshold is 2.78836, versus 3.29736 for 192 points;
these scores have different window-dependent distributions and should not be
compared directly across models. Among the 13 fault flights, 96 points improved
within-flight AUROC on 5, AUPRC on 6, and fixed-threshold F1 on 3. One engine
flight dropped from AUROC 0.824 to 0.679, while some aileron flights gained
ranking accuracy without useful fixed-threshold detections. All three rudder
flights remained strong, but the first dropped from F1 0.867 to 0.809.
`comparison_192_vs_96.csv` uses each run's saved point scores and verifies
identical labels and timestamps; the 192-point `flights.csv` did not include
per-flight AUROC. Both experiments score all 30,901 test points. This is a
single-seed comparison, not evidence that one length generalizes better.

The new run's `config.json`, `train.log`, `best.pt`, `metrics.json`, `flights.csv`,
per-flight scores, and 16 individual plots are under
`results/crossad_alfa/original_w96/`. At training time, SHA-256 was
`8662401f9acdafdaa06ea25da71f390c00fd605ad2fc20c5e29b78e37af76ba4`
for `scripts/crossad_alfa.py`,
`04a168328440cb379ec3da31272ca15c72b8bf7449f2af3e96ff0c609865a660`
for `models/crossad.py`, and
`04fb6b1222cb607758dfc2ad0644ec6c4bb3ca1e16a4a76c2487a29e73525979`
for the fitted `scaler.npz`. The Git revision in `config.json` is a base commit;
these source files were untracked or modified during this experiment.
