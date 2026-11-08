# Simple reconstruction baselines for the three ALFA rudder flights

The three visually strong rudder test flights all share the same acquisition
group (`2018-09-11-15-06-34`). We compared them with fixed, simple baselines
without training or selecting models on that group. All 16 test flights were
scored for each model, not only these three.

## Protocol

- Data: the unchanged `data/processed/alfa_crossad` 18-feature Z-score
  manifest; training uses only normal prefixes of training flights, including
  the 42,445 normal points available to PCA.
- PCA-8 reconstructs each point through eight principal components fitted to
  training-normal points. MLP-AE compresses a 192 x 18 window through 256 and
  64 hidden units; LSTM-AE compresses the window through a 64-unit encoder and
  32-unit latent state, decoded by a 64-unit recurrent decoder.
- The two neural AEs use MSE, Adam (1e-3), batch size 128, seed 2025, at most
  20 epochs, and patience 3 on normal validation reconstruction loss. MLP-AE
  selected epoch 18; LSTM-AE selected epoch 1 and stopped after epoch 4.
- The common evaluator uses offline 192-point windows, tail coverage,
  pointwise mean over 18 squared reconstruction errors, and an independently
  calibrated 99th-percentile threshold from normal validation scores. There
  is no point adjustment. PCA fits on points but is evaluated through the same
  window/point interface. CrossAD retains its own original multiscale score,
  so the models do not share an identical error construction.

## Results (2026-10-27)

| Model | Rudder 1 AUROC / F1 / normal FP points | Rudder 2 | Rudder 3 | Pooled AUROC | Pooled F1 | Normal flights with FP |
|---|---|---|---|---:|---:|---:|
| Original CrossAD | .985 / .867 / 70 | .999 / .931 / 51 | .999 / .934 / 26 | .5810 | .2470 | 2/3 |
| Relation CrossAD | .981 / .861 / 81 | .998 / .937 / 46 | 1.000 / .936 / 25 | .5807 | .2444 | 2/3 |
| PCA-8 | .992 / .837 / 106 | .998 / .913 / 64 | .998 / .895 / 42 | .5515 | .2356 | 3/3 |
| MLP-AE | .996 / .949 / 27 | .987 / .874 / 90 | .963 / .762 / 115 | .5612 | .2581 | 3/3 |
| LSTM-AE | .979 / .832 / 72 | .997 / .929 / 43 | .997 / .919 / 16 | .5576 | .2485 | 2/3 |

PCA alone ranks fault points above normal points extremely well on these
three flights. A high flight-wise AUROC therefore cannot establish that the
CrossAD cross-variable branch was necessary. None of these baselines resolves
the broader generalization or normal-flight false-alarm problem, and the three
rudder flights are not independent acquisition groups. Fixed-threshold F1 and
false positives differ substantially even when AUROC is similarly high.

`results/simple_alfa/all_flights_comparison.csv` aligns scores and labels for
five models x 16 flights; `three_rudder_flights.csv` filters the predefined
group for inspection. Each run (`pca8`, `mlp_ae`, `lstm_ae`) has `config.json`,
`metrics.json`, `flights.csv`, per-flight `scores/*.npz`, and all 16
`figures/per-flight/*.png`. The neural runs include `best.pt`, `latest.pt`,
and `train.log`; PCA has `best.pt` and `pca_fit.json`.

On `h100:/workspace/code/uavad`, with its independent `ltsm` environment:

```bash
python -m scripts.simple_alfa_baselines train --model pca8 --run-dir results/simple_alfa/pca8 --device cuda:1 --evaluate-after
python -m scripts.simple_alfa_baselines train --model mlp_ae --run-dir results/simple_alfa/mlp_ae --batch-size 128 --epochs 20 --device cuda:1 --evaluate-after
python -m scripts.simple_alfa_baselines train --model lstm_ae --run-dir results/simple_alfa/lstm_ae --batch-size 128 --epochs 20 --device cuda:1 --evaluate-after
python -m scripts.report_simple_baselines
```
