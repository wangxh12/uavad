# WTFTP reconstruction experiment

## Data and protocol

`run_reconstruction.py` uses `data/processed/alfa_within_flight_v1` by default.
These are the previously prepared ALFA flight segments with real fault labels,
not the injected-roll CSVs under `datasets/alfa1103`. No injection or preparation
is performed by the reconstruction entrypoint. Prepared artifact checksums and
feature order are verified by the existing loader; the saved normalization is
used as-is, with no refitting on validation/test data.

- Experiment: `exp/exp_reconstruction.py`, class `Exp_Reconstruction`.
- Dataset: `data_provider/reconstruction_loader.py`, class `ALFAReconstructionDataset`.
- Factory: `data_provider.data_factory.reconstruction_data_provider`.
- Model: `models.wtftp_reconstruction.WTFTPReconstruction`, not `models.WTFTP.Model`.
- Input and output: `[batch, 96, 18]`, including measured roll.
- Loss: low- and high-frequency Haar coefficient MSE, using the same window as target.
- Training and validation use normal segments only. Training stride is 1.
- Validation/test stride is 96, with an additional tail window when needed.
- Windows never cross flight or split boundaries.
- Per-point scores average squared reconstruction errors over overlapping windows,
  then average over all 18 channels. Every validation/test point is covered.
- Select the best checkpoint by normal validation coefficient loss. Set the
  threshold from the 99th percentile of the best model's normal validation point
  scores. Test labels are used only for evaluation; there is no point adjustment.

This remains offline detection on later segments of the same flights. Complete
windows use future points within each window, and the prepared split uses known
fault onsets. Do not interpret the metrics or first-alarm delays as causal online
detection or unseen-flight generalization.

## Running

For formal training, SSH to `h100`, enter `/workspace/code/uavad`, and activate
the remote `ltsm` environment. The local WSL environment is independent.

```bash
conda activate ltsm
python run_reconstruction.py --device cuda --gpu 0 \
  --setting WTFTP_alfa_real_reconstruction_w96_seed2025_v1
```

A small smoke run should precede any formal run, in a separate output directory:

```bash
python run_reconstruction.py --device cpu --torch_threads 1 \
  --d_model 8 --e_layers 1 --batch_size 32 \
  --train_epochs 1 --max_train_batches 1 \
  --setting WTFTP_alfa_real_reconstruction_smoke
```

This limits training only. Validation, calibration and testing still cover the
complete prepared validation/test segments. Smoke metrics are not trained-model
results. Large training runs require separate confirmation.

To evaluate an existing checkpoint, retain its model/data/threshold arguments and
select a new results root; evaluation refuses to replace previously saved results:

```bash
python run_reconstruction.py --is_training 0 --device cuda \
  --setting WTFTP_alfa_real_reconstruction_w96_seed2025_v1 \
  --result_path results/reconstruction_recheck
```

## Outputs and compatibility

Checkpoint and result roots default to `checkpoints/reconstruction/<setting>` and
`results/reconstruction/<setting>`. Existing run directories are refused, not
overwritten. The forecasting entrypoint `run.py`, `Exp_Prediction`, forecasting
dataset and `models/WTFTP.py` are unchanged. Existing prediction artifacts and
earlier `results/wtftp_alfa` reconstruction runs are left in place.

Checkpoints contain `checkpoint.pth`, `metadata.json` and `training.csv`. Metadata
records model configuration, normalization, prepared manifest SHA-256, calibration
quantile, threshold and training settings. Checkpoint reuse verifies this contract.

Results contain `metrics.json`, `metadata.json`, `flights.csv` and one `scores/*.npz`
per test flight. Each NPZ stores original labels/times/source rows, normalized
input values (`true`), averaged reconstruction, feature scores, channel-mean
scores, predictions, feature names and overlap counts. The square of the averaged
reconstruction error need not equal the saved score: squared errors are averaged
before, not after, merging overlapping windows.

```bash
python -m pytest -q tests/test_exp_reconstruction.py tests/test_wtftp_reconstruction.py \
  tests/test_alfa_within_flight.py tests/test_alfa1103_wtftp.py
```
