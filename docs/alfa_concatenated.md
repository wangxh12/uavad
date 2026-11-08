# ALFA continuous-concatenation baseline

This is a separate representation of `data/processed/alfa_within_flight_v1`.
The source protocol selected all 46 labeled CSV files (10 entirely normal,
36 with a fault), grouped into 30 acquisition timestamps. It excluded the one
all-`-1` no-ground-truth file. Each selected file contributes its own normal
prefix to train (60%) and validation (20%); the rest, including all faults,
goes to test. The 18 features, file order, labels, and train-only z-score are
unchanged. This is a same-flight chronological split, not a held-out-flight test.

Run locally with the `ltsm` Python environment:

```bash
cd /home/wrj/workspace/uavad
/home/wrj/miniconda3/envs/ltsm/bin/python scripts/prepare_alfa_concatenated.py
```

The script refuses to overwrite its output at
`data/processed/alfa_concatenated_v1`. Each split has one `(N, 18)` float32
`train.npy`, `val.npy`, or `test.npy` and a matching one-dimensional
`*_label.npy`. `*_provenance.npz` records `flight_id`, `source_row`, `time_s`,
and `segment_offsets`; `manifest.json` lists selected files, the source hash,
segment mappings, counts, and output hashes. Provenance is not model input.

`ConcatenatedWindows(output_dir, split)` uses stride 1 for training and stride
96 with a final tail window for validation/test. It slides across the whole
array, so windows at flight joins **do combine unrelated flights**. This is
intentional to reproduce the continuous-concatenation convention, not a claim
that the flight logs are physically continuous. The manifest counts these
cross-boundary windows. Do not feed these arrays through `FlightWindows`, which
restores flight boundaries. Existing CrossAD training scripts still use
`FlightWindows`; this export does not silently change those experiments.

The previous held-out-acquisition-group CrossAD dataset is a different split:
21 train files, 9 validation files, and 16 test files (30 groups total). Its
test flights must not be conflated with the 46-file same-flight test here.
