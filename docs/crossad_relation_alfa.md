# CrossAD with multiscale lagged variable relations on ALFA

The unchanged CrossAD baseline is in `models/crossad.py`. The experimental model is
`models/crossad_relation.py`; it keeps the original channelwise multiscale encoder,
query/context bank, decoder, and pointwise score aggregation. After the encoder,
each scale has a learned top-4 directed variable graph, cross-variable attention
over the current and adjacent patches, and a gated residual connection. A target
never uses its own channel as an attention key. With zero gates, the reconstruction
path is the original CrossAD path.

During training only, a randomly chosen complete channel per window is set to
zero **before downsampling**. The conditional reconstruction of that channel is
compared with its unmasked multiscale targets, weighted by 0.2. The auxiliary
pass reads the fixed context bank but does not update it; the ordinary unmasked
pass updates it as before. Evaluation uses only the ordinary reconstruction and
the original CrossAD anomaly score. There is no causal/online claim: both
neighboring-patch directions and the 192-point window are available offline.

Use the existing `data/processed/alfa_crossad` manifest and train-normal-only
z-score scaler. The timestamp-group splits, 18-feature ordering, normal-only
validation checkpoint selection, validation-only 99th-percentile threshold, and
tail-covering test windows are unchanged. All 16 held-out flights are scored;
`flights.csv` contains per-flight AUROC/AUPRC, fixed-threshold precision/recall/F1,
first-alarm delay and false-alarm points. AUROC is undefined for normal-only
flights. Retrospectively showing the strongest flights is illustrative, not an
unbiased estimate or a test-set tuning procedure. Inspect all per-flight plots
and the normal flights alongside any selected examples.

Local WSL smoke test in its `ltsm` environment:

```bash
python -m pytest tests/test_crossad.py tests/test_alfa.py -q
python -m scripts.crossad_alfa train --model crossad_relation --run-dir results/crossad_alfa/relation_v1_smoke --batch-size 2 --epochs 1 --max-batches 2 --device cuda --evaluate-after
python -m scripts.crossad_alfa train --model crossad_relation --run-dir results/crossad_alfa/relation_v1_smoke --batch-size 2 --epochs 2 --max-batches 2 --device cuda --resume
```

Formal run on `h100:/workspace/code/uavad` in its own `ltsm` environment:

```bash
python -m scripts.crossad_alfa train --model crossad_relation --run-dir results/crossad_alfa/relation_v1 --epochs 20 --batch-size 128 --device cuda:1 --evaluate-after
python -m scripts.visualize_crossad --run-dir results/crossad_alfa/relation_v1
```

The model and manifest SHA-256 hashes, seed, normalization, and other run
settings are saved in `config.json` and checkpoints. `latest.pt` enables exact
epoch-boundary resume; `best.pt` is selected by normal validation reconstruction
loss. The test labels are only read for reporting, not training or selection.

## Completed single-seed run (2026-10-27)

The formal run stopped after epoch 9; epoch 6 had the lowest normal validation
loss (0.085698). All 16 held-out flights (13 fault, 3 normal) have complete
point scores and individual figures under `results/crossad_alfa/relation_v1/`.
The fixed validation-only threshold is 3.53934. Pooled test AUROC/AUPRC/F1
are 0.58071/0.29544/0.24436, compared with original CrossAD
0.58100/0.29590/0.24699. Seven of 13 fault flights have an alarm after onset;
two of three normal flights have false alarms.

The three rudder flights have AUROC 0.9806, 0.9984, and 0.9996, and F1
0.8615, 0.9372, and 0.9364. They were already similarly strong with the
original model. The elevator flight has AUROC 0.6641 but F1 only 0.1788,
with 1,225 false-alarm points before onset. Aileron flight recall is near
zero. High within-flight AUROC does not guarantee an actionable alarm under
the unchanged threshold. `comparison.csv` contains all flights, including
the regressions and normal-flight false alarms; the best examples are selected
retrospectively and do not establish generalization.

As a diagnostic, `scripts.ablate_crossad_relation` set only the trained
relation gates to zero without retraining. Its pooled test AUROC was 0.58065,
nearly identical to the full model (0.58071). This run therefore does not
demonstrate an effective gain from channel interactions; the model and failed
ablation are retained as a reproducible starting point, not a successful
improvement claim. The ablation files are in
`results/crossad_alfa/relation_v1_gate_zero/`.
