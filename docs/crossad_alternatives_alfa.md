# CrossAD dual-path and relational cross-scale variants

The original `models/crossad.py` and the earlier `models/crossad_relation.py`
are unchanged. Both new variants live in `models/crossad_alternatives.py` and
use the original CrossAD multiscale squared-error score and ALFA data split.
No flight-state calibration, revised score, or revised threshold is used.

## Architectures

- `crossad_dual`: The original channelwise encoder and decoder are retained.
  A learned query for each target channel attends to the **other** channels at
  the same patch and scale; the diagonal is masked. The conditional path is
  gated into the decoder seeds, while a separate conditional head reconstructs
  every target only from other-channel representations. During training, its
  MSE has weight 0.2. The query and conditional head never read the target
  input. Normal-validation checkpoint selection uses the ordinary reconstruction
  MSE, and inference uses the original multiscale reconstruction score.
- `crossad_crossscale`: The channelwise encoder and bank are retained. At each
  coarse-to-fine transition, the target's coarse patch queries other channels'
  coarse patches at offsets -1, 0, and +1. Invalid positions and the target's
  own channel are masked. A per-scale, per-target gate injects this message
  into the upsampled decoder seed. This is an **offline** model; it is not a
  causal online alarm.

Zeroing each variant's gates recovers the original CrossAD reconstruction for
identical backbone weights. The two variants address different mechanisms:
target-blind conditional reconstruction versus relationship-conditioned
coarse-to-fine generation. Neither modifies the normal query/context bank.

## Reproduce the local bounded comparison

Run in the local WSL `ltsm` environment from the project root. Each run starts
from the same original `results/crossad_alfa/main/best.pt`, with seed 2025,
batch size 16, at most 50 batches per epoch, and four epochs. That is at most
3,200 presented training windows out of 38,434 available stride-one windows;
the same windows can recur between epochs. The initial smoke runs used only
two batches per epoch and separately verified checkpoint resumption. No H100
training was launched for these variants.

```bash
python -m pytest tests/test_crossad_alternatives.py tests/test_crossad.py tests/test_alfa.py -q
python -m scripts.crossad_alfa train --model crossad_dual --init-checkpoint results/crossad_alfa/main/best.pt --run-dir results/crossad_alfa/dual_local_bounded_20261027 --device cuda --batch-size 16 --max-batches 50 --epochs 4 --evaluate-after
python -m scripts.crossad_alfa train --model crossad_crossscale --init-checkpoint results/crossad_alfa/main/best.pt --run-dir results/crossad_alfa/crossscale_local_bounded_20261027 --device cuda --batch-size 16 --max-batches 50 --epochs 4 --evaluate-after
```

The run configuration records the model and data manifest hashes, source
checkpoint hash, seed, architecture and training budget. `best.pt` is selected
using only normal validation; `metrics.json`, `flights.csv`, and `scores/`
cover all 16 held-out flights and all their points. No point adjustment is used.

## Preliminary results (local, 2026-10-27)

| Model | Test AUROC | Test AUPRC | Test F1 | Fault flights detected | Normal flights with false alarm |
|---|---:|---:|---:|---:|---:|
| Original CrossAD (formal reference) | 0.58100 | 0.29590 | 0.24699 | 7/13 | 2/3 |
| Dual path, bounded fine-tune | 0.58061 | 0.29500 | 0.24675 | 7/13 | 2/3 |
| Cross-scale, bounded fine-tune | 0.58046 | 0.29515 | 0.24645 | 7/13 | 3/3 |

The learned paths affect the reconstruction, but neither shows an overall
improvement in this limited run. Zeroing the trained gates **without retraining**
gives test AUROC 0.58032 for the dual model and 0.58032 for the cross-scale
model, compared with 0.58061 and 0.58046 with gates enabled. This diagnostic
does not establish a useful cross-variable contribution. The short fine-tunes
and single seed cannot establish final model quality; do not compare their
absolute numbers as if they were equal-budget full training runs.

The model checkpoints, scores and labels from the held-out set must not be used
to choose architecture or hyperparameters. A formal comparison would require
pre-committed training budgets and repeat seeds after local validation, and
requires approval before any long H100 run.
