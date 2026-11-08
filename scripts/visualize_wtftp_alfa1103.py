"""Plot validation-calibrated roll forecast thresholds against ALFA 1103 scores."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def trailing_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError('Smoothing window must be positive')
    return np.array([np.median(values[max(0, i - window + 1):i + 1])
                     for i in range(len(values))])


def plot_threshold(run_dir: Path, output: Path, smooth_window: int = 11) -> None:
    model_name = run_dir.name.split('_', 1)[0]
    report = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    with np.load(run_dir / 'scores.npz', allow_pickle=False) as saved:
        rows = saved['row_index'].copy() + 1  # CSV data rows, starting at 1
        labels = saved['labels'].copy()
        scores = saved['roll_score'].copy()
        alarms = saved['predictions'].copy()
    threshold = float(report['threshold'])
    if (not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
            or len(rows) != len(labels) or len(rows) != len(scores)
            or not np.isfinite(scores).all() or threshold <= 0
            or not np.array_equal(alarms, scores > threshold)):
        raise ValueError('Score arrays, row indices or saved threshold are inconsistent')

    onset = rows[np.flatnonzero(labels)[0]] if labels.any() else None
    fig, (timeline, distribution) = plt.subplots(
        2, 1, figsize=(12, 7.2), gridspec_kw={'height_ratios': [1.8, 1]},
        constrained_layout=True)
    palette = {'score': '#276b91', 'fault': '#b74b59', 'alarm': '#dc891d',
               'threshold': '#20262b'}
    positive = np.maximum(scores, 1e-8)
    smoothed = np.maximum(trailing_median(scores, smooth_window), 1e-8)
    if onset is not None:
        timeline.axvspan(onset - 0.5, rows[-1] + 0.5, color=palette['fault'],
                         alpha=0.12, label=f'Injected roll +8 (from row {onset})')
        timeline.axvline(onset - 0.5, color=palette['fault'], lw=1)
    timeline.plot(rows, positive, color=palette['score'], lw=0.65, alpha=0.28,
                  label='Raw squared roll error')
    timeline.plot(rows, smoothed, color=palette['score'], lw=1.45,
                  label=f'Trailing median ({smooth_window} points; display only)')
    timeline.scatter(rows[alarms.astype(bool)], positive[alarms.astype(bool)],
                     s=19, color=palette['alarm'], zorder=3, label='Above threshold')
    timeline.axhline(threshold, color=palette['threshold'], ls='--', lw=1.4,
                     label=f'Validation 99th percentile = {threshold:.3f}')
    timeline.set(xlim=(rows[0], rows[-1]), yscale='log',
                 ylim=(max(positive.min() * 0.7, 1e-8), max(positive.max(), threshold) * 1.5),
                 xlabel='Test CSV row (1-based)', ylabel='Roll anomaly score',
                 title=f'{model_name} test scores and validation-calibrated threshold')
    timeline.grid(alpha=0.18)
    timeline.legend(loc='lower left', ncol=2, fontsize=8.5, frameon=True,
                    facecolor='white', framealpha=0.92, edgecolor='none')

    normal = positive[labels == 0]
    fault = positive[labels == 1]
    bins = np.linspace(np.log10(positive.min()), np.log10(positive.max()), 65)
    for sample, name, color in ((normal, f'Normal (n={len(normal)})', palette['score']),
                                (fault, f'Injected (n={len(fault)})', palette['fault'])):
        if len(sample):
            distribution.hist(np.log10(sample), bins=bins, density=True, histtype='step',
                              lw=1.7, color=color, label=name)
    distribution.axvline(np.log10(threshold), color=palette['threshold'], ls='--',
                         lw=1.4, label=f'Threshold {threshold:.3f}')
    distribution.set(xlabel='log10(roll anomaly score)', ylabel='Density',
                     title='Test score distributions')
    distribution.grid(alpha=0.18)
    distribution.legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'results/WTFTP_alfa1103_cross_roll_seed2025')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smooth-window', type=int, default=11,
                        help='Trailing median window for the timeline only; scores and alarms stay raw')
    args = parser.parse_args()
    output = args.output or args.run_dir / 'threshold.png'
    plot_threshold(args.run_dir, output, args.smooth_window)
    print(output)


if __name__ == '__main__':
    main()
