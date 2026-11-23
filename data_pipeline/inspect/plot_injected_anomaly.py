#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot comparison between clean and injected anomaly datasets."
    )
    parser.add_argument("--clean-csv", type=Path, required=True,
                        help="Path to the clean test.csv (before anomaly injection)")
    parser.add_argument("--anomaly-csv", type=Path, required=True,
                        help="Path to the injected test.csv (after anomaly injection)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save figures")
    parser.add_argument("--feature", type=str, default="field.measured.roll",
                        help="Feature to visualize")
    parser.add_argument("--label-col", type=str, default="label",
                        help="Label column name")
    parser.add_argument("--boundaries-csv", type=Path, default=None,
                        help="Optional boundaries.csv for flight boundary visualization")
    parser.add_argument("--zoom-padding", type=int, default=200,
                        help="Number of points shown before and after anomaly region in zoom figure")
    return parser.parse_args()


def find_anomaly_segments(label_series):
    """Return contiguous anomaly segments [(start, end), ...], end is inclusive."""
    idx = label_series[label_series == 1].index.tolist()
    if not idx:
        return []

    segments = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))
    return segments


def draw_segments(ax, segments):
    for start, end in segments:
        ax.axvspan(start, end, alpha=0.25)


def draw_boundaries(ax, boundaries_df, split="test"):
    if "split" not in boundaries_df.columns:
        return
    if "concat_start" not in boundaries_df.columns:
        return

    subset = boundaries_df[boundaries_df["split"] == split]
    for _, row in subset.iterrows():
        ax.axvline(row["concat_start"], linestyle="--", alpha=0.4)
    if "concat_end_exclusive" in subset.columns and len(subset) > 0:
        ax.axvline(subset.iloc[-1]["concat_end_exclusive"], linestyle="--", alpha=0.4)


def main():
    args = parse_args()

    clean_csv = args.clean_csv.expanduser().resolve()
    anomaly_csv = args.anomaly_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_df = pd.read_csv(clean_csv)
    anomaly_df = pd.read_csv(anomaly_csv)

    if len(clean_df) != len(anomaly_df):
        raise ValueError(
            f"Row count mismatch: clean={len(clean_df)}, anomaly={len(anomaly_df)}"
        )

    if args.feature not in clean_df.columns:
        raise ValueError(f"Feature not found in clean csv: {args.feature}")
    if args.feature not in anomaly_df.columns:
        raise ValueError(f"Feature not found in anomaly csv: {args.feature}")
    if args.label_col not in anomaly_df.columns:
        raise ValueError(f"Label column not found in anomaly csv: {args.label_col}")

    x = range(len(anomaly_df))
    anomaly_segments = find_anomaly_segments(anomaly_df[args.label_col])

    boundaries_df = None
    if args.boundaries_csv is not None:
        boundaries_path = args.boundaries_csv.expanduser().resolve()
        boundaries_df = pd.read_csv(boundaries_path)

    # 1) full figure
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, clean_df[args.feature], label="clean")
    ax.plot(x, anomaly_df[args.feature], label="injected")
    draw_segments(ax, anomaly_segments)

    if boundaries_df is not None:
        draw_boundaries(ax, boundaries_df, split="test")

    ax.set_title(f"Full comparison: {args.feature}")
    ax.set_xlabel("Global row index")
    ax.set_ylabel(args.feature)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    full_path = output_dir / f"{args.feature.replace('.', '_')}_full.png"
    fig.savefig(full_path, dpi=200)
    plt.close(fig)

    # 2) zoom figure
    if anomaly_segments:
        start = min(s for s, _ in anomaly_segments)
        end = max(e for _, e in anomaly_segments)
        left = max(0, start - args.zoom_padding)
        right = min(len(anomaly_df) - 1, end + args.zoom_padding)

        fig, ax = plt.subplots(figsize=(16, 5))
        xs = range(left, right + 1)
        ax.plot(xs, clean_df.iloc[left:right + 1][args.feature], label="clean")
        ax.plot(xs, anomaly_df.iloc[left:right + 1][args.feature], label="injected")

        for seg_start, seg_end in anomaly_segments:
            if seg_end < left or seg_start > right:
                continue
            ax.axvspan(max(seg_start, left), min(seg_end, right), alpha=0.25)

        if boundaries_df is not None:
            subset = boundaries_df[boundaries_df["split"] == "test"]
            for _, row in subset.iterrows():
                pos = row["concat_start"]
                if left <= pos <= right:
                    ax.axvline(pos, linestyle="--", alpha=0.4)
            if "concat_end_exclusive" in subset.columns and len(subset) > 0:
                pos = subset.iloc[-1]["concat_end_exclusive"]
                if left <= pos <= right:
                    ax.axvline(pos, linestyle="--", alpha=0.4)

        ax.set_title(f"Zoomed anomaly region: {args.feature}")
        ax.set_xlabel("Global row index")
        ax.set_ylabel(args.feature)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        zoom_path = output_dir / f"{args.feature.replace('.', '_')}_zoom.png"
        fig.savefig(zoom_path, dpi=200)
        plt.close(fig)
    else:
        print("Warning: no anomaly region found in label column, zoom figure is skipped.")

    print("Saved figures:")
    print(f"  {full_path}")
    if anomaly_segments:
        print(f"  {zoom_path}")


if __name__ == "__main__":
    main()