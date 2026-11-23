#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 ALFA 偏移异常数据集，并自动绘制注入前后对比图。

处理：10 个正常架次分别按时间 6:2:2 切分 -> 同集合拼接 ->
      test.csv 最后 N 点的指定特征 += bias，label 设为 1。

输出：train.csv / val.csv / test.csv / boundaries.csv /
      injection_summary.json / figures/test_injection_comparison.png

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无图形界面的 Linux/WSL 服务器也可以保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALFA 正常架次划分、偏移异常注入及可视化")
    parser.add_argument("--input-dir", type=Path, required=True, help="正常架次 CSV 所在目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="实验数据集输出目录")
    parser.add_argument("--anomaly-length", type=int, default=1000,
                        help="在拼接后的 test.csv 最后 N 点注入（默认 1000）")
    parser.add_argument("--feature", type=str, default="field.measured.roll",
                        help="注入偏移异常的特征名（默认 field.measured.roll）")
    parser.add_argument("--bias", type=float, default=8.0,
                        help="加性偏移量：新值=原值+bias（默认 +8）")
    parser.add_argument("--plot-context", type=int, default=150,
                        help="局部放大图中异常起点前后各显示多少点（默认 150）")
    parser.add_argument("--plot-dpi", type=int, default=360,
                        help="PNG 图片分辨率（默认 360 DPI）")
    return parser.parse_args()


def load_and_split(input_dir: Path, feature: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, int]:
    files = sorted(input_dir.glob("*_no_failure.csv"))
    if not files:
        raise ValueError(f"未找到正常架次 CSV：{input_dir}")

    pieces = {split: [] for split in SPLITS}
    offsets = {split: 0 for split in SPLITS}
    records: list[dict] = []
    reference_columns = None

    for path in files:
        df = pd.read_csv(path)
        columns = df.columns.tolist()
        if reference_columns is None:
            reference_columns = columns
            required = ("time", "label", feature)
            missing = [name for name in required if name not in columns]
            if missing:
                raise ValueError(f"输入数据缺少字段：{missing}")
            if feature in ("time", "label"):
                raise ValueError("--feature 必须是传感器特征，不能是 time 或 label")
        elif columns != reference_columns:
            raise ValueError(f"字段名称或顺序不一致：{path.name}")
        if df.empty:
            raise ValueError(f"空的正常架次：{path.name}")
        if not df["label"].eq(0).all():
            raise ValueError(f"正常架次包含非零/缺失标签：{path.name}")

        values = pd.to_numeric(df[feature], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"注入特征存在非数值/NaN/Inf：{path.name}: {feature}")
        df[feature] = values
        df = df.sort_values("time", kind="stable").reset_index(drop=True)

        n = len(df)
        a, b = int(0.6 * n), int(0.8 * n)
        for split, start, end in (
            ("train", 0, a), ("val", a, b), ("test", b, n)
        ):
            pieces[split].append(df.iloc[start:end].copy())
            records.append({
                "split": split,
                "flight": path.stem,
                "source_start": start,
                "source_end_exclusive": end,
                "concat_start": offsets[split],
                "concat_end_exclusive": offsets[split] + (end - start),
            })
            offsets[split] += end - start

    merged = {
        split: pd.concat(pieces[split], ignore_index=True)
        for split in SPLITS
    }
    return merged, pd.DataFrame.from_records(records), len(files)


def inject_tail(test: pd.DataFrame, feature: str, length: int, bias: float) -> tuple[np.ndarray, int]:
    """原地注入；返回注入前特征序列和异常起点（0-based）。"""
    total = len(test)
    if not 0 <= length <= total:
        raise ValueError(f"--anomaly-length 必须在 [0, {total}] 内，实际为 {length}")
    if not math.isfinite(bias):
        raise ValueError("--bias 必须是有限数值")
    if length > 0 and bias == 0:
        raise ValueError("--bias=0 不会产生偏移异常，请使用非零值")

    clean = test[feature].to_numpy(dtype=np.float64, copy=True)
    start = total - length
    if length:
        col = test.columns.get_loc(feature)
        label_col = test.columns.get_loc("label")
        # 明确用位置索引，避免标签索引或重复索引影响尾部注入。
        test.iloc[start:, col] = clean[start:] + bias
        test.iloc[start:, label_col] = 1
    return clean, start


def plot_injection_comparison(
    clean: np.ndarray,
    injected: np.ndarray,
    feature: str,
    anomaly_start: int,
    boundaries: pd.DataFrame,
    out_path: Path,
    context: int,
    dpi: int,
) -> None:
    """对比注入前后曲线，标注异常区间及架次边界，不连线跨架次。"""
    n = len(injected)
    subset = boundaries.loc[boundaries["split"] == "test"].sort_values("concat_start")
    delta = injected - clean
    has_anomaly = anomaly_start < n
    colors = {"clean": "#315D83", "injected": "#CC573B", "boundary": "#7E8790"}

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10),
        gridspec_kw={"height_ratios": [2.8, 2.2, 1.1]},
        constrained_layout=True,
    )
    ax_overview, ax_zoom, ax_diff = axes

    def signal(ax, left: int, right: int, legend: bool) -> None:
        clean_labeled = False
        injected_labeled = False
        for _, row in subset.iterrows():
            s = max(left, int(row["concat_start"]))
            e = min(right, int(row["concat_end_exclusive"]))
            if s >= e:
                continue
            xs = np.arange(s, e)
            # 只在单架次范围内连线，避免拼接形成虚假的跨架次跳变。
            ax.plot(xs, clean[s:e], color=colors["clean"], lw=1.05,
                    alpha=0.95, label="Original" if legend and not clean_labeled else None)
            clean_labeled = True
            changed_start = max(s, anomaly_start)
            if has_anomaly and changed_start < e:
                ax.plot(np.arange(changed_start, e), injected[changed_start:e],
                        color=colors["injected"], lw=1.45,
                        label="After injection" if legend and not injected_labeled else None)
                injected_labeled = True

    signal(ax_overview, 0, n, legend=True)
    if has_anomaly:
        ax_overview.axvspan(anomaly_start - 0.5, n - 0.5, color=colors["injected"], alpha=0.10,
                            label="Injected region")
        ax_overview.axvline(anomaly_start - 0.5, color=colors["injected"], lw=1.1, ls=":")
    ax_overview.set_title(f"Test set: Original vs. Injected  |  {feature}")
    ax_overview.set_ylabel("Feature value")
    ax_overview.set_xlim(-1, max(n - 1, 1))

    center = anomaly_start if has_anomaly else max(n // 2, 0)
    left, right = max(0, center - context), min(n, center + context + 1)
    signal(ax_zoom, left, right, legend=False)
    if has_anomaly:
        ax_zoom.axvspan(anomaly_start - 0.5, n - 0.5, color=colors["injected"], alpha=0.10)
        ax_zoom.axvline(anomaly_start - 0.5, color=colors["injected"], lw=1.0, ls=":")
    ax_zoom.set_xlim(left - 0.5, right - 0.5)
    ax_zoom.set_ylabel("Feature value")
    ax_zoom.set_title("Detail near injection onset" if has_anomaly else "Detail (no injection)")

    # 曲线差值也只在对应架次内绘制；能验证设定的 +bias 是否真正生效。
    for _, row in subset.iterrows():
        s, e = int(row["concat_start"]), int(row["concat_end_exclusive"])
        if s < e:
            ax_diff.plot(np.arange(s, e), delta[s:e], color=colors["injected"], lw=1.2)
    ax_diff.axhline(0, color=colors["boundary"], lw=0.75, ls=":")
    if has_anomaly:
        ax_diff.axvspan(anomaly_start - 0.5, n - 0.5, color=colors["injected"], alpha=0.10)
        ax_diff.axvline(anomaly_start - 0.5, color=colors["injected"], lw=1.0, ls=":")
    ax_diff.set_xlim(-1, max(n - 1, 1))
    ax_diff.set_title("Difference = After - Original")
    ax_diff.set_ylabel("Delta")
    ax_diff.set_xlabel("Concatenated test row index (0-based)")

    # 注入区间可能跨几个架次，所有架次的边界都画虚线提示。
    starts = subset["concat_start"].astype(int).tolist()
    for ax in axes:
        for i, s in enumerate(starts[1:]):
            if ax.get_xlim()[0] <= s <= ax.get_xlim()[1]:
                ax.axvline(
                    s - 0.5, color=colors["boundary"], lw=0.75, ls="--", alpha=0.6,
                    label="Flight boundary" if ax is ax_overview and i == 0 else None,
                )
        ax.grid(alpha=0.18, linewidth=0.6)

    ax_overview.legend(loc="best", frameon=False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    if output_dir == input_dir:
        raise ValueError("输出目录不能与输入目录相同，避免覆盖原始 CSV")
    if args.plot_context < 1 or args.plot_dpi < 1:
        raise ValueError("--plot-context 和 --plot-dpi 必须为正整数")

    merged, boundaries, num_flights = load_and_split(input_dir, args.feature)
    test = merged["test"]
    clean, start = inject_tail(test, args.feature, args.anomaly_length, args.bias)
    modified = test[args.feature].to_numpy(dtype=np.float64, copy=True)

    # 写入数据前确认只在预期的位置发生变化（最重要的注入正确性检查）。
    if not np.allclose(modified - clean, np.r_[np.zeros(start),
                                              np.full(args.anomaly_length, args.bias)],
                       rtol=1e-9, atol=1e-8):
        raise AssertionError("异常注入幅度或位置不符合设定")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in merged.items():
        frame.to_csv(output_dir / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(frame):5d} 点，异常 {int(frame['label'].eq(1).sum()):4d} 点")
    boundaries.to_csv(output_dir / "boundaries.csv", index=False)

    affected = boundaries.loc[(boundaries["split"] == "test") &
                              (boundaries["concat_end_exclusive"] > start) &
                              (boundaries["concat_start"] < len(test))].copy()
    affected_flights = [
        {"flight": str(row["flight"]),
         "global_start": max(int(row["concat_start"]), start),
         "global_end_exclusive": int(row["concat_end_exclusive"])}
        for _, row in affected.iterrows()
    ] if args.anomaly_length else []

    summary = {
        "input_dir": str(input_dir),
        "num_normal_flights": num_flights,
        "rows": {split: len(df) for split, df in merged.items()},
        "injection": {
            "split": "test", "feature": args.feature, "bias": args.bias,
            "length": args.anomaly_length,
            "global_start": start, "global_end_exclusive": len(test),
            "affected_flights": affected_flights,
        },
        "note": "CSV retains 20 original columns; boundaries use 0-based half-open row intervals.",
    }
    (output_dir / "injection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    figure_path = output_dir / "figures" / "test_injection_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_injection_comparison(clean, modified, args.feature, start, boundaries,
                              figure_path, args.plot_context, args.plot_dpi)

    print(f"\n处理正常架次：{num_flights} 个")
    print(f"注入位置：test.csv 最后 {args.anomaly_length} 点；"
          f"列={args.feature}；偏移={args.bias:+g}")
    print(f"注入区间：[ {start}, {len(test)} )（0-based，右开）")
    print(f"影响架次：{len(affected_flights)} 个")
    print(f"输出目录：{output_dir}")
    print(f"对比图片：{figure_path}")


if __name__ == "__main__":
    main()
