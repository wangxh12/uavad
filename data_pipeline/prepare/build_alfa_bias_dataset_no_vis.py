#!/usr/bin/env python3
"""将 ALFA 正常架次分别按 6:2:2 切分、拼接，并在 test 尾部注入偏移异常。"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="构建 ALFA 正常训练集与尾部偏移异常测试集"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anomaly-length",
        type=int,
        default=1000,
        help="在拼接后的 test.csv 尾部注入的点数",
    )
    parser.add_argument(
        "--feature", default="field.measured.roll", help="注入异常的特征列名"
    )
    parser.add_argument(
        "--bias", type=float, default=8.0, help="偏移量：新值 = 原值 + bias"
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_no_failure.csv"))

    parts = {name: [] for name in ("train", "val", "test")}
    offsets = {name: 0 for name in parts}
    manifest = []
    columns = None

    for path in files:
        df = pd.read_csv(path).sort_values("time", kind="stable").reset_index(drop=True)
        if columns is None:
            columns = df.columns.tolist()
        if df.columns.tolist() != columns:
            raise ValueError(f"字段不一致：{path.name}")
        if args.feature not in df.columns or args.feature in ("time", "label"):
            raise ValueError(f"非法特征列：{args.feature}")

        n = len(df)
        a, b = int(n * 0.6), int(n * 0.8)
        for split, start, end in (("train", 0, a), ("val", a, b), ("test", b, n)):
            parts[split].append(df.iloc[start:end].copy())
            manifest.append(
                {
                    "split": split,
                    "flight": path.stem,
                    "source_start": start,
                    "source_end_exclusive": end,
                    "concat_start": offsets[split],
                    "concat_end_exclusive": offsets[split] + end - start,
                }
            )
            offsets[split] += end - start

    merged = {
        name: pd.concat(chunks, ignore_index=True) for name, chunks in parts.items()
    }
    test = merged["test"]
    k = args.anomaly_length
    if not 0 <= k <= len(test):
        raise ValueError(f"--anomaly-length 需要在 0 到 {len(test)} 之间，实际为 {k}")
    if k:
        idx = test.index[-k:]
        test.loc[idx, args.feature] += args.bias
        test.loc[idx, "label"] = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in merged.items():
        frame.to_csv(args.output_dir / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(frame):5d} 点，异常 {int(frame['label'].sum()):4d} 点")

    pd.DataFrame(manifest).to_csv(args.output_dir / "boundaries.csv", index=False)
    print(f"\n处理正常架次：{len(files)} 个")
    print(f"注入位置：test.csv 最后 {k} 点；列={args.feature}；偏移={args.bias:+g}")
    print(f"输出目录：{args.output_dir}")
    print("已保存 boundaries.csv：记录各架次在三个拼接文件中的行号边界（0起始、右开区间）")


if __name__ == "__main__":
    main()
