"""Create a flight-level ALFA forecasting anomaly-detection dataset.

Run from the repository root:
    python -m data_pipeline.injection.inject_alfa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .transforms import inject_bias

DEFAULT_SOURCE = "datasets/ALFA/intermediate/alfa_20hz_trim10s_18features"
DEFAULT_OUTPUT = "datasets/ALFA/forecast/roll_bias_v1"
ROLL_COLUMN = "field.measured.roll"
MIN_ANOMALY_POINTS = 20  # minimum anomaly duration; independent of model windows


def generate(
    source_dir: Path,
    output_dir: Path,
    seed: int = 2024,
    bias: float = 8.0,
    onset_min: float = 0.4,
    onset_max: float = 0.7,
) -> dict:
    """Split 10 normal flights 6/2/2; inject persistent Roll bias into test only."""
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}；请先改名或移走旧数据。")
    if not (0 < onset_min <= onset_max < 1):
        raise ValueError("异常起点比例应满足 0 < onset_min <= onset_max < 1")

    files = sorted(source_dir.glob("*_no_failure.csv"))
    if len(files) != 10:
        raise ValueError(
            f"预期找到 10 个正常架次，实际找到 {len(files)} 个：{source_dir}"
        )

    frames = {p.name: pd.read_csv(p) for p in files}
    columns = list(frames[files[0].name].columns)
    features = [col for col in columns if col not in ("time", "label")]
    if len(features) != 18 or ROLL_COLUMN not in features:
        raise ValueError("源数据必须包含 18 个特征，其中含 field.measured.roll")

    for name, df in frames.items():
        if list(df.columns) != columns:
            raise ValueError(f"特征列或列顺序不一致：{name}")
        if not (df["label"] == 0).all():
            raise ValueError(f"正常架次中包含非零标签：{name}")
        if len(df) < MIN_ANOMALY_POINTS + 1:
            raise ValueError(
                f"架次过短，无法保留正常前缀和异常片段：{name}（{len(df)} 行）"
            )

    rng = np.random.default_rng(seed)
    shuffled = [files[i].name for i in rng.permutation(len(files))]
    splits = {
        "train": shuffled[:6],
        "val": shuffled[6:8],
        "test": shuffled[8:],
    }

    # Check injection feasibility without reference to seq_len or pred_len.
    for name in splits["test"]:
        n = len(frames[name])
        low = max(1, int(np.ceil(n * onset_min)))
        high = min(int(np.floor(n * onset_max)), n - MIN_ANOMALY_POINTS)
        if low > high:
            raise ValueError(
                f"测试架次 {name}（{n} 行）无法按给定比例注入；"
                "请调整 onset_min / onset_max 或减小 MIN_ANOMALY_POINTS。"
            )

    metadata = {
        "source_dir": str(source_dir),
        "split_method": "flight-level, seeded shuffle, 6/2/2",
        "seed": seed,
        "features": features,
        "injection": {
            "type": "persistent_bias",
            "feature": ROLL_COLUMN,
            "magnitude": bias,
            "onset_fraction_range": [onset_min, onset_max],
            "applied_to": ["test"],
            "min_anomaly_points_per_test_flight": MIN_ANOMALY_POINTS,
        },
        "splits": splits,
        "flights": {},
    }

    for split, names in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            df = frames[name].copy()
            n = len(df)
            start = None
            df["label"] = 0

            if split == "test":
                low = max(1, int(np.ceil(n * onset_min)))
                high = min(int(np.floor(n * onset_max)), n - MIN_ANOMALY_POINTS)
                start = int(rng.integers(low, high + 1))
                df[ROLL_COLUMN], df["label"] = inject_bias(
                    df[ROLL_COLUMN].to_numpy(dtype=np.float64),
                    start=start,
                    end=n,
                    magnitude=bias,
                )

            df.to_csv(split_dir / name, index=False)
            metadata["flights"][name] = {
                "split": split,
                "num_points": n,
                "anomaly_start": start,  # inclusive, None for train/val
                "anomaly_end": n if start is not None else None,  # exclusive
                "anomaly_points": int(df["label"].sum()),
                "anomaly_ratio": float(df["label"].mean()),
                "anomaly_start_time": (
                    float(df["time"].iloc[start]) if start is not None else None
                ),
            }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for split, names in splits.items():
        points = sum(metadata["flights"][n]["num_points"] for n in names)
        anomalies = sum(metadata["flights"][n]["anomaly_points"] for n in names)
        print(f"{split:5s}: {len(names)} 架次，{points} 时间点，{anomalies} 异常点")
        for name in names:
            start = metadata["flights"][name]["anomaly_start"]
            print(f"  {name}  onset={start}")
    print(f"保存到：{output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALFA 按飞行架次划分并注入持续 Roll 偏置"
    )
    parser.add_argument("--source-dir", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bias", type=float, default=8.0)
    parser.add_argument("--onset-min", type=float, default=0.4)
    parser.add_argument("--onset-max", type=float, default=0.7)
    args = parser.parse_args()

    generate(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        bias=args.bias,
        onset_min=args.onset_min,
        onset_max=args.onset_max,
    )


if __name__ == "__main__":
    main()
