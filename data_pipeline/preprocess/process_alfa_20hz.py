#!/usr/bin/env python3
"""Convert ALFA topic CSVs to one aligned, trimmed 18-feature CSV per flight."""

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE = Path("/mnt/d/datasets/ALFA/processed")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/alfa_20hz_trim20s_18features"
DEFAULT_TRIM_SECONDS = 20.0
MAX_GAP_NS = 300_000_000
REFERENCE_TOPIC = "mavros-nav_info-errors"

# (output column, source topic suffix, source CSV column)
FEATURES = [
    *[
        (
            f"field.angular_velocity.{axis}",
            "mavros-imu-data_raw",
            f"field.angular_velocity.{axis}",
        )
        for axis in "xyz"
    ],
    *[
        (
            f"field.linear_acceleration.{axis}",
            "mavros-imu-data_raw",
            f"field.linear_acceleration.{axis}",
        )
        for axis in "xyz"
    ],
    *[
        (
            f"field.magnetic_field.{axis}",
            "mavros-imu-mag",
            f"field.magnetic_field.{axis}",
        )
        for axis in "xyz"
    ],
    ("field.fluid_pressure", "mavros-imu-atm_pressure", "field.fluid_pressure"),
    ("field.temperature", "mavros-imu-temperature", "field.temperature"),
    *[
        (f"field.{field}", REFERENCE_TOPIC, f"field.{field}")
        for field in ("alt_error", "aspd_error", "xtrack_error", "wp_dist")
    ],
    *[
        (f"field.measured.{angle}", f"mavros-nav_info-{angle}", "field.measured")
        for angle in ("pitch", "roll", "yaw")
    ],
]
OUTPUT_COLUMNS = ["time", *(name for name, _, _ in FEATURES), "label"]

TOPIC_FIELDS = {}
for _, topic, field in FEATURES:
    TOPIC_FIELDS.setdefault(topic, []).append(field)


@dataclass
class Series:
    times: list[int]
    values: dict[str, list[float]]


def read_series(path: Path, fields: list[str], *, discrete: bool = False) -> Series:
    """Read a time-ordered topic; only continuous features require the 0.3-s gap limit."""
    times = []
    values = {field: [] for field in fields}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"%time", *fields} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        for row in reader:
            timestamp = int(row["%time"])
            if times:
                gap = timestamp - times[-1]
                if gap <= 0:
                    raise ValueError(f"{path}: timestamps must be strictly increasing")
                if not discrete and gap > MAX_GAP_NS:
                    raise ValueError(
                        f"{path}: sampling gap {gap / 1e9:g}s exceeds 0.3s"
                    )

            parsed = {
                field: int(row[field]) if discrete else float(row[field])
                for field in fields
            }
            if not discrete and not all(math.isfinite(v) for v in parsed.values()):
                raise ValueError(f"{path}: non-finite feature value")

            times.append(timestamp)
            for field, value in parsed.items():
                values[field].append(value)

    if len(times) < (1 if discrete else 2):
        raise ValueError(f"{path}: insufficient observations")
    return Series(times, values)


def interpolate(
    times: list[int], values: list[float], target: list[int], *, yaw: bool = False
) -> list[float]:
    """Linear interpolation; yaw follows the shortest angular path in degrees."""
    if yaw:
        unwrapped = [values[0]]
        for previous, current in zip(values, values[1:]):
            delta = (current - previous + 180) % 360 - 180
            if delta == -180 and current > previous:
                delta = 180
            unwrapped.append(unwrapped[-1] + delta)
        values = unwrapped

    result = []
    for timestamp in target:
        right = bisect.bisect_left(times, timestamp)
        if right < len(times) and times[right] == timestamp:
            value = values[right]
        else:
            left = right - 1
            fraction = (timestamp - times[left]) / (times[right] - times[left])
            value = values[left] + fraction * (values[right] - values[left])
        result.append((value + 180) % 360 - 180 if yaw else value)
    return result


def make_labels(target: list[int], statuses: list[Series]) -> list[int]:
    """Forward-fill each failure status and OR the active statuses at each timestamp."""
    labels = []
    for timestamp in target:
        active = False
        for status in statuses:
            index = bisect.bisect_right(status.times, timestamp) - 1
            if index >= 0 and status.values["field.data"][index] != 0:
                active = True
                break
        labels.append(int(active))
    return labels


def prepare_flight(
    flight_dir: Path, trim_seconds: float
) -> tuple[list[int], list[list[float]], list[int], dict]:
    topics = {
        topic: read_series(flight_dir / f"{flight_dir.name}-{topic}.csv", fields)
        for topic, fields in TOPIC_FIELDS.items()
    }

    common_start = max(series.times[0] for series in topics.values())
    common_end = min(series.times[-1] for series in topics.values())
    cutoff = common_start + round(trim_seconds * 1_000_000_000)
    target = [t for t in topics[REFERENCE_TOPIC].times if cutoff <= t <= common_end]
    if not target:
        raise ValueError(
            f"{flight_dir.name}: no reference samples after {trim_seconds:g}s trim"
        )

    columns = [
        interpolate(
            topics[topic].times,
            topics[topic].values[field],
            target,
            yaw=(name == "field.measured.yaw"),
        )
        for name, topic, field in FEATURES
    ]

    no_failure = "no_failure" in flight_dir.name.lower()
    status_paths = sorted(flight_dir.glob(f"{flight_dir.name}-failure_status-*.csv"))
    if not no_failure and not status_paths:
        raise ValueError(f"{flight_dir.name}: missing failure_status CSV")
    statuses = [
        read_series(path, ["field.data"], discrete=True) for path in status_paths
    ]
    if no_failure and any(
        value != 0 for status in statuses for value in status.values["field.data"]
    ):
        raise ValueError(
            f"{flight_dir.name}: nonzero failure status in no_failure sequence"
        )
    labels = [0] * len(target) if no_failure else make_labels(target, statuses)

    duration_ns = target[-1] - target[0]
    metadata = {
        "sequence": flight_dir.name,
        "output_file": f"{flight_dir.name}.csv",
        "rows": len(target),
        "duration_s": duration_ns / 1e9,
        "mean_frequency_hz": (
            (len(target) - 1) * 1e9 / duration_ns if duration_ns else None
        ),
        "label_counts": dict(sorted(Counter(str(label) for label in labels).items())),
        "first_output_unix_ns": target[0],
        "last_output_unix_ns": target[-1],
        "common_start_unix_ns": common_start,
        "common_end_unix_ns": common_end,
        "trim_cutoff_unix_ns": cutoff,
    }
    return target, columns, labels, metadata


def write_flight(
    path: Path, times: list[int], columns: list[list[float]], labels: list[int]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        start = times[0]
        for timestamp, features, label in zip(times, zip(*columns), labels):
            seconds, nanoseconds = divmod(timestamp - start, 1_000_000_000)
            writer.writerow(
                [
                    f"{seconds}.{nanoseconds:09d}",
                    *(format(value, ".15g") for value in features),
                    label,
                ]
            )


def convert(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    trim_seconds: float = DEFAULT_TRIM_SECONDS,
) -> dict:
    flight_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    included = [
        path for path in flight_dirs if "no_ground_truth" not in path.name.lower()
    ]
    if not included:
        raise ValueError(f"{source}: no eligible flights")

    # Require a new output directory to avoid accidentally overwriting existing data.
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "source": str(source.resolve()),
        "trim_seconds": trim_seconds,
        "max_interpolation_gap_s": MAX_GAP_NS / 1e9,
        "reference_topic": REFERENCE_TOPIC,
        "nominal_frequency_hz": 20,
        "label_policy": "no_failure=0; otherwise OR of forward-filled failure_status values != 0",
        "time_unit": "seconds relative to first output row",
        "columns": OUTPUT_COLUMNS,
        "features": [
            {"column": name, "topic": topic, "source_field": field}
            for name, topic, field in FEATURES
        ],
        "excluded_sequences": [
            path.name for path in flight_dirs if path not in included
        ],
        "sequences": [],
    }
    totals = Counter()

    for flight_dir in included:
        print(f"Processing {flight_dir.name}", flush=True)
        times, columns, labels, metadata = prepare_flight(flight_dir, trim_seconds)
        write_flight(output / metadata["output_file"], times, columns, labels)
        manifest["sequences"].append(metadata)
        totals.update(labels)

    manifest["sequence_count"] = len(included)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "sequences": len(included),
                "rows": sum(entry["rows"] for entry in manifest["sequences"]),
                "labels": dict(sorted(totals.items())),
            },
            ensure_ascii=False,
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trim-seconds", type=float, default=DEFAULT_TRIM_SECONDS)
    args = parser.parse_args()
    convert(args.source, args.output, args.trim_seconds)

