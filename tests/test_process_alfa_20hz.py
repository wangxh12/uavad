from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'datasets/scripts/process_alfa_20hz.py'
SPEC = importlib.util.spec_from_file_location('process_alfa_20hz', SCRIPT)
assert SPEC and SPEC.loader
processor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = processor
SPEC.loader.exec_module(processor)

BASE = 1_700_000_000_000_000_000


class ProcessAlfa20HzTests(unittest.TestCase):
    def make_flight(self, root: Path, name: str, *, no_failure: bool = False) -> Path:
        flight = root / name
        flight.mkdir(parents=True)

        specs: dict[str, list[str]] = {}
        for output, topic, field in processor.FEATURES:
            specs.setdefault(topic, [])
            if field not in specs[topic]:
                specs[topic].append(field)

        # The errors stream starts at t=0 and is the native output time axis.
        # Every other continuous source starts at t=0.15s, making common_start=0.15s.
        for topic, fields in specs.items():
            step = 50_000_000 if topic == 'mavros-nav_info-errors' else 100_000_000
            first = 0 if topic == 'mavros-nav_info-errors' else 150_000_000
            last = 24_000_000_000
            times = list(range(first, last + 1, step))
            path = flight / f'{name}-{topic}.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['%time', *fields])
                for t in times:
                    sec = t / 1e9
                    row = [BASE + t]
                    for field in fields:
                        if field == 'field.measured' and topic.endswith('-yaw'):
                            value = 179.0 if sec < 20.25 else -179.0
                        else:
                            value = sec
                        row.append(value)
                    writer.writerow(row)

        if not no_failure:
            path = flight / f'{name}-failure_status-engines.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['%time', 'field.data'])
                writer.writerows([
                    [BASE, 0],
                    [BASE + 20_500_000_000, 1],
                    [BASE + 22_000_000_000, 0],
                ])
        return flight

    def test_trim_and_preserve_error_timestamps_with_event_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flight = self.make_flight(root, 'demo_engine_failure')
            target_times, values, labels, meta = processor.prepare_flight(flight)

            # common_start=0.15s; trim 20s; retain errors timestamps >=20.15s.
            self.assertEqual(target_times[0], BASE + 20_150_000_000)
            self.assertTrue(all(b - a == 50_000_000 for a, b in zip(target_times, target_times[1:])))
            self.assertEqual(len(values), 18)
            # The raw 10 Hz IMU is interpolated onto the native error-stream times.
            gyro_x_index = [feature[0] for feature in processor.FEATURES].index('field.angular_velocity.x')
            at_20_200 = target_times.index(BASE + 20_200_000_000)
            self.assertAlmostEqual(values[gyro_x_index][at_20_200], 20.2)
            self.assertEqual(labels[0], 0)
            self.assertEqual(labels[target_times.index(BASE + 20_550_000_000)], 1)
            self.assertEqual(labels[target_times.index(BASE + 22_000_000_000)], 0)
            self.assertEqual(meta['trim']['fixed_trim_s'], 20)

    def test_yaw_interpolates_through_shortest_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flight = self.make_flight(root, 'demo_engine_failure')
            target_times, values, _, _ = processor.prepare_flight(flight)
            yaw_index = [feature[0] for feature in processor.FEATURES].index('field.measured.yaw')
            at_20_200 = target_times.index(BASE + 20_200_000_000)
            self.assertAlmostEqual(abs(values[yaw_index][at_20_200]), 180.0)

    def test_convert_writes_one_csv_and_manifest_without_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            root.mkdir()
            self.make_flight(root, 'demo_no_failure', no_failure=True)
            (root / 'demo_no_ground_truth').mkdir()
            output = Path(tmp) / 'output'

            manifest = processor.convert(root, output)

            files = list(output.glob('*.csv'))
            self.assertEqual([path.name for path in files], ['demo_no_failure.csv'])
            self.assertEqual(manifest['sequence_count'], 1)
            with files[0].open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, processor.OUTPUT_COLUMNS)
                rows = list(reader)
            self.assertTrue(rows)
            self.assertTrue(all(row['label'] == '0' for row in rows))
            self.assertEqual(rows[0]['time'], '0.000000000')

    def test_rejects_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            root.mkdir()
            self.make_flight(root, 'demo_no_failure', no_failure=True)
            output = Path(tmp) / 'output'
            output.mkdir()
            (output / 'keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'must not exist or must be empty'):
                processor.convert(root, output)
            self.assertTrue((output / 'keep.txt').exists())


if __name__ == '__main__':
    unittest.main()
