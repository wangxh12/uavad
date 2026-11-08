"""WTFTP same-window reconstruction on ALFA's prepared real-fault split."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from data_provider.data_factory import reconstruction_data_provider
from exp.exp_basic import Exp_Basic
from models.wtftp_reconstruction import WTFTPReconstruction, WTFTPReconstructionConfig


class Exp_Reconstruction(Exp_Basic):
    def __init__(self, args):
        self._flights = None
        self._data = {}
        super().__init__(args)

    def _build_model(self):
        if self.args.model != 'WTFTP':
            raise ValueError('Exp_Reconstruction currently supports WTFTP only')
        # Do not resolve models.WTFTP: that module remains the forecasting model.
        return WTFTPReconstruction(WTFTPReconstructionConfig(
            seq_len=self.args.seq_len, channels=self.args.enc_in,
            embedding=self.args.d_model, encoder_layers=self.args.e_layers,
            decoder_layers=self.args.d_layers, dropout=self.args.dropout)).float()

    def _get_data(self, flag):
        if flag not in self._data:
            dataset, loader = reconstruction_data_provider(self.args, flag, self._flights)
            self._flights = dataset.all_flights
            self._data[flag] = dataset, loader
        return self._data[flag]

    def _contract(self):
        dataset, _ = self._get_data('val')
        manifest = Path(self.args.root_path) / 'manifest.json'
        return {
            'task': 'same_window_reconstruction', 'architecture': 'wtftp_haar_reconstruction',
            'model': self.model.config.to_dict(), 'columns': dataset.columns,
            'normalization': dataset.normalization,
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'threshold_quantile': self.args.threshold_quantile,
            'training_objective': 'MSE(low) + MSE(high)',
            'score_definition': 'mean channel squared reconstruction error; mean across overlapping windows',
        }

    @torch.no_grad()
    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total, count = 0.0, 0
        for values, _, _ in vali_loader:
            loss = self.model(values.float().to(self.device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')
            total += loss.item() * len(values)
            count += len(values)
        if not count:
            raise ValueError('Validation loader is empty')
        return total / count

    def train(self, setting):
        checkpoint_dir = Path(self.args.checkpoints) / setting
        result_dir = Path(self.args.result_path) / setting
        for path in (checkpoint_dir, result_dir):
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite an existing run: {path}. Use a new --setting.')
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        contract = self._contract()
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = checkpoint_dir / 'checkpoint.pth'
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        best_loss, stale, best_epoch = float('inf'), 0, 0
        with (checkpoint_dir / 'training.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
            for epoch in range(self.args.train_epochs):
                learning_rate = self.args.learning_rate * 0.5 ** (epoch // 10)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                self.model.train()
                total, count = 0.0, 0
                for batch_index, (values, _, _) in enumerate(train_loader):
                    if 0 < self.args.max_train_batches <= batch_index:
                        break
                    values = values.float().to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.model(values)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm,
                                                       error_if_nonfinite=True)
                    optimizer.step()
                    total += loss.item() * len(values)
                    count += len(values)
                if not count:
                    raise ValueError('Training loader is empty')
                val_loss = self.vali(val_data, val_loader)
                writer.writerow([epoch + 1, total / count, val_loss, learning_rate])
                stream.flush()
                print(f'Epoch {epoch + 1:03d} | train coefficient loss: {total / count:.6f} '
                      f'| val coefficient loss: {val_loss:.6f}', flush=True)
                if val_loss < best_loss:
                    best_loss, stale, best_epoch = val_loss, 0, epoch + 1
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    stale += 1
                    if stale >= self.args.patience:
                        print('Early stopping.', flush=True)
                        break
        self._load_checkpoint(checkpoint_path)
        validation = self._score_loader(val_loader)
        threshold = float(np.quantile(np.concatenate([row['scores'] for row in validation]),
                                      self.args.threshold_quantile))
        metadata = {
            'contract': contract, 'threshold': threshold, 'best_val_loss': best_loss,
            'best_epoch': best_epoch, 'seed': self.args.seed,
            'train_windows': len(train_data), 'validation_points': sum(
                len(row['scores']) for row in validation),
            'batch_size': self.args.batch_size, 'max_train_batches': self.args.max_train_batches,
            'train_epochs': self.args.train_epochs, 'patience': self.args.patience,
            'learning_rate': self.args.learning_rate, 'max_grad_norm': self.args.max_grad_norm,
            'data': str(Path(self.args.root_path).resolve()),
        }
        (checkpoint_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                      encoding='utf-8')
        return self.model

    def _load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    @torch.no_grad()
    def _score_loader(self, loader):
        """Average per-window squared errors, not errors of averaged predictions."""
        self.model.eval()
        flights = loader.dataset.flights
        errors = [np.zeros(flight.values.shape, dtype=np.float64) for flight in flights]
        reconstructed = [np.zeros_like(values) for values in errors]
        counts = [np.zeros(len(flight.labels), dtype=np.int32) for flight in flights]
        for values, indexes, starts in loader:
            values = values.float().to(self.device)
            output = self.model.reconstruct(values)
            squared = (output - values).square()
            if not torch.isfinite(squared).all():
                raise RuntimeError('Non-finite reconstruction scores')
            for error, prediction, index, start in zip(
                    squared.cpu().numpy(), output.cpu().numpy(), indexes.tolist(), starts.tolist()):
                stop = start + len(error)
                errors[index][start:stop] += error
                reconstructed[index][start:stop] += prediction
                counts[index][start:stop] += 1
        rows = []
        for flight, error, reconstruction, count in zip(flights, errors, reconstructed, counts):
            if not np.all(count > 0):
                raise RuntimeError(f'Window scoring left uncovered points: {flight.name}')
            features = (error / count[:, None]).astype(np.float32)
            rows.append({'flight': flight, 'feature_scores': features, 'scores': features.mean(1),
                         'reconstruction': (reconstruction / count[:, None]).astype(np.float32),
                         'overlap_count': count})
        if not rows:
            raise ValueError('Score loader is empty')
        return rows

    @staticmethod
    def _metrics(rows):
        labels = np.concatenate([row['labels'] for row in rows])
        scores = np.concatenate([row['scores'] for row in rows])
        predictions = np.concatenate([row['predictions'] for row in rows])
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0)
        return {
            'points': len(labels), 'anomaly_points': int(labels.sum()),
            'auroc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            'auprc': float(average_precision_score(labels, scores)) if labels.any() else None,
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        }

    @torch.no_grad()
    def test(self, setting, test=0):
        result_dir = Path(self.args.result_path) / setting
        if result_dir.exists():
            raise FileExistsError(f'Refusing to overwrite results: {result_dir}. Use a new --result_path.')
        checkpoint_dir = Path(self.args.checkpoints) / setting
        metadata = json.loads((checkpoint_dir / 'metadata.json').read_text(encoding='utf-8'))
        if metadata['contract'] != self._contract():
            raise ValueError('Checkpoint model, prepared data, normalization or threshold settings differ')
        threshold = float(metadata['threshold'])
        if not np.isfinite(threshold):
            raise ValueError('Checkpoint threshold must be finite')
        self._load_checkpoint(checkpoint_dir / 'checkpoint.pth')
        dataset, loader = self._get_data('test')
        scored = self._score_loader(loader)
        result_dir.mkdir(parents=True, exist_ok=False)
        score_dir = result_dir / 'scores'
        score_dir.mkdir()
        segments = {segment['name']: segment for segment in dataset.manifest['segments']
                    if segment['split'] == 'test'}
        rows, report = [], []
        for row in scored:
            flight = row['flight']
            predictions = (row['scores'] > threshold).astype(np.int8)
            kind = '+'.join(kind for kind in ('engine', 'elevator', 'aileron', 'rudder')
                            if kind in flight.name) or 'none'
            rows.append({'labels': flight.labels, 'scores': row['scores'],
                         'predictions': predictions, 'type': kind})
            onset = flight.normal_end
            alarms = np.flatnonzero(predictions[onset:])
            delay = float(flight.time[onset + alarms[0]] - flight.time[onset]) if len(alarms) else None
            flight_metrics = self._metrics([rows[-1]])
            report.append({
                'flight': flight.name, 'group': flight.group, 'fault_type': kind,
                'anomaly_points': int(flight.labels.sum()), 'detected': bool(len(alarms)),
                'first_alarm_delay_s': delay, 'false_alarm_points': int(predictions[:onset].sum()),
                'normal_flight_false_alarm': bool(predictions.any()) if onset == len(predictions) else None,
                **{key: flight_metrics[key] for key in ('auroc', 'auprc', 'precision', 'recall', 'f1')},
            })
            segment = segments[flight.name]
            np.savez_compressed(
                score_dir / f'{Path(flight.name).stem}.npz',
                time_s=flight.time, source_row=np.arange(segment['source_start'], segment['source_stop']),
                labels=flight.labels, feature_scores=row['feature_scores'], scores=row['scores'],
                predictions=predictions, reconstruction=row['reconstruction'], true=flight.values,
                overlap_count=row['overlap_count'], features=np.array(dataset.columns))
        normal = [row for row in report if not row['anomaly_points']]
        faults = [row for row in report if row['anomaly_points']]
        metrics = {
            'protocol': f'offline {self.args.seq_len}-point reconstruction; real ALFA faults; no point adjustment',
            'evaluation_scope': dataset.manifest['evaluation_scope'],
            'threshold': threshold, 'threshold_quantile': self.args.threshold_quantile,
            'overall': self._metrics(rows),
            'normal_flights': {'count': len(normal), 'with_false_alarm': sum(
                row['normal_flight_false_alarm'] for row in normal)},
            'fault_events': {'count': len(faults), 'detected': sum(row['detected'] for row in faults),
                             'delays_s': [row['first_alarm_delay_s'] for row in faults]},
            'by_fault_type': {kind: self._metrics([row for row in rows if row['type'] == kind])
                              for kind in sorted({row['type'] for row in rows})},
        }
        with (result_dir / 'flights.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
        for name, contents in (('metrics.json', metrics), ('metadata.json', metadata)):
            (result_dir / name).write_text(json.dumps(contents, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Saved: {result_dir}', flush=True)
        return metrics
