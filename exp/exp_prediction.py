"""9-step non-roll sensors -> next-step roll forecasting experiment."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic


class Exp_Prediction(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        if (int(args.seq_len), int(args.pred_len)) != (9, 1):
            raise ValueError('Exp_Prediction requires --seq_len 9 --pred_len 1.')

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if getattr(self.args, 'use_gpu', False) and getattr(self.args, 'use_multi_gpu', False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    @staticmethod
    def _last_labels(point_labels):
        """Convert [B], [B,T] or [B,T,C] point labels to [B]."""
        labels = torch.as_tensor(point_labels)
        if labels.ndim > 1:
            labels = labels[:, -1]
            while labels.ndim > 1:
                labels = labels.amax(dim=-1)
        return labels.reshape(-1)

    def _unpack_batch(self, batch):
        """Return 17 input channels, a 10-roll supervision window and its last point."""
        if not isinstance(batch, (list, tuple)) or len(batch) != 5:
            raise ValueError('Expected (non-roll inputs, roll window, marks, marks, label).')
        x, roll_window, _, _, labels = batch
        if (x.ndim != 3 or x.shape[1:] != (9, self.args.enc_in)
                or roll_window.ndim != 3 or roll_window.shape[1:] != (10, 1)):
            raise ValueError('Expected input [B,9,17] and roll target [B,10,1].')
        x = x.float().to(self.device, non_blocking=True)
        roll_window = roll_window.float().to(self.device, non_blocking=True)
        return x, roll_window[:, -1:, :], self._last_labels(labels), roll_window

    def _loss(self, roll_window, outputs):
        """Supervise the output wavelet coefficients using target roll only."""
        with torch.no_grad():
            true_low, true_high = self._base_model().haar_forward(roll_window)
        low_weight = float(getattr(self.args, 'wt_low_weight', 1.0))
        high_weight = float(getattr(self.args, 'wt_high_weight', 1.0))
        time_weight = float(getattr(self.args, 'wt_time_weight', 0.0))
        forecast_weight = float(getattr(self.args, 'wt_forecast_weight', 0.0))
        if low_weight + high_weight + time_weight + forecast_weight <= 0:
            raise ValueError('At least one WTFTP loss weight must be positive.')
        loss = (
            low_weight * F.mse_loss(outputs['low_coeff'], true_low)
            + high_weight * F.mse_loss(outputs['high_coeff'], true_high)
            + time_weight * F.mse_loss(outputs['reconstruction'], roll_window)
            + forecast_weight * F.mse_loss(outputs['forecast'], roll_window[:, -1:, :])
        )
        return loss

    def _base_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=float(getattr(self.args, 'learning_rate', 1e-3)))

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion=None):
        """Validate the next-step forecast rather than the auxiliary DWT loss."""
        self.model.eval()
        weighted_error = 0.0
        count = 0
        for batch in vali_loader:
            x, y, _, _ = self._unpack_batch(batch)
            forecast = self.model(x)
            weighted_error += F.mse_loss(forecast, y, reduction='sum').item()
            count += y.numel()
        if count == 0:
            raise ValueError('Validation loader is empty.')
        return weighted_error / count

    def train(self, setting):
        _, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        if len(train_loader) == 0:
            raise ValueError('Training loader is empty.')
        ckpt_dir = os.path.join(
            getattr(self.args, 'checkpoints', './checkpoints'), setting)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')

        optimizer = self._select_optimizer()
        patience = int(getattr(self.args, 'patience', 5))
        max_epochs = int(getattr(self.args, 'train_epochs', 10))
        max_grad_norm = float(getattr(self.args, 'max_grad_norm', 0.0))
        best_val = float('inf')
        stale_epochs = 0
        for epoch in range(max_epochs):
            self.model.train()
            sum_loss = 0.0
            n_samples = 0
            for batch_index, batch in enumerate(train_loader):
                if batch_index >= int(getattr(self.args, 'max_train_batches', 0)) > 0:
                    break
                x, y, _, roll_window = self._unpack_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                if self.args.model == 'WTFTP':
                    outputs = self.model(x, return_aux=True)
                    loss = self._loss(roll_window, outputs)
                else:
                    loss = F.mse_loss(self.model(x), y)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                sum_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
            val_mse = self.vali(vali_data, vali_loader)
            train_loss = sum_loss / n_samples
            print(f'Epoch {epoch + 1:03d} | train loss: {train_loss:.6f} '
                  f'| val next-step MSE: {val_mse:.6f}')
            if np.isfinite(val_mse) and val_mse < best_val:
                best_val = val_mse
                stale_epochs = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print('Early stopping.')
                    break
        if not os.path.isfile(ckpt_path):
            raise RuntimeError('No valid checkpoint saved (possible NaN loss).')
        self._load_checkpoint(ckpt_path)
        scores = self._score_loader(vali_loader)['roll_score']
        threshold = float(np.quantile(scores, float(self.args.threshold_quantile)))
        root = Path(self.args.root_path)
        metadata = {
            'model': self.args.model, 'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len, 'enc_in': self.args.enc_in,
            'c_out': self.args.c_out, 'd_model': self.args.d_model,
            'e_layers': self.args.e_layers, 'd_layers': self.args.d_layers,
            'dropout': self.args.dropout, 'columns': vali_data.columns,
            'input_columns': vali_data.input_columns,
            'target_column': 'field.measured.roll',
            'training_objective': ('haar_coefficient_mse' if self.args.model == 'WTFTP'
                                   else 'next_roll_mse'),
            'normalization': vali_data.normalization,
            'normalization_offset': vali_data.offset.tolist(),
            'normalization_scale': vali_data.scale.tolist(),
            'mean': vali_data.mean.tolist(), 'std': vali_data.std.tolist(),
            'train_sha256': hashlib.sha256((root / self.args.train_file).read_bytes()).hexdigest(),
            'val_sha256': hashlib.sha256((root / self.args.val_file).read_bytes()).hexdigest(),
            'threshold_quantile': self.args.threshold_quantile,
            'roll_threshold': threshold, 'best_val_mse': best_val,
            'score_definition': 'squared normalized one-step roll forecast error',
        }
        (Path(ckpt_dir) / 'metadata.json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        return self.model

    def _load_checkpoint(self, path):
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # older torch
            state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    @torch.no_grad()
    def _score_loader(self, loader):
        self.model.eval()
        preds, trues, roll_scores, labels = [], [], [], []
        for batch in loader:
            x, y, label, _ = self._unpack_batch(batch)
            forecast = self.model(x)
            squared = (forecast - y).square()[:, 0, 0]
            preds.append(forecast[:, 0, :].cpu().numpy())
            trues.append(y[:, 0, :].cpu().numpy())
            roll_scores.append(squared.cpu().numpy())
            if label is not None:
                labels.append(label.cpu().numpy())
        if not preds:
            raise ValueError('Score loader is empty.')
        return {key: np.concatenate(value) for key, value in {
            'pred': preds, 'true': trues, 'roll_score': roll_scores, 'labels': labels,
        }.items() if value}

    @torch.no_grad()
    def test(self, setting, test=0):
        """Score each target row causally, using a validation-only threshold."""
        test_data, test_loader = self._get_data('test')
        ckpt_dir = Path(getattr(self.args, 'checkpoints', './checkpoints')) / setting
        metadata = json.loads((ckpt_dir / 'metadata.json').read_text(encoding='utf-8'))
        root = Path(self.args.root_path)
        for key in ('model', 'seq_len', 'pred_len', 'enc_in', 'c_out', 'd_model',
                    'e_layers', 'd_layers', 'dropout'):
            if metadata[key] != getattr(self.args, key):
                raise ValueError(f'Checkpoint configuration differs: {key}')
        if (metadata.get('normalization', 'standard') != test_data.normalization
                or metadata['columns'] != test_data.columns
                or metadata['input_columns'] != test_data.input_columns
                or metadata['target_column'] != 'field.measured.roll' or not np.allclose(
                metadata['mean'], test_data.mean) or not np.allclose(
                metadata['std'], test_data.std) or not np.allclose(
                metadata.get('normalization_offset', metadata['mean']), test_data.offset)
                or not np.allclose(
                metadata.get('normalization_scale', metadata['std']), test_data.scale)):
            raise ValueError('Test features or training normalization differ from checkpoint')
        for name in ('train', 'val'):
            if metadata[f'{name}_sha256'] != hashlib.sha256(
                    (root / getattr(self.args, f'{name}_file')).read_bytes()).hexdigest():
                raise ValueError(f'{name} CSV differs from checkpoint')
        if test:
            self._load_checkpoint(ckpt_dir / 'checkpoint.pth')
        arrays = self._score_loader(test_loader)
        rows = np.arange(test_data.seq_len, len(test_data.values))
        labels = test_data.labels[rows]
        np.testing.assert_array_equal(labels, arrays['labels'])
        score = arrays['roll_score']
        predictions = (score > metadata['roll_threshold']).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0)
        metrics = {
            'scored_points': len(rows), 'unscored_prefix': test_data.seq_len,
            'anomaly_points': int(labels.sum()),
            'threshold': metadata['roll_threshold'],
            'threshold_quantile': metadata['threshold_quantile'],
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'auroc': float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else None,
            'auprc': float(average_precision_score(labels, score)) if labels.any() else None,
            'mse_roll': float(np.mean((arrays['pred'] - arrays['true']) ** 2)),
            'mae_roll': float(np.mean(np.abs(arrays['pred'] - arrays['true']))),
        }
        result_dir = Path(getattr(self.args, 'result_path', './results')) / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_dir / 'scores.npz', row_index=rows, labels=labels,
                            roll_score=score,
                            predictions=predictions, pred=arrays['pred'], true=arrays['true'])
        (result_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n',
                                                  encoding='utf-8')
        print(json.dumps(metrics, indent=2))
        print(f'Saved: {result_dir}')
        return metrics

    @torch.no_grad()
    def predict(self, sequence, setting=None, load=False):
        """Predict roll from an already-normalized [B,9,17] array/tensor.

        Does not call data_provider. Returns a NumPy array [B,1,1].
        """
        if load:
            if setting is None:
                raise ValueError('Set setting when load=True.')
            path = os.path.join(getattr(self.args, 'checkpoints', './checkpoints'),
                                setting, 'checkpoint.pth')
            self._load_checkpoint(path)
        x = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
        self.model.eval()
        return self.model(x).cpu().numpy()
