"""Reconstruction-based UAV anomaly detection, following TSLib's Exp pattern.

Dataset contract (data_provider(args, flag)):
    train/val: (x, labels) or (x, ...), x: [B, seq_len, enc_in]
    test:      (x, point_labels), labels: [B, seq_len] or [B, seq_len, 1]

The model must reconstruct *all* input channels and return [B, seq_len, enc_in].
Scores are per-time-point mean squared errors averaged over all channels.
No point adjustment; threshold fitted on training scores only.

For overlapping windows, the flattened scores/labels represent window positions,
not unique timestamps. Aggregate by original timestamps in your data pipeline
before reporting true point-level metrics.
"""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_pt_utils import get_parameter_names

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from ts_ad_evaluation import Evaluator


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        module = self.model_dict[self.args.model]
        model_class = module.Model if hasattr(module, "Model") else module
        model = model_class(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(
            "Model parameters: ",
            sum(
                param.numel()
                for param in self.model.parameters()
                if param.requires_grad
            ),
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _collect_scores(self, data_loader, collect_labels=False):
        self.model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch_x, batch_label = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                with self._autocast():
                    reconstructed = self._forward(batch_x)
                # [B, L, C] -> [B, L] (all channels participate).
                score = (reconstructed.float() - batch_x.float()).square().mean(dim=-1)
                scores.append(score.cpu().numpy().reshape(-1))

                if collect_labels:
                    if batch_label is None:
                        raise ValueError("Point-level labels required for evaluation")
                    label = torch.as_tensor(batch_label)
                    if label.ndim == 3 and label.shape[-1] == 1:
                        label = label[..., 0]

                    labels.append(label.cpu().numpy().reshape(-1))

        all_scores = np.concatenate(scores)
        all_labels = np.concatenate(labels) if collect_labels else None
        return all_scores, all_labels

    def test(self, setting, test=0):
        print(
            "Model parameters: ",
            sum(param.numel() for param in self.model.parameters()),
        )
        if test:
            print("loading model")
            self._load_checkpoint(setting)
            checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )

        # _, train_loader = self._get_data("train")
        _, test_loader = self._get_data("test")
        _, val_loader = self._get_data("val")

        val_scores = []
        test_scores = []
        test_labels = []

        # cal anomaly_scores
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                output = self.model(batch_x)
                # MAE score
                score = (output.float() - batch_x.float()).abs().mean(dim=-1)
                score = score.detach().cpu().numpy()
                val_scores.append(score)

            time_now = time.time()
            # test
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                output = self.model(batch_x)
                # MAE score
                score = (output.float() - batch_x.float()).abs().mean(dim=-1)
                score = score.detach().cpu().numpy()
                test_scores.append(score)
                test_labels.append(batch_y)

            print("Inference time: {}".format(time.time() - time_now))

        val_scores = np.concatenate(val_scores, axis=0).reshape(-1)
        val_scores = np.array(val_scores)
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
        test_scores = np.array(test_scores)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        result_dir = os.path.join("results", setting)
        os.makedirs(result_dir, exist_ok=True)

        init_scores = val_scores
        evaluator = Evaluator(
            gt=test_labels,
            anomaly_score=test_scores,
            save_path=result_dir,
            init_score=init_scores,
            flight_lengths=test_data.flight_lengths,
        )

        # evaluator.evaluate(method="percentile", percentile=99.5)
        evaluator.evaluate(method="3sigma", sigma=3)
        # evaluator.evaluate(method="spot", q=1e-3, level=0.98)

        return
