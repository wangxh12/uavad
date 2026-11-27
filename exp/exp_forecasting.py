import json
import os
import time
from contextlib import nullcontext

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.threshold import calculate_threshold
from ts_ad_evaluation.evaluator import Evaluator


class Exp_Forecasting(Exp_Basic):
    target_excluded_from_x = True

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def _autocast(self):
        if self.args.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss, n_samples = 0.0, 0
        f_dim = -1 if self.args.features == "MS" else 0

        for batch_x, batch_y, _ in vali_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            dec_inp = batch_x.new_zeros(
                (batch_x.size(0), self.args.pred_len, self.args.dec_in)
            )
            with self._autocast():
                outputs = self.model(batch_x, None, dec_inp, None)

            outputs = outputs[:, -self.args.pred_len :, :].float()
            target = batch_y[:, -self.args.pred_len :, :]

            loss = criterion(outputs, target)
            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        self.model.train()
        return total_loss / n_samples

    def train(self, setting):
        _, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.args.use_amp and self.device.type == "cuda"
        )
        f_dim = -1 if self.args.features == "MS" else 0

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()
            total_loss, n_samples = 0.0, 0

            for batch_x, batch_y, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # No historical target observations or calendar time features.
                dec_inp = batch_x.new_zeros(
                    (batch_x.size(0), self.args.pred_len, self.args.dec_in)
                )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(batch_x, None, dec_inp, None)

                outputs = outputs[:, -self.args.pred_len :, :].float()
                target = batch_y[:, -self.args.pred_len :, :]

                loss = criterion(outputs, target)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

            train_loss = total_loss / n_samples
            val_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"Train {train_loss:.6f} | Val {val_loss:.6f} | "
                f"{time.time() - epoch_start:.1f}s"
            )

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self._load_checkpoint(setting)
        return self.model

    def test(self, setting, test=0):
        # Pointwise labels[seq_len:] require single-step, stride-one forecasting.

        if test:
            print("Loading model checkpoint")
            self._load_checkpoint(setting)

        result_dir = os.path.join("results", setting)
        if (
            not os.path.exists(result_dir)
            and int(os.environ.get("LOCAL_RANK", "0")) == 0
        ):
            os.makedirs(result_dir)

        _, val_loader = self._get_data("val")
        _, test_loader = self._get_data("test")

        init_scores = []
        test_labels = []
        test_scores = []

        self.model.eval()
        with torch.no_grad():
            # collect validation scores
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = batch_x.new_zeros(
                    (batch_x.size(0), self.args.pred_len, self.args.dec_in)
                )
                with self._autocast():
                    outputs = self.model(batch_x, None, dec_inp, None)

                outputs = outputs[:, -self.args.pred_len :, :].float()
                target = batch_y[:, -self.args.pred_len :, :]
                init_scores.append(torch.abs(outputs - target).mean(dim=-1).cpu().numpy())

            # collect test scores
            for batch_x, batch_y, label in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = batch_x.new_zeros(
                    (batch_x.size(0), self.args.pred_len, self.args.dec_in)
                )
                with self._autocast():
                    outputs = self.model(batch_x, None, dec_inp, None)

                outputs = outputs[:, -self.args.pred_len :, :].float()
                target = batch_y[:, -self.args.pred_len :, :]
                test_scores.append(torch.abs(outputs - target).mean(dim=-1).cpu().numpy())
                test_labels.append(label.cpu().numpy())

        init_scores = np.concatenate(init_scores, axis=0)
        test_scores = np.concatenate(test_scores, axis=0)
        gt = np.concatenate(test_labels, axis=0)

        evaluator = Evaluator(
            init_scores, gt, test_scores, save_path=result_dir, args=self.args
        )
        evaluator.evaluate(
            metrics=self.args.metric,
            thres_method=self.args.thres_method,
            smooth_method=self.args.smooth_method,
        )
        
        return 
