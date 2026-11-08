"""Train/test WTFTP reconstruction on prepared ALFA real-fault segments."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from exp.exp_reconstruction import Exp_Reconstruction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--is_training', type=int, choices=[0, 1], default=1)
    parser.add_argument('--model', choices=['WTFTP'], default='WTFTP')
    parser.add_argument('--root_path', default='data/processed/alfa_within_flight_v1')
    parser.add_argument('--setting', default=None)
    parser.add_argument('--checkpoints', default='checkpoints/reconstruction')
    parser.add_argument('--result_path', default='results/reconstruction')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=18)
    parser.add_argument('--c_out', type=int, default=18)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='Nonzero only for a small smoke run; validation/test remain complete')
    parser.add_argument('--threshold_quantile', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--torch_threads', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.seq_len < 2 or args.seq_len % 2 or args.enc_in != 18 or args.c_out != 18 or
            args.d_model < 2 or args.d_model % 2 or args.e_layers < 1 or args.d_layers != 1):
        parser.error('WTFTP requires an even window/d_model, 18 channels, e_layers >= 1 and d_layers = 1')
    if (min(args.batch_size, args.train_epochs, args.patience, args.torch_threads) < 1 or
            args.max_train_batches < 0 or args.num_workers < 0 or args.gpu < 0 or
            not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not np.isfinite(args.max_grad_norm) or args.max_grad_norm < 0 or
            not 0 <= args.dropout < 1 or not 0 < args.threshold_quantile < 1):
        parser.error('Invalid training, worker, device, dropout or threshold settings')
    if args.setting is None:
        args.setting = f'WTFTP_alfa_real_reconstruction_w{args.seq_len}_seed{args.seed}'
    if args.setting in {'', '.', '..'} or '/' in args.setting or '\\' in args.setting:
        parser.error('--setting must be a directory name, not a path')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available')
    args.use_gpu = args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
    args.use_multi_gpu = False
    args.task_name = 'anomaly_detection'
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(__file__).resolve().parent
    for name in ('root_path', 'checkpoints', 'result_path'):
        path = Path(getattr(args, name))
        setattr(args, name, str(path if path.is_absolute() else root / path))
    experiment = Exp_Reconstruction(args)
    if args.is_training:
        experiment.train(args.setting)
    return experiment.test(args.setting)


if __name__ == '__main__':
    main()
