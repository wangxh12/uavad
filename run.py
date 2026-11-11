import argparse
import random

import numpy as np
import torch

from utils.print_args import print_args


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="UAV Anomaly Detection")

    # Basic configuration
    parser.add_argument("--task_name", type=str, required=True, choices=["forecast", "anomaly_detection"])
    parser.add_argument("--is_training", type=int, choices=[0, 1], default=1, help="1: train and test; 0: test a saved checkpoint")
    parser.add_argument("--model_id", type=str, default="UAV")
    parser.add_argument("--model", type=str, required=True, help="Model name, matching a Python module under models/")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run_idx", type=int, default=0, help="Run index of the checkpoint used when is_training=0")

    # Data loading    
    parser.add_argument('--data', type=str, default='MSL', help='dataset type')
    parser.add_argument('--split_config', type=str, default=None, help='Explicit time-split JSON for WideCSV/TLM_UAV (path relative to working directory)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--label_file", type=str, default="label.csv", help="CSV with one binary label column, aligned with test rows")
    parser.add_argument("--normalization", choices=["standard", "minmax"], default="standard", help="Fit scaling on training rows only")
    parser.add_argument("--features", type=str, choices=["M", "S", "MS"], default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of input data')
    parser.add_argument('--step', type=int, default=1, help='step')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")

    # Task configuration
    parser.add_argument("--seq_len", type=int, default=9, help="Input length for both tasks")
    parser.add_argument("--label_len", type=int, default=0, help="Decoder known-context length, if applicable")
    parser.add_argument("--pred_len", type=int, default=1, help="Forecast horizon; not used for reconstruction")
    parser.add_argument("--anomaly_ratio", type=float, default=1.0, help="Anomaly ratio in percent (TSLib convention: 1.0 means 1%%)")

    # Common model configuration
    parser.add_argument("--enc_in", type=int, required=True,
                        help="Number of input flight variables")
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--distil", action="store_false", default=True,
                        help="Disable encoder distillation when provided")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    # Optimization
    # parser.add_argument('--max_iters', type=int, default=100000, help='for DADA')
    parser.add_argument("--percentage", type=float, default=1, help="the percentage(*100) of train data")
    parser.add_argument('--des', type=str, default='zero_shot', help='exp description')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='finetuning epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', choices=['type1', 'type2', 'type3', 'cosine', 'constant'], default='constant', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup steps')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--decay_fac', type=float, default=0.75, help='adam beta2') 

    # Device configuration
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default="0,1")
    
    # Bi-AEWMA
    parser.add_argument('--use_biaewma', action='store_true')
    parser.add_argument('--smooth_rho', type=float, default=None)
    
    # Threshold 
    parser.add_argument('--threshold_method',type=str,default='percentile',choices=['percentile', '3sigma', 'mad', 'iqr'])
    parser.add_argument('--threshold_source',type=str,default='val',choices=['train', 'val', 'train_val', 'train_test'])
    parser.add_argument('--threshold_sigma', type=float, default=3.0)
    parser.add_argument('--threshold_mad_k', type=float, default=3.0)
    parser.add_argument('--threshold_iqr_k', type=float, default=1.5)
    
    # TimeCNN
    parser.add_argument('--conv_dropout', type=float, default=0.0)
    parser.add_argument('--use_norm', type=int, default=1)
    
    # FACT
    parser.add_argument('--core', type=float, default=0.5)
    parser.add_argument('--dilation', type=int, nargs='+', default=[1, 2])
    # parser.add_argument('--num_kernels', type=int, default=3)

    args = parser.parse_args(argv)

    if args.dec_in is None:
        args.dec_in = args.enc_in
    if args.c_out is None:
        args.c_out = args.enc_in

    # For reconstruction, model output should match the seq_len input.
    # pred_len is a forecast-only parameter and is not forcibly overwritten.
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_multi_gpu and args.use_gpu:
        try:
            args.device_ids = [int(x) for x in args.devices.replace(" ", "").split(",")]
        except ValueError:
            parser.error("devices must be comma-separated GPU IDs, e.g. 0,1")
        if not args.device_ids:
            parser.error("devices must contain at least one GPU ID")
        args.gpu = args.device_ids[0]
    else:
        args.use_multi_gpu = False

    args.device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    if args.use_amp and not args.use_gpu:
        print("[Warning] AMP disabled because CUDA is unavailable or disabled.")
        args.use_amp = False
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(args, run_idx):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
        f"_dl{args.d_layers}_df{args.d_ff}_{args.des}_{run_idx}"
    )


def main(argv=None):
    args = get_args(argv)
    print('Args in experiment:')
    print_args(args)

    if args.task_name == "forecast":
        from exp.exp_forecasting import Exp_Forecasting as Exp
    else:
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection as Exp


    run_indices = range(args.itr) if args.is_training else [args.run_idx]
    for run_idx in run_indices:
        set_seed(args.seed + run_idx)
        setting = build_setting(args, run_idx)
        exp = Exp(args)

        if args.is_training:
            print(f"\n>>>>>>> Start training: {setting}")
            exp.train(setting)
            print(f"\n>>>>>>> Start testing: {setting}")
            exp.test(setting)
        else:
            print(f"\n>>>>>>> Test saved model: {setting}")
            exp.test(setting, test=1)

        del exp
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


