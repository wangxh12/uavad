#!/usr/bin/env bash
set -e

# Run from the project root, regardless of the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --model_id ALFA_bias_roll_1000_8_LSTM \
  --model LSTM \
  --data ALFA_Forecast \
  --root_path ./datasets/ALFA/forecast/bias_roll_1000_8 \
  --features MS \
  --target field.measured.roll \
  --seq_len 9 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 18 \
  --dec_in 18 \
  --c_out 1 \
  --d_model 64 \
  --e_layers 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --num_workers 0 \
  --train_epochs 30 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --weight_decay 0.0001 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --threshold_method percentile \
  --anomaly_ratio 1 \
  --threshold_sigma 3 \
  --smooth_rho 0.1 \
  --itr 1 \
  --des lstm_baseline