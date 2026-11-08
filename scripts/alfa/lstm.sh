#!/bin/bash

python run.py \
    --task_name forecast \
    --is_training 1 \
    --model_id ALFA_Roll \
    --model LSTM_Forecast \
    --data ALFA \
    --root_path ./datasets/alfa_forecast1107/ \
    --features MS \
    --target field.measured.roll \
    --seq_len 9 \
    --label_len 0 \
    --pred_len 1 \
    --enc_in 17 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 64 \
    --e_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --anomaly_ratio 1.0 \
    --num_workers 0 \
    --itr 1 \
    --des test