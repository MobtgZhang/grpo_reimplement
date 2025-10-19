#!/bin/bash
# 生成随机端口号
function get_available_port() {
    while true; do
        port=$(shuf -i 20000-65535 -n 1)
        (echo >/dev/tcp/127.0.0.1/$port) >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo $port
            break
        fi
    done
}

# 获取可用端口
available_port=$(get_available_port)

echo "Available port: $available_port"
CUDA_VISIBLE_DEVICES=0,1,2,3
model_path=./models/Qwen/Qwen2.5-7B-Instruct
checkpoints_path="./checkpoints"
train_file=./Data/processed/train.json
save_model_name=qwen2.5-7b-instruct-grpo
deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $available_port  train.py \
    --model_path $model_path \
    --checkpoints_path $checkpoints_path \
    --train_file $train_file \
    --save_name $save_model_name \
    --styled \
    --shuffle_data \
    --max_epoches 2 \
    --save_every_steps 400 
