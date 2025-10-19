CUDA_VISIBLE_DEVICES=0,1,2,3
train_file=./data/Qwen2.5-7B-Instruct/styled.pth
model_path=./models/Qwen/Qwen2.5-7B-Instruct
save_name=qwen2.5-7b-instruct-styled
deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port 45125 main.py \
    --train_file $train_file \
    --model_path $model_path \
    --is_save_epoch 