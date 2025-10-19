# 用于大语言模型微调的相关代码
import os
import random
import json
import datetime
import argparse
import pathlib
import loguru
from tqdm import tqdm
import numpy as np
import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
from peft import get_peft_model, PeftModel, LoraConfig
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer,AutoModelForCausalLM

# ===================================================
from dataset import MultipleKnowledgeDataset
from draw_results import draw_loss
    
def set_seed(seed):
    """
    设置随机数种子, 保证结果可重现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_path",type=str,default="./checkpoints")
    parser.add_argument("--model_path",type=str,default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load_ckpt_path",type=str,default=None)
    parser.add_argument("--config_path",type=str,default="./config")
    parser.add_argument("--save_name",type=str,default=None)
    parser.add_argument("--train_file",type=str,default="./data/Qwen2.5-7B-Instruct/styled.pth")
    parser.add_argument("--use_lora",action="store_true")
    
    # 文件名是否需要时间戳
    parser.add_argument("--no_timestamp",action="store_true")
    # distribute params
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--shuffle_data",action="store_true")
    # 设置随机数
    parser.add_argument("--seed",default=1234,type=int)
    parser.add_argument("--is_save_epoch",action="store_true")
    
    # 设置最大训练步数
    parser.add_argument("--max_steps",default=None,type=int)
    # 设置保存截止步数
    parser.add_argument("--save_steps",default=None,type=int)
    # 设置最大训练次数
    parser.add_argument("--max_epoches",default=4,type=int)
    # 设置最大截断梯度值
    parser.add_argument("--max_grad_norm", type=float, default=3.0, help="Max gradient norm for gradient clipping")
    
    # 是否需要保存当前的优化器状态？
    parser.add_argument("--save_optimizer",action="store_true")
    args = parser.parse_args()
    return args
def initialize_args(args):
    set_seed(args.seed)
    # 保存目录一般设置为 checkpoints/model-name_steps_epoches_timestamp
    if args.load_ckpt_path:
        load_ckpt_name = pathlib.Path(args.load_ckpt_path).name
        args.save_name = load_ckpt_name
        outs = load_ckpt_name.split("_")
        if len(outs)==4:
            model_name,start_steps,start_epoch,timestamp = outs
        elif len(outs)==3:
            model_name,start_steps,start_epoch = outs
        else:
            raise ValueError(f"Unknown: {args.load_ckpt_path}")
        args.start_steps = start_steps
        args.start_epoch = start_epoch      
        args.model_name = model_name  
        # 下一次训练的数量应该是在上一次基础上增加，所以应该是训练之后再确定保存文件路径
    else:
        args.start_steps = 0
        args.start_epoch = 0   
        # 设置模型名称
        if args.save_name:
            pass
        else:
            args.save_name = pathlib.Path(args.model_path).name
    #save_path = os.path.join(args.checkpoints_path,args.save_name)
    #os.makedirs(save_path, exist_ok=True)
    #return deepspeed_config
def setup_distributed_environment(local_rank):
    """
    配置分布式训练环境
    """
    if local_rank != -1:  # 使用分布式训练
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=torch.cuda.device_count(),
        )
    else:  # 单卡训练
        device = torch.device("cuda")
    deepspeed.init_distributed()
    # 注意所有的logger info 全都集中在 local_rank=0的情况下
    if torch.distributed.get_rank() == 0:
        loguru.logger.info("Distributed environment is initialized.")
    return device
def prepare_dsloader(args):
    if torch.distributed.get_rank() == 0:
        loguru.logger.info("Loading dataset...")
    # 设置deepspeed配置文件
    deepspeed_config_file = os.path.join(args.config_path,"deepspeed_config.json")
    with open(deepspeed_config_file, "r", encoding="utf-8") as f:
        deepspeed_config = json.load(f)
    # 加载数据集
    train_set = MultipleKnowledgeDataset(args.train_file,args.model_path)

    ds_set_size = len(train_set)
    # 根据数据量区间调整 global batch size（可选，如果你想让大数据更稳）
    if ds_set_size <= 300:
        target_global_batch_size = 32
    elif ds_set_size <= 1000:
        target_global_batch_size = 48
    elif ds_set_size <= 5000:
        target_global_batch_size = 64
    else:
        target_global_batch_size = 96
    micro_batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
    num_gpus = torch.cuda.device_count()
    # 计算梯度累积步数
    gas = max(1, target_global_batch_size // (micro_batch_size * num_gpus))
    deepspeed_config["gradient_accumulation_steps"] = gas
    # sampler 函数选择
    train_sampler = DistributedSampler(train_set, shuffle=args.shuffle_data) if args.local_rank != -1 else None
    train_dsloader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
        collate_fn=train_set.collate_fn
        # drop_last=True,
    )
    loguru.logger.info("Dataset is loaded.")
    return train_dsloader
def initialize_model(args,device):
    loguru.logger.info("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        ).to(device)
        if args.load_ckpt_path:
            # load_ckpt_path = os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
            model = PeftModel.from_pretrained(model, args.load_ckpt_path, is_trainable=True)
        else:
            lora_config_file = os.path.join(args.config_path,"lora_config.json")
            with open(lora_config_file,mode="r",encoding="utf-8") as rfp:
                lora_config = LoraConfig(**json.load(rfp))
            model = get_peft_model(model, lora_config)
    else:
        # model_path = args.model_path if not args.load_ckpt_path else os.path.join(args.load_ckpt_path, args.ckpt_path, f"step_{args.load_ckpt_step}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        ).to(device)
    return model, tokenizer  
def train_model(args,model, tokenizer, train_ds_loader, device):
    # 加载deepspeed 配置文件
    deepspeed_config_file = os.path.join(args.config_path,"deepspeed_config.json")
    with open(deepspeed_config_file, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
    # 定义deepspeed引擎
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )
    # 开启训练模式
    engine.train()
    # 初始化step和epoch
    begin_epoch = 0  # 第一个epoch编号（加载存档点时有变化）
    current_steps = 0  # 加载存档点时，第一个epoch需要从第几个batch开始
    end_epoch = args.max_epoches
    
    # 如果需要加载检查点
    if args.load_ckpt_path:
        # 保存目录一般设置为 checkpoints/model_name/numbers_epoches_timestamp
        if args.local_rank == -1 or dist.get_rank() == 0:
            loguru.logger.info("Loading ckpt...")
        # 加载优化器状态，可能未保存
        try:
            engine.load_checkpoint(args.load_ckpt_path)
        except:
            pass
        # 加载训练过程的损失值
        
        loss_fn = os.path.join(args.load_ckpt_path, "loss.json")
        with open(loss_fn, "r") as f:
            losses_list = json.load(f)
    
        # 恢复 step 和 epoch
        current_steps = args.start_steps 
        begin_epoch = args.start_epoch
    
    # 初始化进度条
    if args.local_rank == -1 or dist.get_rank() == 0:
        # 注意训练过程是在上述基础上继续进行训练，所以有以下的内容：
        if args.max_steps:
            total_train_steps = min(args.max_steps,(args.max_epoches - begin_epoch)*len(train_ds_loader))
        else:
            total_train_steps = (args.max_epoches - begin_epoch)*len(train_ds_loader)
        main_pbar = tqdm(total=total_train_steps, ncols=95)
        
    # 训练过程
    losses_list = []
    for epoch in range(begin_epoch, end_epoch + 1):
        # 如果使用了 DistributedSampler，需要在每个 epoch 调用 set_epoch
        if args.local_rank != -1 and isinstance(train_ds_loader.sampler, DistributedSampler):
            train_ds_loader.sampler.set_epoch(epoch)
        for batch_id, batch in enumerate(train_ds_loader):
            # 这里进行encode 处理
            batch = {k: v.to(device) for k, v in batch.items()}  # 把输入放进显卡
            # 前向传播，计算loss，反向传播
            loss = engine(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).loss
            # 向前传播
            engine.backward(loss)
            # 梯度截断
            torch.nn.utils.clip_grad_norm_(engine.module.parameters(), max_norm=args.max_grad_norm)
            # optimizer 更新
            engine.step()
            current_steps += 1
            losses_list.append(loss.item())
            # 更新训练进度条
            if args.local_rank == -1 or dist.get_rank() == 0:
                main_pbar.update()
                # np.mean(losses[-200:])
                main_pbar.set_description(f"epoch:{epoch},batch:{batch_id + 1}/{len(train_ds_loader)}, loss:{loss.item():.4f}")
            
            # 如果达到最大步数，停止训练
            if args.max_steps and current_steps >= args.max_steps:
                break
            # 如果达到保存步数，保存模型
            if args.save_steps and current_steps % args.save_steps == 0:
                save_checkpoint(args,engine, tokenizer, current_steps,epoch, losses_list)
        # 是否在每个epoch结束时保存模型
        if args.is_save_epoch:
            save_checkpoint(args,engine, tokenizer, current_steps,epoch, losses_list)
    # 确保保存了最后一个存档点
    if not args.is_save_epoch or (args.max_steps and current_steps >= args.max_steps):
        save_checkpoint(args,engine, tokenizer, current_steps,epoch, losses_list)

    if args.local_rank == -1 or dist.get_rank() == 0:
        main_pbar.close()

def save_checkpoint(args,engine, tokenizer, current_steps,current_epoch, losses_list):
    # 保存模型和训练损失
    # 这里来定义模型保存的地址
    # 模型保存的地址为，checkpoints/model_name/steps_epoches_timestamp
    save_model_step_name = f"{current_steps}_{current_epoch}"
    # 时间戳
    if args.no_timestamp:
        pass
    else:
        # save start time
        t = datetime.datetime.now()
        save_timestamp = f"{t.year}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}"
        save_model_step_name = f"{save_model_step_name}_{save_timestamp}"
    # 所以说模型保存的地址如下所示
    save_model_path = os.path.join(args.checkpoints_path,args.save_name,save_model_step_name)
    os.makedirs(save_model_path, exist_ok=True)

    # 使用 engine.save_checkpoint 保存完整的训练状态和deepspeed格式模型
    if args.save_optimizer:
        engine.save_checkpoint(save_model_path)

    # 保存pytorch .bin格式模型
    # engine.save_16bit_model(save_path)
    
    # 保存safetensor格式
    engine.module.save_pretrained(save_model_path, torch_dtype=torch.bfloat16, safe_serialization=True)

    # 保存tokenizer
    tokenizer.save_pretrained(save_model_path)

    # 获取当前配置，并且进行保存config
    saved_config = engine.module.config.to_dict()
    
    # 保存配置
    with open(os.path.join(save_model_path, 'ckpt_config.json'), 'w') as f:
        json.dump(saved_config, f, indent=4)

    # 保存损失函数
    save_loss_file = os.path.join(save_model_path, "loss_list.json")
    if args.local_rank == -1 or dist.get_rank() == 0:
        with open(save_loss_file, "w") as f:
            json.dump(losses_list, f)
        # 这里可以画出一下具体的损失函数图
        try:
            draw_loss(losses_list,save_model_path)
        except ImportError as e:
            if args.local_rank == -1 or dist.get_rank() == 0:
                loguru.logger.info("The loss picture is not draw")

def main():
    args = get_args()
    # 初始化训练参数
    initialize_args(args)
    # 设置分布式训练环境
    device = setup_distributed_environment(args.local_rank)
    # 初始化数据集
    train_ds_loader = prepare_dsloader(args)
    
    # 设置模型
    model,tokenizer = initialize_model(args,device)
    
    # 开始训练模型
    train_model(args,model,tokenizer, train_ds_loader, device)
    
if __name__ == "__main__":
    main()
