from typing import Dict
import re
from collections import Counter
import math
import os
import pathlib
import datetime
import random
import json
import argparse

from tqdm import tqdm
import loguru
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader,DistributedSampler
import deepspeed
from peft import get_peft_model, PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# ===============================
from grpo import GRPO, GRPOConfig
from dataset import GRPORLDataset
from draw_results import draw_loss_curve

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1245)
    parser.add_argument("--model_path", type=str, default="",required=True)
    parser.add_argument("--config_path", type=str, default="./config")
    parser.add_argument("--log_path", type=str, default="./training_logs")
    parser.add_argument("--checkpoints_path", type=str, default="./checkpoints")
    parser.add_argument("--train_file", type=str, default=None,required=True)
    # GRPO groups size
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_epoches", type=int, default=4)
    
    parser.add_argument("--save_name", type=str, default=None,required=True)
    parser.add_argument("--styled", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    # 文件名是否需要时间戳
    parser.add_argument("--no_timestamp",action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    
    parser.add_argument("--shuffle_data", action="store_true")
    
    # GRPO learning rate
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    # GRPO max sequence length
    parser.add_argument("--max_new_tokens", type=int, default=784)
    args = parser.parse_args()
    return args
def set_seed(seed):
    """设置随机数种子, 保证结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def setup_distributed_environment(local_rank):
    """设置分布式训练环境"""
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
    if dist.get_rank()==0:
        loguru.logger.info("\n" + "=" * 20 + "\nDistributed environment is initialized.\n" + "=" * 20)
    return device
def initialize_model(args,device):
    """加载和初始化模型"""
    if dist.get_rank()==0:
        loguru.logger.info("\n" + "=" * 20 + "\nLoading model...\n" + "=" * 20 + "\n")
    # 加载tokenizer以及Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    # Set proper padding in tokenizer
    tokenizer.padding_side = 'left'  # Important for decoder-only models
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
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
    # optimizer = MemoryEfficientAdamW(model.parameters(),lr=args.learning_rate,)
    return tokenizer,model


def ngram_bleu(reference, candidate, n=4):
    weights = [0.25] * n
    p_ns = []
    for i in range(1, n+1):
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i)]))
        cand_ngrams = Counter(zip(*[candidate[j:] for j in range(i)]))
        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = max(1, sum(cand_ngrams.values()))
        p_ns.append(overlap / total)
    
    # brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    bp = math.exp(1 - ref_len / cand_len) if cand_len < ref_len else 1.0
    
    return bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, p_ns) if p > 0))

class RewardModel:
    def __init__(self):
        pass
    def compute_rewards(self,flat_generations,flat_problems):
        """
        Compute reward scores for generated texts.
        Returns a tensor of rewards in the range [0, 1].
        """
        # 如果是一个选择题，那么就直接验证答案是否正确
        # 如果是一个开放式问题，那么就直接使用BLEU计算相似度
        rewards = []
        pattern = r"<think>([\s\S]*?)</think><answer>([\s\S]*?)</answer>"
        # pattern = r"^<think>(.*?)</think><answer>(.*?)</answer>$"
        for flat_ans,data_dict in zip(flat_generations,flat_problems):
            # 首先验证是否符合格式要求
            results = re.search(pattern, flat_ans)
            if results:
                think, flat_ans = results.groups()
                if "options" in data_dict:
                    targ_ans = data_dict["answer"].split(",")
                    flat_ans = flat_ans.split(",")
                    if any(len(x)==1 for x in flat_ans):
                        if len(flat_ans) == len(targ_ans):
                            score = 1.0
                            for ans in targ_ans:
                                if ans not in flat_ans:
                                    score = 0.0
                                    break
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                    rewards.append(score)
                else:
                    targ_ans = data_dict["answer"]
                    value = ngram_bleu(flat_ans,targ_ans)
                    rewards.append(value)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards)


class TrainingLogger:
    """
    Simple training logger that saves metrics to disk and prints updates.
    Maintains running statistics for easy progress monitoring.
    """
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_history = []
        self.running_stats = {}
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a single training step."""
        metrics['step'] = step
        self.metrics_history.append(metrics)
        
        # Update running statistics
        for key, value in metrics.items():
            if key not in self.running_stats:
                self.running_stats[key] = []
            self.running_stats[key].append(value)
            if len(self.running_stats[key]) > 100:  # Keep last 100 values
                self.running_stats[key].pop(0)
    
    def get_running_averages(self) -> Dict[str, float]:
        """Get running averages of all metrics."""
        return {
            k: np.mean(v) for k, v in self.running_stats.items()
            if k != 'step'
        }
    def get_history(self):
        return self.metrics_history
    def save_logs(self,save_name):
        """Save all metrics to disk."""
        with open(self.log_dir / f'metrics_step{save_name}.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
def prepare_dataloader(args,ds_config,dataset):
    """准备数据加载器"""
    if dist.get_rank()==0:
        loguru.logger.info("\n" + "=" * 20 + "\nLoading dataset...\n" + "=" * 20 + "\n")
    
    sampler = DistributedSampler(dataset, shuffle=args.shuffle_data) if args.local_rank != -1 else None
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        collate_fn=dataset.collate_fn,
    )
    if dist.get_rank()==0:
        loguru.logger.info("\n" + "=" * 20 + "\nDataset is loaded.\n" + "=" * 20 + "\n")
    return dataloader
def train_step(grpo: GRPO,global_step,batch,reward_model=None):
    """
    Perform a single training step with proper sequence length handling.
    """
    
    # Generate responses with explicit max length
    generations, logits = grpo.generate(
        batch["prompts"],
        num_samples=grpo.config.group_size,
        max_new_tokens=grpo.config.max_new_tokens,
        pad_token_id=grpo.tokenizer.pad_token_id
    )
    # if dist.get_rank()==0:
    #     loguru.logger.info(f"Step {global_step}: Generated {len(generations)} responses")
    # Flatten generations for reward computation
    flat_generations = [text for sublist in generations for text in sublist]
    flat_problems = [item for _ in range(grpo.config.group_size) for item in batch["problem"]]
    
    # Compute rewards
    rewards = reward_model.compute_rewards(flat_generations,flat_problems)
    #if dist.get_rank()==0:
    #    loguru.logger.info(f"The Rewards:\n{rewards}")
        # Create group indices
    group_indices = torch.arange(len(flat_generations)) // grpo.config.group_size
    
    # Tokenize with careful length handling
    encoded = grpo.tokenizer(
        flat_generations,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    max_len = min(encoded.input_ids.size(1), logits.size(1))
    input_ids = encoded.input_ids[:, -max_len:]
    attention_mask = encoded.attention_mask[:, -max_len:]
    logits = logits[:, :max_len]
    # Update policy with aligned tensors
    metrics = grpo.update(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rewards=rewards,
        group_indices=group_indices,
        old_logits=logits,
    )
    
    metrics.update({
        "mean_reward": rewards.mean().item(),
        "max_reward": rewards.max().item(),
        "min_reward": rewards.min().item(),
        "sequence_length": max_len,
    })
    return metrics
def save_pretrained(args,grpo,logger,epoch,global_step):
    # Save final model and logs
    dist.barrier()
    if dist.get_rank() == 0:
        save_model_step_name = f"{global_step}_{epoch}"
        # 时间戳
        if args.no_timestamp:
            pass
        else:
            # save start time
            t = datetime.datetime.now()
            save_timestamp = f"{t.year}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}"
            save_model_step_name = f"{save_model_step_name}_{save_timestamp}"
        if not args.save_name:
            args.save_name = pathlib.Path(args.model_path).name
        else:
            pass
        loguru.logger.info("\nSaving model and logs...")
        # 所以说模型保存的地址如下所示
        save_model_path = os.path.join(args.checkpoints_path,args.save_name,save_model_step_name)
        os.makedirs(save_model_path, exist_ok=True)
        logger.save_logs(args.save_name)
        grpo.engine.module.save_pretrained(save_model_path,safe_serialization=True,torch_type=torch.bfloat16)
        grpo.tokenizer.save_pretrained(save_model_path)
        # 这里我感觉缺少画出来的图，包括有整体训练过程的loss曲线图，以及reward变化曲线图
        try:
            # 加载
            metric_loss_dict_list = logger.get_history()
            # 画出loss曲线图
            losses_list = [item["total_loss"] for item in metric_loss_dict_list]
            draw_loss_curve(losses_list, save_model_path, args.save_name+"_loss", title="Training Loss Curve", xlabel="Steps", ylabel="Loss")
            # 画出reward曲线图
            rewards_list = [item["mean_reward"] for item in metric_loss_dict_list]
            draw_loss_curve(rewards_list, save_model_path, args.save_name+"_reward", title="Training Reward Curve", xlabel="Steps", ylabel="Reward")
        except ImportError as e:
            loguru.logger.warning(f"Failed to draw loss curve: {e}")
    dist.barrier()
def main():
    args = get_args()
    set_seed(args.seed)
    # Initilize the deepspeed distribution environment
    device = setup_distributed_environment(args.local_rank)
    ds_config_file = os.path.join(args.config_path, "deepspeed_config.json")
    with open(ds_config_file, "r") as rfp:
        ds_config = json.load(rfp)
    
    # Initialize reward model and logger
    logger = TrainingLogger(args.log_path)
    # 加载训练数据
    dataset = GRPORLDataset(args.train_file,args.model_path,args.config_path,args.styled)
    dataloader = prepare_dataloader(args,ds_config,dataset)
    
    # Initialize models and tokenizer
    if dist.get_rank()==0:
        loguru.logger.info("Initializing models...")
    tokenizer,model = initialize_model(args,device)
    # Initialize GRPO with custom configuration
    grpo = GRPO(
        model=model,
        tokenizer=tokenizer,
        config=GRPOConfig(
            group_size=args.group_size,  # Number of generations per prompt
            learning_rate=args.learning_rate,
            max_new_tokens=args.max_new_tokens
        ),
        ds_config=ds_config,
        device=device # Use GPU for faster computation
    )

    # Training loop
        # Training loop
    if dist.get_rank()==0:
        loguru.logger.info("Starting training...")
    # num_epochs = 3
    reward_model  = RewardModel()
    global_step = 0
    total_steps = args.max_epoches * len(dataloader)
    main_pbar = tqdm(total=total_steps,ncols=100)
    
    for epoch in range(args.max_epoches):
        # Progress bar for each epoch
        for batch_dict in dataloader:
            # Train GRPO
            metrics = train_step(grpo,global_step,batch_dict,reward_model=reward_model)
            logger.log_metrics(metrics, global_step)
            global_step += 1
            # Update progress bar with current metrics
            running_avgs = logger.get_running_averages()
            # pbar.set_postfix({
            #    'loss': f"{running_avgs['total_loss']:.4f}",
            #    'reward': f"{running_avgs['mean_reward']:.4f}"
            #})
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_pretrained(args,grpo,logger,epoch,global_step)
            if dist.get_rank()==0:
                main_pbar.update()
                main_pbar.set_postfix({
                    'epoch':f"{epoch+1}/{args.max_epoches}",
                    'loss': f"{running_avgs['total_loss']:.4f}",
                    'reward': f"{running_avgs['mean_reward']:.4f}"
                })
        if args.save_every_epoch:
            save_pretrained(args,grpo,logger,epoch,global_step)
    # 保证最后一个epoch的模型被保存
    if not args.save_every_epoch and dist.get_rank()==0:
        save_pretrained(args,grpo,logger,epoch,global_step)
if __name__ == '__main__':
    main()
