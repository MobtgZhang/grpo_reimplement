from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import loguru 
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel,PreTrainedTokenizer
import deepspeed

from ce_kernel import compute_policy_loss_triton,compute_kl_divergence_triton

@dataclass
class GRPOConfig:
    """Configuration for LLM-specific GRPO."""
    epsilon: float = 0.8          # Clipping parameter
    beta: float = 0.05           # KL penalty coefficient
    group_size: int = 4         # Number of generations per prompt
    learning_rate: float = 3e-4
    max_grad_norm: float = 2.0
    max_new_tokens: int = 784
    temperature: float = 0.9     # Sampling temperature
    top_p: float = 0.9          # Nucleus sampling parameter
    top_k: int = 50          # Nucleus sampling parameter
class GRPO:
    """
    GRPO implementation specialized for Language Models.
    
    This implementation handles:
    - Token-based generation
    - Proper masking for variable-length sequences
    - Integration with HuggingFace models
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[GRPOConfig] = None,
        ds_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.device = device
        
        # Initialize optimizer with weight decay for non-bias parameters
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm'])], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm'])], 'weight_decay': 0.0}
        ]
        self.model = model
        
        # self.optimizer = torch.optim.AdamW(param_groups, lr=self.config.learning_rate)
        
        self.engine, _, _, _ = deepspeed.initialize(
            config=ds_config,
            model=model.to(device),
            model_parameters=param_groups,
        )
        
        self.engine.train()
        
        if dist.get_rank()==0:
            loguru.logger.info(f"Initialized LLM GRPO")
    def compute_sequence_level_advantages(
        self,
        rewards: torch.Tensor,
        group_indices: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages for complete sequences.
        
        Args:
            rewards: Tensor of shape (batch_size,) containing sequence-level rewards
            group_indices: Tensor of shape (batch_size,) containing group indices
            attention_mask: Tensor of shape (batch_size, seq_len) for masking
            
        Returns:
            Tensor of shape (batch_size,) containing sequence-level advantages
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_indices)
        
        for group_idx in unique_groups:
            mask = group_indices == group_idx
            group_rewards = rewards[mask]
            
            # Normalize advantages within each group
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std()
            if std_reward > 0:
                advantages[mask] = (group_rewards - mean_reward) / std_reward
            else:
                advantages[mask] = group_rewards - mean_reward
                
        return advantages

    def compute_token_level_advantages(
        self,
        sequence_advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand sequence-level advantages to token level.
        
        Args:
            sequence_advantages: Tensor of shape (batch_size,) with sequence advantages
            attention_mask: Tensor of shape (batch_size, seq_len) for masking
            
        Returns:
            Tensor of shape (batch_size, seq_len) with token-level advantages
        """
        # Expand advantages to token level and mask padding
        token_advantages = sequence_advantages.unsqueeze(-1).expand(-1, attention_mask.size(1))
        token_advantages = token_advantages * attention_mask.float()
        return token_advantages
    def compute_kl_divergence(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between old and new token distributions.
        """
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)
        
        kl = torch.sum(
            old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)),
            dim=-1
        )
        
        # Apply attention mask and average
        masked_kl = kl * attention_mask.float()
        return masked_kl.sum() / attention_mask.sum()
    def update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        group_indices: torch.Tensor,
        old_logits: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """
        Update policy using token-level GRPO.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token ids
            attention_mask: Tensor of shape (batch_size, seq_len) for masking
            rewards: Tensor of shape (batch_size,) with sequence-level rewards
            group_indices: Tensor of shape (batch_size,) with group indices
            old_logits: Tensor of shape (batch_size, seq_len, vocab_size) with old logits
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        rewards = rewards.to(self.device)
        group_indices = group_indices.to(self.device)
        old_logits = old_logits.to(self.device)
        # Compute sequence-level advantages
        advantages = self.compute_sequence_level_advantages(rewards, group_indices, attention_mask)     
        # Expand to token level
        token_advantages = self.compute_token_level_advantages(advantages, attention_mask)
        
        # Forward pass for new logits
        outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        new_logits = outputs.logits
        # policy_loss = compute_policy_loss_triton(old_logits, new_logits, token_advantages,input_ids, attention_mask)
        policy_loss = self.compute_policy_loss(old_logits, new_logits, token_advantages,input_ids, attention_mask)
        
        # kl_loss = compute_kl_divergence_triton(old_logits, new_logits, attention_mask)
        kl_loss = self.compute_kl_divergence(old_logits, new_logits, attention_mask)
        total_loss = policy_loss + self.config.beta * kl_loss
        # 注意这里需要改成deepspeed的更新方式
        # Optimize
        self.engine.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.engine.module.parameters(), self.config.max_grad_norm)
        self.engine.step()
        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
        }
    def compute_policy_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        advantages: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-level policy loss with proper masking.
        """
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)
        
        # Get probabilities for actual tokens
        old_token_probs = torch.gather(
            old_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        new_token_probs = torch.gather(
            new_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute probability ratio
        ratio = new_token_probs / (old_token_probs + 0.1)
        # 限制 ratio 的范围，防止梯度爆炸
        ratio = torch.clamp(ratio, 0.1, 10.0)
        # 限制 advantages 大小，防止 loss 过大
        advantages = torch.clamp(advantages, -10, 10)
        # Compute surrogate losses
        surr1 = ratio * advantages
        
        surr2 = torch.clamp(
            ratio,
            1 - self.config.epsilon,
            1 + self.config.epsilon
        )
        #surr2 = torch.clamp(
        #    ratio,
        #    1 - self.config.epsilon,
        #    1 + self.config.epsilon
        #) * advantages
        
        # Apply attention mask and average
        policy_loss = -torch.min(surr1, surr2) * attention_mask.float()
        return policy_loss.sum() / attention_mask.sum()
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        num_samples: int = 1,
        **generation_kwargs
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            prompts: List of input prompts
            num_samples: Number of generations per prompt
            generation_kwargs: Additional kwargs for model.generate()
            
        Returns:
            Tuple containing:
                - List of lists of generated texts
                - Tensor of logits for each generation
        """
        # Encode prompts
        generated_tokens= self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch_size = generated_tokens.input_ids.shape[0]
        
        # Repeat inputs for multiple samples
        input_ids = generated_tokens.input_ids.repeat_interleave(num_samples, dim=0)
        attention_mask = generated_tokens.attention_mask.repeat_interleave(num_samples, dim=0)
        # Generate responses
        outputs = self.engine.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_k = self.config.top_k,
            top_p = self.config.top_p,
            temperature = self.config.temperature,
            output_scores=True,             # 返回每一步的 logits       
            return_dict_in_generate=True,  # 返回 dict 而不是 tensor
            pad_token_id = self.tokenizer.eos_token_id
        )
        generated_ids = outputs.sequences
        logits_list = outputs.scores
        prompt_len = generated_tokens["input_ids"].shape[1]
        generated_texts = self.tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
        # 这里需要将 batch_size * num_samples，变成 (batch_size，num_samples)
        generated_texts = [
            generated_texts[i * num_samples : (i + 1) * num_samples] 
            for i in range(batch_size)
        ]
        logits = torch.stack(logits_list, dim=1)
        return generated_texts, logits
