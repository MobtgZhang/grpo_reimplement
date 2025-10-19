import torch
import triton
import triton.language as tl
import math

# --------------------------
# KL kernel (chunked over V) - fixed (no tl.constant)
# --------------------------
@triton.jit
def compute_kl_kernel_chunked(
    old_logits_ptr, new_logits_ptr, attention_mask_ptr, output_ptr,
    V: tl.constexpr, BLOCK_V: tl.constexpr, NUM_BLOCKS: tl.constexpr
):
    pid = tl.program_id(0)   # one program per position (b*n)
    v = tl.arange(0, BLOCK_V)   # BLOCK_V must be power of 2

    # 1) compute global max for old and new across all chunks
    max_old = -1e30        # use Python float (no tl.constant)
    max_new = -1e30
    for b in range(NUM_BLOCKS):
        base_idx = pid * V + b * BLOCK_V + v
        mask = (b * BLOCK_V + v) < V
        old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=-1e30).to(tl.float32)
        new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=-1e30).to(tl.float32)
        chunk_max_old = tl.max(old_chunk, axis=0)
        chunk_max_new = tl.max(new_chunk, axis=0)
        max_old = tl.maximum(max_old, chunk_max_old)
        max_new = tl.maximum(max_new, chunk_max_new)

    # 2) compute sum(exp(x - max)) for old and new across chunks
    sum_old = 0.0
    sum_new = 0.0
    for b in range(NUM_BLOCKS):
        base_idx = pid * V + b * BLOCK_V + v
        mask = (b * BLOCK_V + v) < V
        old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)
        new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)

        # multiply by mask to zero out contributions of padded entries
        sum_old = sum_old + tl.sum(tl.exp(old_chunk - max_old) * mask.to(tl.float32), axis=0)
        sum_new = sum_new + tl.sum(tl.exp(new_chunk - max_new) * mask.to(tl.float32), axis=0)

    eps = 1e-12
    # 3) compute KL by iterating chunks again and accumulating
    kl_acc = 0.0
    for b in range(NUM_BLOCKS):
        base_idx = pid * V + b * BLOCK_V + v
        mask = (b * BLOCK_V + v) < V
        old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)
        new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)

        old_exp = tl.exp(old_chunk - max_old) * mask.to(tl.float32)
        new_exp = tl.exp(new_chunk - max_new) * mask.to(tl.float32)

        old_prob = old_exp / (sum_old + eps)
        new_prob = new_exp / (sum_new + eps)

        log_old = (old_chunk - max_old) - tl.log(sum_old + eps)
        log_new = (new_chunk - max_new) - tl.log(sum_new + eps)

        kl_chunk = old_prob * (log_old - log_new)
        kl_acc = kl_acc + tl.sum(kl_chunk, axis=0)

    # apply attention mask scalar
    attn = tl.load(attention_mask_ptr + pid, mask=tl.constexpr(True), other=0.0).to(tl.float32)
    out = kl_acc * attn
    tl.store(output_ptr + pid, out)


# --------------------------------
# policy_loss kernel (chunked over V) - fixed (no tl.constant)
# --------------------------------
@triton.jit
def policy_loss_kernel_chunked(
    old_logits_ptr, new_logits_ptr, advantages_ptr, target_ids_ptr, attention_mask_ptr, loss_out_ptr,
    V: tl.constexpr, BLOCK_V: tl.constexpr, NUM_BLOCKS: tl.constexpr, CLIP_EPS: tl.constexpr
):
    pid = tl.program_id(0)   # position index
    v = tl.arange(0, BLOCK_V)

    # 1) global max across blocks
    max_old = -1e30
    max_new = -1e30
    for b in range(NUM_BLOCKS):
        base_idx = pid * V + b * BLOCK_V + v
        mask = (b * BLOCK_V + v) < V
        old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=-1e30).to(tl.float32)
        new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=-1e30).to(tl.float32)
        max_old = tl.maximum(max_old, tl.max(old_chunk, axis=0))
        max_new = tl.maximum(max_new, tl.max(new_chunk, axis=0))

    # 2) compute sum exp across blocks
    sum_old = 0.0
    sum_new = 0.0
    for b in range(NUM_BLOCKS):
        base_idx = pid * V + b * BLOCK_V + v
        mask = (b * BLOCK_V + v) < V
        old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)
        new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)
        sum_old = sum_old + tl.sum(tl.exp(old_chunk - max_old) * mask.to(tl.float32), axis=0)
        sum_new = sum_new + tl.sum(tl.exp(new_chunk - max_new) * mask.to(tl.float32), axis=0)

    eps = 1e-12
    # 3) gather prob of selected action: only the block containing the target contributes
    target = tl.load(target_ids_ptr + pid, mask=tl.constexpr(True), other=0).to(tl.int32)
    target_block = target // BLOCK_V
    local_idx = target - target_block * BLOCK_V

    base_idx = pid * V + target_block * BLOCK_V + v
    mask = (target_block * BLOCK_V + v) < V
    old_chunk = tl.load(old_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)
    new_chunk = tl.load(new_logits_ptr + base_idx, mask=mask, other=0.0).to(tl.float32)

    old_exp = tl.exp(old_chunk - max_old) * mask.to(tl.float32)
    new_exp = tl.exp(new_chunk - max_new) * mask.to(tl.float32)

    sel_mask = (v == local_idx)
    old_p_a = tl.sum(old_exp * sel_mask.to(tl.float32), axis=0) / (sum_old + eps)
    new_p_a = tl.sum(new_exp * sel_mask.to(tl.float32), axis=0) / (sum_new + eps)

    ratio = new_p_a / (old_p_a + eps)
    ratio = tl.maximum(0.1, tl.minimum(ratio, 10.0))

    adv = tl.load(advantages_ptr + pid, mask=tl.constexpr(True), other=0.0).to(tl.float32)
    attn = tl.load(attention_mask_ptr + pid, mask=tl.constexpr(True), other=0.0).to(tl.float32)

    surr1 = ratio * adv
    clipped_ratio = tl.minimum(tl.maximum(ratio, 1.0 - CLIP_EPS), 1.0 + CLIP_EPS)
    surr2 = clipped_ratio * adv
    loss = -tl.minimum(surr1, surr2) * attn
    tl.store(loss_out_ptr + pid, loss)

def compute_kl_divergence_triton(old_logits, new_logits, attention_mask, BLOCK_V=256):
    # old_logits/new_logits: (B, N, V)
    assert old_logits.device.type == "cuda"
    B, N, V = old_logits.shape
    # BLOCK_V must be power of 2
    assert (BLOCK_V & (BLOCK_V - 1)) == 0, "BLOCK_V must be power of two"
    NUM_BLOCKS = (V + BLOCK_V - 1) // BLOCK_V

    old_flat = old_logits.reshape(-1, V).contiguous()
    new_flat = new_logits.reshape(-1, V).contiguous()
    attn_flat = attention_mask.reshape(-1).contiguous()

    out = torch.zeros((B * N,), dtype=torch.float32, device=old_logits.device)
    grid = (B * N,)

    compute_kl_kernel_chunked[grid](
        old_flat, new_flat, attn_flat, out,
        V, BLOCK_V, NUM_BLOCKS,
        num_warps=4
    )

    denom = attention_mask.to(torch.float32).sum()
    return out.sum() / (denom + 1e-12)


def compute_policy_loss_triton(old_logits, new_logits, advantages, target_ids, attention_mask,
                                       BLOCK_V=256, clip_eps=0.2):
    assert old_logits.device.type == "cuda"
    B, N, V = old_logits.shape
    assert (BLOCK_V & (BLOCK_V - 1)) == 0, "BLOCK_V must be power of two"
    NUM_BLOCKS = (V + BLOCK_V - 1) // BLOCK_V

    old_flat = old_logits.reshape(-1, V).contiguous()
    new_flat = new_logits.reshape(-1, V).contiguous()
    adv_flat = advantages.reshape(-1).contiguous()
    tgt_flat = target_ids.reshape(-1).contiguous()
    attn_flat = attention_mask.reshape(-1).contiguous()

    out = torch.zeros((B * N,), dtype=torch.float32, device=old_logits.device)
    grid = (B * N,)

    policy_loss_kernel_chunked[grid](
        old_flat, new_flat, adv_flat, tgt_flat, attn_flat, out,
        V, BLOCK_V, NUM_BLOCKS, clip_eps,
        num_warps=4
    )

    denom = attention_mask.to(torch.float32).sum()
    return out.sum() / (denom + 1e-12)



import torch.nn.functional as F
# PyTorch baseline (stable)
def compute_kl_divergence_torch(old_logits, new_logits, attention_mask):
    old = old_logits.float()
    new = new_logits.float()
    log_p = F.log_softmax(old, dim=-1)
    log_q = F.log_softmax(new, dim=-1)
    p = torch.exp(log_p)
    kl_per_token = torch.sum(p * (log_p - log_q), dim=-1)
    kl_masked = kl_per_token * attention_mask
    return kl_masked.sum() / (attention_mask.sum() + 1e-12)

# PyTorch PPO surrogate (compute probs fully)
def compute_policy_loss_torch(old_logits, new_logits, advantages, target_ids, attention_mask, clip_eps=0.2):
    """
    PyTorch baseline implementation of PPO policy loss.
    """
    # 确保 target_ids 为 int64 类型
    target_ids = target_ids.long()
    
    # Softmax
    old_p = F.softmax(old_logits, dim=-1)
    new_p = F.softmax(new_logits, dim=-1)
    
    # Gather probabilities of the selected actions
    idx = target_ids.unsqueeze(-1)
    old_pa = torch.gather(old_p, dim=-1, index=idx).squeeze(-1)
    new_pa = torch.gather(new_p, dim=-1, index=idx).squeeze(-1)
    
    # Ratio
    ratio = new_pa / (old_pa + 1e-10)
    
    # PPO clipping
    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2) * attention_mask
    
    return loss.sum() / attention_mask.sum()

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    B, N, V = 2, 4, 8

    # bf16 logits (matches your earlier usage)
    old_logits = torch.randn((B, N, V), dtype=torch.bfloat16, device=device)
    new_logits = torch.randn((B, N, V), dtype=torch.bfloat16, device=device)

    # attention_mask: 1/0
    attention_mask = torch.tensor([[1,1,0,1],[1,0,1,1]], dtype=torch.float32, device=device)

    kl_t = compute_kl_divergence_triton(old_logits, new_logits, attention_mask)
    print("✅ KL Divergence (Triton):", float(kl_t))
    kl_pt = compute_kl_divergence_torch(old_logits, new_logits, attention_mask)
    print("✅ KL Divergence (Torch):", float(kl_pt))
    print("Difference:", abs(float(kl_t) - float(kl_pt)))

    # PPO loss test
    advantages = torch.randn((B, N), dtype=torch.float32, device=device)
    target_ids = torch.randint(0, V, (B, N), dtype=torch.int32, device=device)

    loss_t = compute_policy_loss_triton(old_logits, new_logits, advantages, target_ids, attention_mask)
    print("✅ PPO Loss (Triton):", float(loss_t))

    loss_pt = compute_policy_loss_torch(old_logits, new_logits, advantages, target_ids, attention_mask)
    print("✅ PPO Loss (Torch):", float(loss_pt))
    print("Difference:", abs(float(loss_t) - float(loss_pt)))