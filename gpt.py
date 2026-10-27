import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import os

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
block_size = 64
n_embd = 128     
n_head = 4       
n_kv_head = 2    
n_layer = 4      
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# ==========================================
# 2. BPE TOKENIZER & CHAT DATA (FIXED)
# ==========================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

# Define our chat pairs
chat_pairs = [
    ("<|user|> What is the capital of France?\n<|assistant|>", " The capital of France is Paris.\n<|endoftext|>"),
    ("<|user|> Who wrote Romeo and Juliet?\n<|assistant|>", " Romeo and Juliet was written by William Shakespeare.\n<|endoftext|>"),
    ("<|user|> What is the largest ocean?\n<|assistant|>", " The largest ocean on Earth is the Pacific Ocean.\n<|endoftext|>"),
]

all_tokens = []
all_masks = []

for user_text, assistant_text in chat_pairs:
    # Encode user prompt (Loss mask = 0, we don't train on the user's question)
    user_tokens = enc.encode(user_text)
    
    # Encode assistant answer (Loss mask = 1, we DO train on the answer)
    # We MUST pass allowed_special to allow <|endoftext|>
    assistant_tokens = enc.encode(assistant_text, allowed_special={'<|endoftext|>'})
    
    # Combine them
    all_tokens.extend(user_tokens + assistant_tokens)
    all_masks.extend([0.0] * len(user_tokens) + [1.0] * len(assistant_tokens))

# Convert to tensors
data = torch.tensor(all_tokens, dtype=torch.long)
loss_mask = torch.tensor(all_masks, dtype=torch.float32)

print(f"Total tokens: {len(data)}")
print(f"Tokens to train on (Assistant only): {int(loss_mask.sum())}")

# ==========================================
# 3. RMSNORM (No change)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# ==========================================
# 4. ROPE (No change)
# ==========================================
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0) 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 5. SWIGLU (No change)
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ==========================================
# 6. MODERN ATTENTION (No change)
# ==========================================
class ModernAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head 
        self.head_dim = n_embd // n_head
        
        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis, mask=None, start_pos=0, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, freqs_cis[start_pos:start_pos+T])
        
        if use_cache:
            if self.cache_k is not None:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
                k, v = self.cache_k, self.cache_v
            else:
                self.cache_k, self.cache_v = k, v
            
        if self.n_head != self.n_kv_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            
        wei = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        if mask is not None and wei.shape[-2] > 1:
            mask_slice = mask[:, :, :wei.shape[-2], :wei.shape[-1]]
            wei = wei.masked_fill(mask_slice == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

# ==========================================
# 7. MODERN BLOCK (No change)
# ==========================================
class ModernBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.attention = ModernAttention(n_embd, n_head, n_kv_head)
        self.feed_forward = SwiGLU(n_embd, hidden_dim=4 * n_embd)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward(self, x, freqs_cis, mask, start_pos=0, use_cache=False):
        x = x + self.attention(self.norm1(x), freqs_cis, mask, start_pos, use_cache)
        x = x + self.feed_forward(self.norm2(x))
        return x

# ==========================================
# 8. MODERN GPT MODEL (Updated for Masking)
# ==========================================
class ModernGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([ModernBlock(n_embd, n_head, n_kv_head) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.register_buffer('freqs_cis', precompute_freqs_cis(n_embd // n_head, block_size * 2))

    def forward(self, idx, targets=None, loss_mask=None, start_pos=0, use_cache=False):
        B, T = idx.shape
        x = self.embedding(idx)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, self.freqs_cis, mask, start_pos, use_cache)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            # --- CRITICAL CHANGE: APPLY LOSS MASK ---
            # Calculate loss without reduction (so we get a loss for every token)
            loss_unreduced = F.cross_entropy(logits, targets, reduction='none')
            # Apply the mask (zeroes out the user's question)
            loss_masked = loss_unreduced * loss_mask.view(B*T)
            # Average only over the assistant's tokens
            loss = loss_masked.sum() / loss_mask.sum()
            
        return logits, loss

    def clear_cache(self):
        for block in self.blocks:
            block.attention.cache_k = None
            block.attention.cache_v = None

# ==========================================
# 9. TRAINING LOOP (SFT)
# ==========================================
model = ModernGPT(vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training started (SFT)...")
for steps in range(2000):
    ix = torch.randint(len(data) - block_size, (1,))
    x = data[ix:ix+block_size].unsqueeze(0).to(device)
    y = data[ix+1:ix+block_size+1].unsqueeze(0).to(device)
    
    # Get the loss mask for this specific chunk
    mask_chunk = loss_mask[ix:ix+block_size].unsqueeze(0).to(device)
    
    logits, loss = model(x, y, loss_mask=mask_chunk)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print(f"Step {steps}, Loss: {loss.item():.4f}")

# ==========================================
# 10. GENERATION
# ==========================================
def generate(prompt, max_new_tokens=50):
    model.eval()
    model.clear_cache()
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, _ = model(idx, start_pos=0, use_cache=True)
        logits = logits[:, -1, :]
        
    for i in range(max_new_tokens):
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        with torch.no_grad():
            logits, _ = model(idx_next, start_pos=idx.shape[1] - 1, use_cache=True)
            logits = logits[:, -1, :]
            
    return enc.decode(idx[0].tolist())

# ==========================================
# 11. TESTING SFT
# ==========================================
print("\n--- Model Output ---")
test_prompt = "<|user|> What is the capital of France?\n<|assistant|>"
print(generate(test_prompt, max_new_tokens=20))