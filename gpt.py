import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import os
from datasets import load_dataset

# ==========================================
# 1. HYPERPARAMETERS & SETUP
# ==========================================
block_size = 128 
n_embd = 128     
n_head = 4       
n_kv_head = 2    
n_layer = 4      
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# ==========================================
# 2. BPE TOKENIZER & DATA LOADING
# ==========================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

data_file = "squad_data.md"

if not os.path.exists(data_file):
    print("Downloading SQuAD dataset and formatting it...")
    dataset = load_dataset("rajpurkar/squad", split="train")
    with open(data_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            if i >= 1000: 
                break
            question = example['question']
            answer = example['answers']['text'][0] if example['answers']['text'] else "No answer provided."
            f.write(f"Q: {question}\nA: {answer}\n\n")
    print("Data saved to squad_data.md")

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Total tokens in dataset: {len(data)}")

# ==========================================
# 3. RMSNORM
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
# 4. ROPE
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
# 5. SWIGLU
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
# 6. MODERN ATTENTION (WITH CONTROLLED CACHE)
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
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, freqs_cis[start_pos:start_pos+T])
        
        # ONLY USE CACHE DURING GENERATION
        if use_cache:
            if self.cache_k is not None:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
                k = self.cache_k
                v = self.cache_v
            else:
                self.cache_k = k
                self.cache_v = v
            
        if self.n_head != self.n_kv_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            
        wei = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        
        if mask is not None:
            if wei.shape[-2] > 1:
                mask_slice = mask[:, :, :wei.shape[-2], :wei.shape[-1]]
                wei = wei.masked_fill(mask_slice == 0, float('-inf'))
                
        wei = F.softmax(wei, dim=-1)
        
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

# ==========================================
# 7. MODERN BLOCK
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
# 8. MODERN GPT MODEL
# ==========================================
class ModernGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
            ModernBlock(n_embd, n_head, n_kv_head) for _ in range(n_layer)
        ])
        
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.register_buffer('freqs_cis', precompute_freqs_cis(n_embd // n_head, block_size * 2))

    def forward(self, idx, targets=None, start_pos=0, use_cache=False):
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
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def clear_cache(self):
        for block in self.blocks:
            block.attention.cache_k = None
            block.attention.cache_v = None

# ==========================================
# 9. TRAINING LOOP (use_cache=False)
# ==========================================
model = ModernGPT(vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training started...")
for steps in range(2000):
    ix = torch.randint(len(data) - block_size, (1,))
    x = data[ix:ix+block_size].unsqueeze(0).to(device)
    y = data[ix+1:ix+block_size+1].unsqueeze(0).to(device)
    
    # use_cache is False by default here. No cache accumulation during training.
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print(f"Step {steps}, Loss: {loss.item():.4f}")

# ==========================================
# 10. GENERATION (use_cache=True)
# ==========================================
def generate(prompt, max_new_tokens=50):
    model.eval()
    model.clear_cache()
    
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Pass use_cache=True during the prompt processing
        logits, _ = model(idx, start_pos=0, use_cache=True)
        logits = logits[:, -1, :]
        
    for i in range(max_new_tokens):
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        with torch.no_grad():
            # Pass use_cache=True during generation
            logits, _ = model(idx_next, start_pos=idx.shape[1] - 1, use_cache=True)
            logits = logits[:, -1, :]
            
    return enc.decode(idx[0].tolist())

# ==========================================
# 11. TESTING
# ==========================================
print("\n--- Model Output ---")
test_prompt = "Q: What is the capital of France?\nA:"
print(f"Prompt: {test_prompt}")
print(f"Output: {generate(test_prompt, max_new_tokens=20)}")