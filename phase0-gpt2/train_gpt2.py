from dataclasses import dataclass 
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12 
    n_head: int = 12  
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3 )   # один Linear: n_embd -> 3 * n_embd (q, k, v разом)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   # выходная проекция: n_embd -> n_embd
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # буфер с причинной маской (нижнетреугольная матрица)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        y = attn @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y 


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T))
        for block in self.transformer.h: 
            x = block(x)
        x = self.transformer['ln_f'](x)
        x = self.lm_head(x)
        return x 


if __name__ == "__main__":
    cfg = GPTConfig()
    attn = CausalSelfAttention(cfg)
    x = torch.rand(2, 10, cfg.n_embd)
    y = attn(x)
    print(y.shape)

    x2 = x.clone()
    x2[:, -1, :] = torch.rand(2, cfg.n_embd)
    y2 = attn(x2)
    print("casul ok:", torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6))
    # model = GPT(GPTConfig())
    # n_params = sum(p.numel() for p in model.parameters())
    # print(f"{n_params/1e6:.1f}M parameters")

    # idx = torch.randint(0, 50257, (2, 32))
    # logits = model(idx)
    # print(logits.shape)
