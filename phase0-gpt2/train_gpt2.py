from dataclasses import dataclass 
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12 
    n_head: int = 12  
    n_embd: int = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':            dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':     dict(n_layaer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':      dict(n_layaer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':         dict(n_layaer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        #print(sorted(set(sd_keys) - set(sd_keys_hf)))   # есть у меня, нет в HF
        #print(sorted(set(sd_keys_hf) - set(sd_keys)))   # есть в HF, нет у меня
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: 
            x = block(x)
        x = self.transformer['ln_f'](x)
        x = self.lm_head(x)
        return x 


if __name__ == "__main__":
    # model = GPT(GPTConfig())
    # n_params = sum(p.numel() for p in model.parameters())
    # print(f"{n_params/1e6:.1f}M parameters")

    # idx = torch.randint(0, 50257, (2, 32))
    # logits = model(idx)
    # print(logits.shape)

    # cfg = GPTConfig()
    # attn = CausalSelfAttention(cfg)
    # x = torch.rand(2, 10, cfg.n_embd)
    # y = attn(x)
    # print(y.shape)

    # x2 = x.clone()
    # x2[:, -1, :] = torch.rand(2, cfg.n_embd)
    # y2 = attn(x2)
    # print("casul ok:", torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6))

    # model = GPT.from_pretrained('gpt2')
    # print("weights loaded ok")

    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    num_return_sequence = 5 
    max_length = 30


    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequence, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
      with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix) 
        x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequence):
      print(enc.decode(x[i].tolist()))

