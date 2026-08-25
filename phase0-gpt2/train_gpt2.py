import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import os 
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group 
from torch.nn.parallel import DistributedDataParallel as DDP 


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B, self.T = B, T
        self.process_rank = process_rank 
        self.num_processes = num_processes 
        with open('input.txt') as f:
            text = f.read()
        self.tokens = torch.tensor(tiktoken.get_encoding('gpt2').encode(text))
        if self.process_rank == 0:
            print(f"loaded {len(self.tokens)} tokens")
        self.current_position = B * T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = B * T * self.process_rank
        return x, y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':            dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':     dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':      dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':         dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
      param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
      decay_params = [p for p in param_dict.values() if p.dim() >= 2]
      nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
      optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},      # было 'weigth_decay' — AdamW молча брал дефолт 0.01 для ОБЕИХ групп
          {'params': nodecay_params, 'weight_decay': 0.0},
      ]
      num_decay = sum(p.numel() for p in decay_params)
      num_nodecay = sum(p.numel() for p in nodecay_params)
      print(f"decayed: {len(decay_params)} tensors, {num_decay:,} params")
      print(f"non-decayed: {len(nodecay_params)} tensors, {num_nodecay:,} params")
      import inspect
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
      use_fused = fused_available and 'cuda' in device
      optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                    betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
      return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

max_lr = 6e-4 
min_lr = max_lr * 0.1 
warmup_steps = 10 
max_steps = 50


def get_lr(it):
  if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps
  if it > max_steps:
    return min_lr
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)


ddp = int(os.environ.get('RANK', -1)) != - 1
if ddp: 
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) 
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1  # было ddp_word_size — NameError при локальном запуске 
    master_process = True 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}") 

device_type = 'cuda' if device.startswith('cuda') else 'cpu'  # autocast хочет 'cuda', а device бывает 'cuda:0'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B, T = 32, 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, \
    f"total_batch_size {total_batch_size} not divisible by B*T*world_size = {B*T*ddp_world_size}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)  # глобальный rank, не local (важно для мультинод)

torch.set_float32_matmul_precision('high')  # TF32 — до создания/компиляции модели

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model 

if master_process:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params/1e6:.1f}M parameters")
    
model.train()

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(50):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    loss = loss /grad_accum_steps
    loss_accum += loss.detach()
    if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # было required_... — DDP атрибут не видел и синкал каждый микрошаг
    loss.backward()
  if ddp: 
      dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  if device_type == 'cuda':
      torch.cuda.synchronize()
  t1  = time.time()
  dt = (t1 - t0) * 1000
  tok_s = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
  if master_process:
    print(f"step {step:2d} | loss {loss_accum.item():.4f} | norm {norm:.4f} | lr {lr:.4e} | {dt:7.2f} ms | {tok_s:8.0f} tok/sec")

if ddp:
  destroy_process_group()