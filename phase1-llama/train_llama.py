"""
train_llama.py — фаза 1: GPT-2 -> Llama, компоненты за флагами конфигурации.

A/B-прогоны без правок кода (флаги накопительно):
    NORM_TYPE=layernorm POS_TYPE=learned                       # baseline (GPT-2)
    NORM_TYPE=rmsnorm                                          # +RMSNorm
    NORM_TYPE=rmsnorm POS_TYPE=rope                            # +RoPE
    NORM_TYPE=rmsnorm POS_TYPE=rope MLP_TYPE=swiglu            # +SwiGLU
    NORM_TYPE=rmsnorm POS_TYPE=rope MLP_TYPE=swiglu N_KV_HEAD=4  # +GQA (полная Llama)
Бюджет прогона:  MAX_STEPS=1000 WARMUP_STEPS=100 (по умолчанию боевые 19073/715)

Тесты (без GPU и данных):
    python train_llama.py --test-rope       # инвариант относительных позиций
    python train_llama.py --test-configs    # forward + параметры всех 5 конфигураций
"""
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------------------------------------------------------
# логирование метрик: одна строка на измерение -> "шаг метрика значение"

log_dir = os.path.join(os.path.expanduser("~"), "llm-track-logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt")


def log(step, **metrics):
    with open(log_file, "a") as f:
        for k, v in metrics.items():
            f.write(f"{step} {k} {v:.6f}\n")


# -----------------------------------------------------------------------------
# HellaSwag: датасет грузится лениво, чтобы импорт модуля не лез в сеть

_hswag = None


def hellaswag_examples():
    global _hswag
    if _hswag is None:
        _hswag = load_dataset("Rowan/hellaswag", split="validation")
    return _hswag


def render_example(example, enc):
    """пример -> tokens (4,T), mask (4,T) с единицами на токенах окончания, label"""
    ctx_tokens = enc.encode(example["ctx"])
    tok_rows, mask_rows = [], []
    for end in example["endings"]:
        end_tokens = enc.encode(" " + end)      # пробел: окончание продолжает фразу
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(t)] = torch.tensor(t)
        mask[i, :len(m)] = torch.tensor(m)
    return tokens, mask, int(example["label"])


def get_most_likely_row(tokens, mask, logits):
    """индекс варианта с минимальным средним лоссом на токенах окончания"""
    shift_logits = logits[:, :-1, :]
    shift_tokens = tokens[:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    losses = losses.view(tokens.size(0), -1)
    shift_mask = mask[:, 1:]
    masked_losses = losses * shift_mask
    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


@torch.no_grad()
def evaluate_hellaswag(model, enc, device, device_type, ddp_rank, ddp_world_size, ddp):
    model.eval()
    num_correct = num_total = 0
    for i, example in enumerate(hellaswag_examples()):
        if i % ddp_world_size != ddp_rank:            # каждый ранк — свою долю примеров
            continue
        tokens, mask, label = render_example(example, enc)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)
        num_correct += int(get_most_likely_row(tokens, mask, logits) == label)
        num_total += 1
    if ddp:
        t = torch.tensor([num_correct, num_total], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        num_correct, num_total = t[0].item(), t[1].item()
    model.train()
    return num_correct / num_total


# -----------------------------------------------------------------------------
# RoPE

def precompute_rope_cache(head_dim, max_seq_len, theta=10000.0, device=None):
    """cos, sin формы (max_seq_len, head_dim//2): угол поворота для каждой пары координат"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: (B, nh, T, hs) -> повёрнутый той же формы. Пары половинами: (i, i+hs/2)"""
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


def test_rope():
    """инвариант: q·k зависит только от разности позиций"""
    head_dim = 64
    cos, sin = precompute_rope_cache(head_dim, 200)
    q, k = torch.randn(1, 1, 1, head_dim), torch.randn(1, 1, 1, head_dim)

    def rot(vec, pos):
        c = cos[pos:pos + 1].view(1, 1, 1, -1)
        s = sin[pos:pos + 1].view(1, 1, 1, -1)
        v1, v2 = vec.chunk(2, dim=-1)
        return torch.cat([v1 * c - v2 * s, v1 * s + v2 * c], dim=-1)

    d_near = (rot(q, 5) * rot(k, 3)).sum().item()        # дистанция 2
    d_far = (rot(q, 105) * rot(k, 103)).sum().item()     # та же дистанция 2, другие позиции
    d_other = (rot(q, 105) * rot(k, 100)).sum().item()   # дистанция 5
    print(f"q·k  (5,3)={d_near:.6f}  (105,103)={d_far:.6f}  (105,100)={d_other:.6f}")
    print("relative-position invariance:", abs(d_near - d_far) < 1e-3)
    print("different distance differs:  ", abs(d_near - d_other) > 1e-3)


# -----------------------------------------------------------------------------
# данные

def load_tokens(filename):
    npt = np.load(filename)
    return torch.tensor(npt.astype(np.int64), dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = os.environ.get("FINEWEB_DIR", "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"
        if self.process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# модель

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_type: str = 'layernorm'    # 'layernorm' | 'rmsnorm'
    pos_type: str = 'learned'       # 'learned'   | 'rope'
    mlp_type: str = 'gelu'          # 'gelu'      | 'swiglu'
    n_kv_head: int = 12             # 12 = MHA, 4 = GQA, 1 = MQA


class RMSNorm(nn.Module):
    """x / rms(x) * gamma — без центрирования и без beta"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma, нейтральный старт = единицы

    def forward(self, x):
        # в fp32: деление на маленький корень чувствительно к точности
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def make_norm(config):
    if config.norm_type == 'rmsnorm':
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd)


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


class SwiGLU(nn.Module):
    """w_down( SiLU(w_gate(x)) * w_up(x) ) — гейт; hidden = 8/3*n_embd, чтобы параметров было как у GELU-MLP"""
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 256 * ((hidden + 255) // 256)        # 2048 при n_embd=768
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1            # выходная проекция в residual — как c_proj

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def make_mlp(config):
    return SwiGLU(config) if config.mlp_type == 'swiglu' else MLP(config)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head     # сколько Q-голов делят одну K/V
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.pos_type = config.pos_type
        kv_dim = config.n_kv_head * self.head_dim
        # q полный (n_embd), k и v урезаны до n_kv_head голов: 768 -> 768 + 2*kv_dim
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * kv_dim)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # выходная проекция
        self.c_proj.NANOGPT_SCALE_INIT = 1
        if config.pos_type == 'rope':
            cos, sin = precompute_rope_cache(self.head_dim, config.block_size)
            # буферы, не параметры: не учатся, но едут на device вместе с моделью
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        # маска не нужна при flash attention (is_causal=True), оставлена для имён HF-весов
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        kv_dim = self.n_kv_head * self.head_dim
        q, k, v = qkv.split([self.n_embd, kv_dim, kv_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # [B, nh,  T, hs]
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # [B, nkv, T, hs]
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.pos_type == 'rope':
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)   # v не трогаем: он несёт содержание
        if self.n_rep > 1:
            # GQA: размножаем K/V на группы Q-голов; в KV-кэше при инференсе хранятся только n_kv_head
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = make_mlp(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=make_norm(config),
        ))
        if config.pos_type == 'learned':
            # при RoPE таблица позиций не нужна вовсе (-0.8M параметров)
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight    # weight tying
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
        """загрузка весов GPT-2 (только для дефолтной конфигурации: layernorm/learned/gelu/MHA)"""
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['n_kv_head'] = config_args['n_head']   # MHA, как у GPT-2
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # веса Conv1D из TF лежат транспонированными
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
            {'params': decay_params, 'weight_decay': weight_decay},
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
        x = self.transformer.wte(idx)
        if self.config.pos_type == 'learned':
            pos = torch.arange(T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        # при pos_type='rope' позиция вносится поворотом q,k внутри attention
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# LR-шедулер (бюджет прогона задаётся окружением: A/B гоняем короткими прогонами)

max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = int(os.environ.get('MAX_STEPS', 19073))
warmup_steps = int(os.environ.get('WARMUP_STEPS', 715))


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------

def generate_samples(raw_model, enc, device, device_type, ddp_rank,
                     num_return_sequences=4, max_length=32):
    """сэмплы через raw_model: переменная длина + compile = рекомпиляции"""
    tokens = enc.encode("Hello, I'm a language model,")
    xgen = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    xgen = xgen.repeat(num_return_sequences, 1).to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)          # свой генератор: не трогает обучение
    while xgen.size(1) < max_length:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = raw_model(xgen)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        xgen = torch.cat((xgen, torch.gather(topk_indices, -1, ix)), dim=1)
    for i in range(num_return_sequences):
        print(f"rank {ddp_rank} sample {i}: {enc.decode(xgen[i].tolist())}")


def evaluate_val(model, val_loader, device, device_type, ddp, val_loss_steps=20):
    model.eval()
    val_loader.reset()                              # одни и те же токены каждый раз
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_loss_accum += (loss / val_loss_steps).detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return val_loss_accum.item()


def main():
    # --- DDP ---
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'   # autocast хочет 'cuda'

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding('gpt2')

    # --- батч ---
    total_batch_size = 524288
    B, T = 64, 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, \
        f"total_batch_size {total_batch_size} not divisible by B*T*world_size = {B * T * ddp_world_size}"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"=> steps: {max_steps} (warmup {warmup_steps})")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                                  num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                                num_processes=ddp_world_size, split='val')

    torch.set_float32_matmul_precision('high')      # TF32 до создания модели

    # --- конфигурация архитектуры из окружения (A/B без правок кода) ---
    config = GPTConfig(
        norm_type=os.environ.get('NORM_TYPE', 'layernorm'),
        pos_type=os.environ.get('POS_TYPE', 'learned'),
        mlp_type=os.environ.get('MLP_TYPE', 'gelu'),
        n_kv_head=int(os.environ.get('N_KV_HEAD', 12)),
    )
    if master_process:
        print(config)

    model = GPT(config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{n_params / 1e6:.1f}M parameters")

    model.train()
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4,
                                               device=device)

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # --- val loss + HellaSwag ---
        if step % 250 == 0 or last_step:
            val_loss = evaluate_val(model, val_loader, device, device_type, ddp)
            hella_acc = evaluate_hellaswag(raw_model, enc, device, device_type,
                                           ddp_rank, ddp_world_size, ddp)
            if master_process:
                log(step, val_loss=val_loss, hella=hella_acc)
                print(f"validation loss: {val_loss:.4f} | hellaswag: {hella_acc:.4f}")

        # --- сэмплы ---
        if (step > 0 and step % 250 == 0) or last_step:
            model.eval()
            generate_samples(raw_model, enc, device, device_type, ddp_rank)
            model.train()

        # --- шаг обучения ---
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                # синк градиентов только на последнем микрошаге
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
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
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tok_s = (B * T * grad_accum_steps * ddp_world_size) / (t1 - t0)
        if master_process:
            log(step, train_loss=loss_accum.item(), lr=lr, norm=norm.item(),
                dt=dt, tok_s=tok_s)
            print(f"step {step:5d} | loss {loss_accum.item():.4f} | norm {norm:.4f} | "
                  f"lr {lr:.4e} | {dt:7.2f} ms | {tok_s:8.0f} tok/sec")

        # --- чекпоинт: один файл, перезаписывается (в $HOME: /tmp эфемерный) ---
        if master_process and (last_step or (step > 0 and step % 5000 == 0)):
            ckpt = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
            }
            torch.save(ckpt, os.path.join(os.path.expanduser('~'), 'model_latest.pt'))

    if ddp:
        destroy_process_group()


def test_configs():
    """forward + число параметров для всех конфигураций фазы 1 (ловит опечатки и размерности)"""
    cfgs = [
        dict(),
        dict(norm_type='rmsnorm'),
        dict(norm_type='rmsnorm', pos_type='rope'),
        dict(norm_type='rmsnorm', pos_type='rope', mlp_type='swiglu'),
        dict(norm_type='rmsnorm', pos_type='rope', mlp_type='swiglu', n_kv_head=4),
    ]
    for c in cfgs:
        model = GPT(GPTConfig(**c))
        n = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, 1000, (2, 16))
        logits, loss = model(x, x)
        print(f"{n / 1e6:6.1f}M  logits {tuple(logits.shape)}  loss {loss.item():.2f}  {c or 'baseline'}")


if __name__ == "__main__":
    if '--test-rope' in sys.argv:
        test_rope()
    elif '--test-configs' in sys.argv:
        test_configs()
    else:
        main()