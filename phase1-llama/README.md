# Phase 1 — GPT-2 → Llama

**Weeks 4–6.** Branch from my Phase 0 code, upgrade one component at a time (~2–3 days each), verifying the loss curve after every swap:

1. LayerNorm → RMSNorm
2. Learned positional embeddings → RoPE
3. GELU MLP → SwiGLU
4. MHA → grouped-query attention

Reading: Llama 3 technical report.

## Done when
- [+] All four components swapped, each verified with an A/B loss curve
- [+] Llama-ified model matches or beats my GPT-2 loss curve on the same data/budget
- [+] One notes/ entry per component that took >30 min to understand

## Phase 1 — GPT-2 → Llama: вклад каждого компонента

Условия: 1000 шагов × 524288 токенов (0.5B), FineWeb-Edu, 2×H100, seed 1337, 
warmup 100, cosine → 6e-5. Флаги накопительные. Шум между прогонами одной конфигурации: ±0.01.

| # | конфигурация | параметры | val loss | Δ loss | tok/s | вывод |
|---|---|---|---|---|---|---|
| 0 | GPT-2 (LayerNorm + learned pos + GELU + MHA) | 124.5M | 4.308 | — | 885k | baseline |
| 1 | + RMSNorm | 124.5M | 4.299 | −0.01 | 894k | качество то же, +1% скорость |
| 2 | + RoPE | 123.7M | 4.085 | **−0.21** | 830k | главный выигрыш; −0.8M params, −7% скорость |
| 3 | + SwiGLU | 123.6M | 4.000 | **−0.08** | 832k | при равных параметрах, скорость та же |
| 4 | + GQA (4 kv-heads) = **Llama** | **114.2M** | **4.013** | +0.01 | **869k** | −9.4M params бесплатно; KV-кэш ×3 меньше |

**Итог:** Llama-архитектура — val loss **4.01 vs 4.31** (−0.30) при **−8% параметров** и **−2% времени шага**.

Выигрыш по качеству дали два компонента — RoPE (~70%) и SwiGLU (~30%). 
RMSNorm и GQA — про эффективность, не про качество.