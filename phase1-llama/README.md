# Phase 1 — GPT-2 → Llama

**Weeks 4–6.** Branch from my Phase 0 code, upgrade one component at a time (~2–3 days each), verifying the loss curve after every swap:

1. LayerNorm → RMSNorm
2. Learned positional embeddings → RoPE
3. GELU MLP → SwiGLU
4. MHA → grouped-query attention

Reading: Llama 3 technical report.

## Done when
- [ ] All four components swapped, each verified with an A/B loss curve
- [ ] Llama-ified model matches or beats my GPT-2 loss curve on the same data/budget
- [ ] One notes/ entry per component that took >30 min to understand
