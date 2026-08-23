# Phase 0 — GPT-2 (124M) from scratch

**Weeks 1–3.** Reproduce the [build-nanogpt](https://github.com/karpathy/build-nanogpt) progression with my own hands. Watch a segment → close the video → write the code → diff against Karpathy's commit only when stuck or done.

## Pacing
- **Week 1 — the model:** attention, blocks, GPT class, load HF GPT-2 weights, sampling works
- **Week 2 — training speed:** training loop, mixed precision, torch.compile, flash attention, grad accumulation
- **Week 3 — the run:** DDP, FineWeb-Edu data pipeline, HellaSwag eval, (optional) overnight 8xGPU run

## Done when
- [ ] My code generates coherent text from loaded GPT-2 weights
- [ ] Training step time within ~2x of reference on the same hardware
- [ ] A training run (any size) shows healthy loss curve and beats random (>25%) on HellaSwag

Session 8 — mixed precision baseline (Colab L4, B=16, T=1024, steps 10-50)

fp32:        1319.38 ms |  6209 tok/s | 1.00x
tf32:         944.67 ms |  8672 tok/s | 1.40x
tf32+bf16:    ~745    ms | ~11000 tok/s | 1.77x

Наблюдения:
- TF32 дал 1.40x бесплатно (одна строка кода, точность на глаз не пострадала — loss ~6.3 во всех режимах).
- bf16 поверх TF32 добавил ещё 1.27x, суммарно 1.77x. Меньше, чем у Карпаты на A100 (~3x+),
  ожидаемо: L4 слабее по tensor-core throughput и уже частично ограничен пропускной памятью.
- Скорости стабильны по шагам (±0.5%) — замер честный (synchronize работает).
