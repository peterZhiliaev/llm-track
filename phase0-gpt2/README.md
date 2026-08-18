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
