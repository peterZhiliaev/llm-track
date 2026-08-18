# llm-track

Hands-on track: from an empty file to a modern LLM stack, one codebase evolving through every phase.

Started after completing Karpathy's ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) as a viewer — this repo is about building it, then everything that came after GPT-2.

## Phases

| Phase | Weeks | Goal | Status |
|-------|-------|------|--------|
| [0 — GPT-2 from scratch](phase0-gpt2/) | 1–3 | Reproduce GPT-2 (124M) with my own hands | 🔨 in progress |
| [1 — GPT-2 → Llama](phase1-llama/) | 4–6 | Upgrade to modern architecture (RMSNorm, RoPE, SwiGLU, GQA) | ⏳ |
| [2 — Post-training](phase2-posttraining/) | 7–10 | Tokenizer, SFT, RL — the nanochat pipeline | ⏳ |
| [3 — Inference](phase3-inference/) | 11–12 | KV cache, quantization, speculative decoding | ⏳ |
| [4 — MoE + graduation](phase4-moe/) | 13–14 | Mixture-of-Experts layer; reproduce a piece of a recent paper solo | ⏳ |

## Working rules

- **2 hrs/day guaranteed = building.** Code only, no video. Optional 2 hrs = reading (papers, newsletters).
- **Type every line myself.** Reference commits are for diffing when stuck, not for copying.
- **Stuck protocol:** 30 min solo debugging → diff against reference → ask an LLM to explain the diff. In that order.
- **Stop mid-task**, never at a clean break. Leave a "next:" note in the log.
- **Commit daily**, even ugly WIP.
- **Weekly review** (20 min): building or just reading? phase deadline realistic? one fix for next week?

## Log

Daily 3-line entries in [log.md](log.md). Weekly reviews at the bottom of each week.

## Reading list (optional hours)

- Phase 0: GPT-2 paper, GPT-3 paper
- Phase 1: Llama 3 technical report
- Phase 2: Karpathy "Deep Dive into LLMs", DeepSeek-R1 paper, rlhfbook.com (selective)
- Phase 3: vLLM paper, modded-nanogpt speedrun log
- Phase 4: DeepSeek-V3 report + one self-chosen recent paper
- Ongoing: Interconnects (N. Lambert), Ahead of AI (S. Raschka)
