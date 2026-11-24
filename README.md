# From Zero to Reasoner

**Teaching small language models to think with reinforcement learning**
WiDS 6.0 · Analytics Club, IIT Bombay · Guided project · 6 weeks · 8–10 h/week
Mentor: Shreyansh Tripathi ([@Shrey-tri-coder](https://github.com/Shrey-tri-coder))

In early 2025, DeepSeek-R1 showed that a language model can learn to reason (plan, check its own work, backtrack) through reinforcement learning alone, rewarded only for reaching the right final answer. Open replications such as TinyZero reproduced this "aha moment" on models small enough to train on a single GPU.

In this project you implement that recipe yourself. You go from policy gradients to GRPO, written from scratch, and use it to train a sub-1B open model (Qwen2.5-0.5B-Instruct) on a task of your choice, on free Colab or Kaggle GPUs.

## The question each mentee answers

> Can a language model under 1B parameters learn to solve a reasoning task it initially fails, using only RL against an automatic checker and free-tier compute? Does what it learns transfer to harder problems it never saw?

On Demo Day every mentee shows:

1. A reward/accuracy curve that starts near zero and climbs
2. One annotated reasoning trace where the model catches and fixes its own mistake
3. A held-out scorecard: base 0.5B vs. trained 0.5B vs. a larger base model
4. One reward hack found in their own run, and the fix

## Plan

| Week | Theme | Build | Deliverable |
| --- | --- | --- | --- |
| 1 | RL foundations | REINFORCE on CartPole (PyTorch + Gymnasium) | Learning curves + note on baselines |
| 2 | LMs as policies | Per-token log-probs and base accuracy for Qwen2.5-0.5B on 3 candidate tasks | **Checkpoint 1:** baseline scorecard + chosen task |
| 3 | GRPO from scratch | ~200-line GRPO loop in plain PyTorch on a toy task | Rising reward curve on the toy task |
| 4 | Your task, your reward | Task generator + checker; first run with TRL GRPOTrainer / Unsloth + LoRA | **Checkpoint 2:** first run on own task |
| 5 | Make it work, then break it | Ablations; hunt for reward hacks | Ablation table + one hack and fix |
| 6 | Breakthrough | Held-out and harder-level evaluation; push model to Hugging Face | **Demo Day:** model, 4-page report, 5-min demo |

Materials are released week by week: [Week 1 is here](week1/).

## Task menu

Any task works if it has a problem generator, an automatic checker, a difficulty knob, and a base-model success rate of roughly 5–60% at the easiest level. (If every sample in a GRPO group gets the same reward, the advantage is zero and nothing is learned.)

Countdown arithmetic · Sudoku · multi-digit multiplication · Text-to-SQL on SQLite · regex from examples · chess mate-in-1/2 · knights-and-knaves logic · calendar reasoning · unit-conversion word problems · bring your own (approved at Checkpoint 1)

Every task implements the same interface so results are comparable on the cohort leaderboard:

```python
def generate(level: int, n: int) -> list[dict]:   # problems with answers
    ...
def reward(problem: dict, completion: str) -> float:  # score in [0, 1]
    ...
```

## Setup

- A Google account (Colab) and a Kaggle account. No local GPU needed.
- Python 3.10+, `torch`, `gymnasium` (Weeks 1–3); `transformers`, `trl` or `unsloth`, `peft` (Weeks 4–6)

## Core reading

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html): policy gradients
- [DeepSeekMath](https://huggingface.co/papers/2402.03300): introduces GRPO
- [DeepSeek-R1](https://huggingface.co/papers/2501.12948): the R1-Zero result
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): minimal R1-Zero replication on Countdown
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) and [DAPO](https://arxiv.org/abs/2503.14476): what goes wrong and how to fix it
