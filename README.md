# InboxRL: Pure-RL Calendar Event Extraction from Institute Emails

InboxRL trains a small language model (Qwen2.5-0.5B-Instruct + LoRA) **purely with reinforcement learning (GRPO)** —
no supervised fine-tuning, no teacher forcing — to read an academic institute email and output its calendar events
(`deadline`, `exam`, `class`, `event`, `meeting`) as JSON, which a CLI then exports to an `.ics` calendar file.
Labels are used **only inside a verifiable reward**, never as training targets.

CSL3090 Artificial Intelligence, Assignment 2 (RL mini-project), IIT Jodhpur.

**Team:** Hemant Patel (B23CH1018) · Kartik Gehlot (B23EE1088)

---

## RL formulation (short)

| | |
|---|---|
| **State** $s_t$ | system prompt + email (sender, subject, body, received date $d_0$) + tokens generated so far; ≤ 640 prompt tokens, embedded as 896-d hidden states |
| **Action** $a_t$ | next token from the vocabulary, $\lvert\mathcal V\rvert = 151{,}936$ |
| **Transition** | deterministic append; episode ends at `<|im_end|>` or 192 tokens; $\gamma = 1$ |
| **Reward** | sparse, terminal: $R = 0.5\,r_{fmt} + 2\,r_{match} + r_{empty} + 0.25\,r_{len} \in [-1.75, 2.5]$ |
| **Algorithm** | GRPO (TRL `GRPOTrainer`), group size 4, KL $\beta = 0.04$ to the frozen base model, L1–L2 → L1–L5 curriculum |

- $r_{fmt}$: +1 if the output is a JSON list that validates against the event schema, else −1 (and no match credit).
- $r_{match}$: Hungarian matching between predicted and gold events; pair similarity = 0.35·date + gate·(0.20·time + 0.25·title-token-F1 + 0.10·type + 0.10·venue), where gate = 1 if the date is right and 0.25 otherwise. Normalised by $\max(|P|, |G|)$, so misses and hallucinations both cost reward.
- $r_{empty}$: +1 for `[]` on a no-event email, −1 for hallucinating there, −0.5 for `[]` when events exist.
- $r_{len}$: −1 for any text around the JSON, −0.5 more beyond 800 characters.

Details, equations and the two reward-hacking exploits we found and fixed are in `report/main.tex`.

---

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Any CUDA GPU works; precision is picked automatically
(bf16 on RTX 30xx/40xx, fp16 on older cards such as the T4, fp32 on CPU).

```bash
git clone https://github.com/hhemantpatel/inboxrl.git
cd inboxrl
uv sync                      # add `--extra qlora` only if you want 4-bit loading / 8-bit Adam
uv run pytest                # 40+ tests: schema, reward exploits, generator, env, data tools
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
```

> The `torch` wheel in `uv.lock` targets CUDA 13 (driver ≥ 580). On an older driver install a matching wheel, e.g.
> `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`.

## Runbook (training machine)

All commands are run from the repo root. Every training run writes to `runs/<run-name>/`
(`metrics.csv`, `config.yaml`, `adapter/`, TensorBoard logs); every evaluation writes `results/<name>_<split>.json`
plus raw completions in `results/completions/`.

**1. Data.** The synthetic splits are committed; regenerate them with
```bash
uv run python -m inboxrl.data.synth          # train 2000 / val 200 / test 300 (template-disjoint test)
```

**2. Smoke test (a few minutes on a GPU).** Note the `s/it` it prints.
```bash
uv run python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml --smoke-test
```
The smoke test runs tiny steps (2 completions each). A real step has 16 completions; budget about 4–8× the smoke
`s/it`. Pick `max_steps` in the config so the main run takes roughly 4–6 h (the default is 800). If you change it,
also check `curriculum_stage1_steps` (default 250). Use `configs/grpo_qwen05b_8gb.yaml` on a GPU with ≥ 8 GB.

**3. Baselines** (can run while nothing else uses the GPU):
```bash
uv run python -m inboxrl.baselines.run_baselines              # empty, random, rule, 0.5B zero/3-shot, 1.5B zero-shot
```

**4. Main runs (3 seeds), then evaluate each:**
```bash
for s in 0 1 2; do
  uv run python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml --run-name main_s$s --seed $s $([ $s = 0 ] && echo --promote)
  for split in synth_test real_test; do
    uv run python -m inboxrl.evaluate --extractor agent --model runs/main_s$s/adapter --split $split --name grpo_main_s$s
  done
done
```
`--promote` copies the seed-0 adapter to `checkpoints/final/` (the committed, trained agent). A run that is
interrupted can continue with `--resume` (same `--run-name`).

**5. Ablations** (shorter runs; evaluated on the synthetic test set):
```bash
for a in no_curriculum beta0 format_only G2; do
  uv run python -m inboxrl.train_grpo --config configs/ablation_$a.yaml --run-name ablation_$a --seed 0
  uv run python -m inboxrl.evaluate --extractor agent --model runs/ablation_$a/adapter --split synth_test --name grpo_$a
done
```

**6. Figures and tables for the report** (reads only `runs/main_s*/metrics.csv` and `results/*.json`):
```bash
uv run python -m inboxrl.plots
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
This writes `report/figures/*.pdf`, `report/results_table.tex` and `report/ablation_table.tex`. **Never type a result
into the report by hand.** Then fill in the red `\todo{}` sentences from the generated tables.

**What to copy back from the training machine:** `runs/*/metrics.csv`, `runs/*/config.yaml`, `results/`,
`checkpoints/final/`. Checkpoint folders (`runs/*/checkpoint-*`) are gitignored.

## Real emails (held-out test set)

```bash
# 1. Google Takeout -> Mail -> .mbox ; list private names to mask in data/real/names_to_mask.txt (gitignored)
uv run python -m inboxrl.data.ingest_mbox --mbox ~/Takeout/Mail/Inbox.mbox --out data/real/raw_anonymized.jsonl
# 2. READ the anonymised file yourself before going further.
# 3. Label (split the work between both teammates):
uv run python -m inboxrl.data.label_cli --input data/real/raw_anonymized.jsonl --out data/real/labeled.jsonl
# 4. Hold out 60 emails as the real test set; the rest are mixed into curriculum stage 2 at 10%
uv run python -m inboxrl.data.split --n-test 60
```
Only the anonymised, labelled JSONL files (`data/real/real_test.jsonl`, `data/real/real_train.jsonl`) are committed.
Raw `.mbox`/`.eml` files, `raw_anonymized.jsonl` and the names list are gitignored.

## Using the trained agent (CPU is fine)

```bash
uv run inboxrl extract --text "Quiz 2 of CSL3090 is on 5 Nov at 10 am in LHC-101." --received-date 2026-10-23
uv run inboxrl extract --mbox ~/Takeout/Mail/Inbox.mbox --since 2026-10-01 --out events.ics
```
Import `events.ics` into Google Calendar (Settings → Import). Times are written in Asia/Kolkata, and re-importing
updates existing events instead of duplicating them. `--base-only` runs the untrained base model for comparison.

## Results

All numbers are generated: see `report/results_table.tex`, `report/ablation_table.tex` and `report/figures/`
after step 6.

## Repository layout

```
configs/                 GRPO configs (4 GB, 8 GB) and ablation configs (inherit via `base:`)
src/inboxrl/
  schema.py              Event schema + tolerant JSON parser
  prompts.py             system prompt, token-budgeted prompt builder
  rewards.py             reward components, Hungarian matching, TRL reward functions
  env.py                 EmailExtractionEnv: the MDP with reset()/step()
  train_grpo.py          GRPO training (curriculum, logging, promotion)
  evaluate.py            metrics with bootstrap CIs; extractors: empty/random/rule/zero-shot/few-shot/agent
  plots.py               figures + LaTeX tables from real outputs only
  agent.py               model/adapter loading, batched greedy generation
  cli.py                 `inboxrl {extract,eval,train,synth}`
  data/                  synth.py, ingest_mbox.py, anonymize.py, label_cli.py, split.py
  baselines/             rule_based.py, prompting.py, run_baselines.py
  demo/                  extract.py (email/mbox -> .ics), ics.py
data/synthetic/          generated splits      data/real/   anonymised labelled splits
checkpoints/final/       trained LoRA adapter (after step 4)
results/                 evaluation JSONs       runs/        training logs (metrics.csv, config.yaml)
report/                  IEEE two-column LaTeX report
tests/                   pytest suite
```

## References

GRPO: Shao et al., *DeepSeekMath* (arXiv:2402.03300) · Qwen2.5 (arXiv:2412.15115) · LoRA (Hu et al., 2021) ·
TRL (von Werra et al.) · Hungarian method (Kuhn, 1955). All project code is our own; TRL is used as a library.
