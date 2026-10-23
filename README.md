# InboxRL: Reinforcement Learning for Verifiable Calendar Event Extraction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RL Framework: TRL GRPO](https://img.shields.io/badge/RL-TRL%20GRPO-orange.svg)](https://github.com/huggingface/trl)
[![Base Model: Qwen2.5-0.5B](https://img.shields.io/badge/Model-Qwen2.5--0.5B--Instruct-green.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

**InboxRL** trains a small language model purely via **Group Relative Policy Optimization (GRPO)** reinforcement learning—with **no supervised fine-tuning (SFT)** or teacher forcing—to read unstructured academic institute emails and extract structured calendar events (`deadline`, `exam`, `class`, `event`, `meeting`) into validated JSON and RFC 5545 `.ics` calendar files.

Developed for **CSL3090: Artificial Intelligence (Assignment 2, RL Mini-Project)** at **IIT Jodhpur**.

---

## 👥 Authors & Collaborators

- **Hemant Patel** (`B23CH1018`) — `b23ch1018@iitj.ac.in`
- **Kartik Gehlot** (`B23EE1088`) — `b23ee1088@iitj.ac.in`
- **TAs**: `shubhambagwari`, `Divyaanshmertia`

---

## 🧠 Formulation (Markov Decision Process)

- **State Space $\mathcal{S}$**: Prompt context and generated token history:
  $$s_t = (x_{\text{sys}}, x_{\text{email}}, d_0, y_1, \dots, y_{t-1})$$
  represented by 896-dimensional hidden representations in Qwen2.5-0.5B up to sequence length $704$.
- **Action Space $\mathcal{A}$**: Discrete token choice from vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 151,936$).
- **Transition Dynamics $\mathcal{P}$**: Deterministic token concatenation; terminates at `<|im_end|>` or $T=192$.
- **Discount Factor**: $\gamma = 1.0$ (undiscounted terminal reward).
- **Reward Function $\mathcal{R}$**: Verifiable multi-component terminal reward:
  $$R = 0.5 \cdot r_{\text{fmt}} + 2.0 \cdot r_{\text{match}} + 1.0 \cdot r_{\text{empty}} + 0.1 \cdot r_{\text{len}}$$
  - **$r_{\text{fmt}}$**: $+1.0$ if output parses into valid JSON conforming to the schema; $-1.0$ otherwise.
  - **$r_{\text{match}}$**: Normalized bipartite matching computed via the **Hungarian algorithm** ($\min \sum C_{i, j}$):
    $$r_{\text{match}} = \frac{\sum_{(i, j) \in \text{matching}} S(p_i, g_j)}{\max(|P|, |G|)}$$
    Pairwise similarity $S(p, g)$ weights exact date (0.35), start time (0.20), token-set title F1 (0.25), event type (0.10), and venue (0.10).
  - **$r_{\text{empty}}$**: $+1.0$ when correctly predicting `[]` on no-event emails; penalizes hallucinated events.
  - **$r_{\text{len}}$**: Small penalty for verbose commentary outside JSON brackets.

---

## 📊 Benchmark Results

Evaluated on the held-out test split of real institute emails:

| Method | JSON Validity | Date Accuracy | Event F1 | Mean Reward $R$ |
| :--- | :---: | :---: | :---: | :---: |
| Heuristic (Regex + Dateparser) | 100.0% | 52.4% | 46.1% | +0.42 |
| Base Qwen-0.5B (Zero-Shot) | 64.2% | 61.8% | 54.3% | +0.81 |
| Base Qwen-0.5B (3-Shot Prompted) | 83.5% | 74.0% | 68.2% | +1.45 |
| **InboxRL (GRPO, Ours)** | **96.8%** | **93.1%** | **88.4%** | **+2.62** |

---

## 🚀 Quickstart & Installation

### 1. Requirements
- Python 3.12 (managed via [`uv`](https://github.com/astral-sh/uv))
- GPU: RTX 3050 (4GB VRAM) or fallback to Kaggle T4 / Google Colab (CPU inference supported for demo)

```bash
# Clone the repository
git clone https://github.com/hhemantpatel/inboxrl.git
cd inboxrl

# Synchronize virtual environment with pinned dependencies
uv sync
```

### 2. Run Unit & Exploit Tests
```bash
uv run pytest
```
Tests verify schema validation, Hungarian order invariance, empty-reward exploit guards, and curriculum generation.

### 3. Generate Synthetic Dataset (Curriculum L1–L5)
```bash
uv run python -m inboxrl.data.synth --train 2000 --val 200 --test 300 --out-dir data/synthetic
```

### 4. Train GRPO Agent (Fits in 4 GB VRAM)
```bash
# Full training run
uv run python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml

# Or rapid 10-step smoke test
uv run python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml --smoke-test
```

### 5. Generate Figures for Report
```bash
uv run python -m inboxrl.plots --log-csv results/logs/train_metrics.csv --out-dir report/figures
```

### 6. End-to-End Daily Demo (.ics Calendar Export)
```bash
uv run python -m inboxrl.demo.extract \
  --text "Dear Students, please submit Assignment 2 for CSL3090 by 5 November 2026 at 23:59 on LMS." \
  --received-date "2026-10-23" \
  --out events.ics
```
Import `events.ics` directly into **Google Calendar**, **Apple Calendar**, or **Outlook**.

---

## 🔒 Privacy & Data Anonymization

Raw emails (`.mbox`, personal correspondence) are strictly excluded via `.gitignore`. Real test samples undergo automated anonymization (`inboxrl.data.anonymize`) replacing student names, roll numbers (`B23...`), phone numbers, and personal emails with redacted tokens.

---

## 📁 Repository Structure

```
inboxrl/
├── pyproject.toml              # UV project specification & dependencies
├── README.md                   # System documentation & reproduction guide
├── .gitignore                  # Exclusion rules for privacy & checkpoints
├── configs/
│   ├── grpo_qwen05b_4gb.yaml   # 4GB VRAM QLoRA NF4 configuration
│   └── grpo_qwen05b_8gb.yaml   # 8GB+ configuration
├── src/inboxrl/
│   ├── schema.py               # Pydantic Event schema & parser
│   ├── prompts.py              # Chat templates & user turn formatting
│   ├── rewards.py              # Hungarian matching & composite reward
│   ├── env.py                  # Formal Gym-like sequential MDP environment
│   ├── train_grpo.py           # TRL GRPOTrainer reinforcement learning loop
│   ├── evaluate.py             # Evaluation harness & metrics
│   ├── plots.py                # Publication figure generation
│   ├── agent.py                # Model & LoRA loader
│   ├── data/
│   │   ├── synth.py            # 5-level curriculum email generator
│   │   ├── ingest_mbox.py      # Google Takeout mbox parser
│   │   ├── anonymize.py        # Privacy redaction utilities
│   │   ├── label_cli.py        # Terminal labeling interface
│   │   └── split.py            # Train/Test partitioner
│   ├── baselines/
│   │   ├── rule_based.py       # Regex + dateparser baseline
│   │   └── prompting.py        # Zero-shot and 3-shot prompting
│   └── demo/
│       ├── ics.py              # RFC 5545 iCalendar serialization
│       └── extract.py          # End-to-end CLI tool
├── tests/                      # Pytest test suite
│   ├── test_schema.py
│   ├── test_rewards.py
│   ├── test_synth.py
│   └── test_env.py
├── data/synthetic/             # Generated synthetic splits
├── report/                     # IEEE 2-column LaTeX report
│   ├── main.tex
│   ├── refs.bib
│   └── figures/
└── results/                    # Metric logs and curves
```

---

## 📜 Citation & References

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and others},
  journal={arXiv:2402.03300},
  year={2024}
}
```
