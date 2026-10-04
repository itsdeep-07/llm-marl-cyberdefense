# PROJECT CONTEXT: LLM-Augmented Multi-Agent RL for Automated Cyber Incident Response

> Drop this file in the repo root. Copy it to `AGENTS.md` or `GEMINI.md` (or paste it at the start of a chat) so your AI coding tool reads it first.

---

## 1. What this project is

A Soft Computing course project (B.Tech CSE, NSUT). We build a system where several **Blue (defender) reinforcement-learning agents** protect a simulated enterprise network against a **Red (attacker)** agent. An **LLM orchestrator** sits *above* the RL agents. It does not control the network directly. It handles coordination, prioritization, conflict flagging and natural-language incident reports.

**Deliverables:** working code, reproducible experiments with comparison results, a formatted report (a 9-page document for Phases 1-3 already exists), and a presentation/viva. Explaining design choices clearly is as important as the numbers.

## 2. Core idea and the design choice we must defend

- RL agents are good at fast, repeated tactical decisions inside a simulator.
- LLMs are good at summarizing, prioritizing and explaining, but are slow, costly and non-deterministic.
- **So we keep them separate:** RL agents act in the environment; the LLM reads summarized state and agent proposals and produces structured guidance and reports. The LLM is **never inside the RL training loop**.

## 3. Environment and benchmark

- **CybORG / CAGE Challenge 2** is the simulator and benchmark. It is a Python library installed from the cage-challenge GitHub repo (no hosting needed).
- CAGE 2 has a single Blue agent. For true multi-agent Blue, **CAGE Challenge 4** (CybORG v4) provides multiple Blue agents. *Verify the details in the repo READMEs.*
- **Open decision (record the answer here once made):**
  - Option A: CAGE 2 for single-agent baselines, CAGE 4 for the multi-agent part.
  - Option B: stay in CAGE 2 and split the Blue action space by subnet ourselves (needs justification in the report).
  - Decision: `TBD`

## 4. Formal framing (from the report)

- Dec-POMDP / POSG model; the Red-vs-Blue interaction is formulated as a zero-sum game.
- Blue and Red action space tables and the CAGE 2 reward formula (with term-by-term rationale) are in `docs/` (the existing Word report). Do not re-derive them; read them.
- Algorithms covered: Q-learning/DQN, PPO, actor-critic, MAPPO.
- Architecture comparison: hierarchical vs MAPPO.

## 5. Tech stack

| Purpose | Tool |
|---|---|
| Language | Python (3.9 or 3.10 is safest for CybORG; follow the CAGE README) |
| Simulator | CybORG (CAGE 2, possibly CAGE 4) |
| RL | Stable-Baselines3 (PPO baseline), PyTorch (custom MAPPO) |
| LLM | Hosted model via API (key in `.env`, never committed) |
| Logging | TensorBoard (or Weights & Biases) |
| Config | YAML files in `configs/` |
| Editor and VCS | VS Code (or any AI IDE), Git + GitHub |

## 6. Repository structure

```
llm-marl-cyber-defense/
├── README.md
├── PROJECT_CONTEXT.md     # this file
├── requirements.txt       # pinned
├── .env.example           # GEMINI_API_KEY=your_key_here (real .env is gitignored)
├── envs/wrappers.py       # obs flattening, reward wrappers, subnet splitting
├── agents/
│   ├── baselines.py       # random, do-nothing, rule-based
│   ├── ppo_single.py      # single-agent PPO
│   └── mappo.py           # multi-agent (independent PPO first, then MAPPO)
├── llm/
│   ├── orchestrator.py    # prioritization, conflict flagging, report writing
│   ├── prompts.py         # prompt templates
│   └── schema.py          # strict JSON output validation
├── experiments/
│   ├── run_baselines.py
│   ├── train_ppo.py
│   ├── train_marl.py
│   └── evaluate.py
├── configs/               # one YAML per experiment (seeds, hyperparameters)
├── results/               # CSVs, plots, sample incident reports
├── notebooks/             # exploration only; core logic lives in .py files
└── docs/                  # report, diagrams
```

## 7. Build roadmap (do in order)

1. **Environment running:** CybORG installed; a random Blue agent runs 100 episodes; mean reward logged.
2. **Baselines:** random, do-nothing, rule-based defender; then single-agent PPO vs each Red strategy (B-line, Meander) and a mix.
3. **Multi-agent Blue:** independent PPO first, then MAPPO (centralized critic, decentralized actors). At least 3 seeds per configuration.
4. **LLM orchestrator:** reads summarized observations and proposed actions; returns validated JSON (priorities, conflicts) plus a readable incident report.
5. **Evaluation and ablation:** RL only vs RL + LLM vs rule-based; report mean and std reward, compromise metrics, LLM cost and latency; remove one component at a time.
6. **Report and demo:** add results to the report, write limitations, record a backup demo episode.

Each step must produce saved results in `results/` before moving on.

## 8. Rules for the AI assistant (important)

- **Do not invent CybORG APIs.** It is a niche library. If unsure of a function, class or wrapper name, say so and point to the file in the repo to check. Prefer reading the source over guessing.
- Pin versions; never upgrade packages without asking.
- Every experiment takes a `--seed` and a `--config` argument and is reproducible with one command.
- Keep functions small and commented; this is a student project that must be explainable in a viva.
- LLM output must always go through `llm/schema.py` validation. Free-form text must never control agents directly.
- Never hardcode API keys. Read from `.env` with python-dotenv.
- Cache LLM responses during development so runs are reproducible and cheap.
- Prefer a simple working version first, then improve. Don't add features that aren't in the roadmap.
- When making a design choice, add one line in `docs/decisions.md` saying what and why.

## 9. Key references

- Landolt et al. (2025)
- Kiely et al., CAGE Challenge 2 and CAGE Challenge 4
- Nguyen and Reddi, IEEE TNNLS (2023)
- Liu (2024)

Full list of 10 numbered references is in the report appendix.

## 10. Glossary (for explanations and viva)

CybORG, MARL, PPO, MAPPO, Dec-POMDP, reward shaping, ablation study, Red/Blue agents, CAGE Challenge, kill chain, payoff matrix.

## 11. First task to give the AI

> Read PROJECT_CONTEXT.md. Write `experiments/run_baselines.py`: create the CAGE 2 environment with a B-line Red agent, run 100 episodes each of a random Blue agent, a do-nothing Blue agent and a simple rule-based Blue agent, and save mean and std reward to `results/baselines.csv`. Take a `--seed` argument. Keep it under 120 lines. Tell me which CybORG wrappers you use and which parts you are unsure about.
