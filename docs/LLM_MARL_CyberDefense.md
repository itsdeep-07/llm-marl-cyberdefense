

# LLM-Augmented Multi-Agent RL for Automated Cyber Incident Response
### Complete Theory + Phase/Batch Notes (Viva Edition)

> **Course:** Soft Computing | **Repo:** `itsdeep-07/llm-marl-cyberdefense`
> **How to use:** Section 0–3 = the 5-minute story. Sections 4–9 = phase-by-phase depth. Section 10–12 = viva Q&A, limitations, glossary.
> **Honesty rule for viva:** Anything marked 🟡 *PLANNED* or *ILLUSTRATIVE* must be presented as planned/illustrative. Never quote a number you have not produced.

---

## 0. One-Paragraph Pitch (memorise this)

We build an **automated cyber incident response system** on a simulated enterprise network. Several **Blue (defender) RL agents** learn, through trial and error, to detect and contain a **Red (attacker)** — using algorithms like PPO and MAPPO inside the **CybORG / CAGE Challenge** environment. On top of the RL agents sits an **LLM orchestrator** that does what RL is bad at: reading the situation in language, **prioritising** which hosts/zones matter most, **coordinating** agents, and writing a **human-readable incident report**. RL gives fast, learned, reliable *actions*; the LLM gives *reasoning, prioritisation and explanation*. We keep these two concerns **deliberately separate** so the system stays fast, safe and explainable.

**End goal:** a working, reproducible pipeline + dashboard that (1) trains defenders, (2) shows they beat baselines, (3) shows LLM augmentation adds measurable value (via ablation), and (4) produces analyst-style incident reports.

---

## 1. Problem Statement & Motivation

| Question | Answer |
|---|---|
| What is the problem? | Security Operations Centres (SOCs) face thousands of alerts; human analysts are slow, overloaded, and attackers move in minutes. |
| Why not rules/playbooks? | Static rules (SIEM signatures, SOAR playbooks) can't adapt to novel attacker behaviour and don't generalise. |
| Why RL? | Incident response is **sequential decision-making under uncertainty** with delayed reward — exactly the RL setting. |
| Why *multi*-agent? | Real networks are partitioned (zones/subnets); defenders have **partial views** and must coordinate. |
| Why an LLM on top? | RL policies are opaque numeric networks. LLMs read logs/alerts, reason in language, prioritise, and **explain**. |
| Why is it "Soft Computing"? | Soft computing = tolerance to imprecision, uncertainty, approximation. We use neural function approximation (DQN/PPO nets), learning from noisy partial observations, heuristic/approximate reasoning (LLM), and optionally fuzzy severity scoring. (See §11 Q1.) |

---

## 2. End Goal, Deliverables, and What the Output Looks Like

### 2.1 Deliverables
1. **Codebase** (public GitHub) — env wrappers, agents, LLM orchestrator, evaluation scripts.
2. **Trained models** — checkpoints for baselines, single-agent PPO, multi-agent (hierarchical / MAPPO).
3. **Evaluation results** — learning curves, comparison tables, ablation study.
4. **Streamlit dashboard** — live/replay view of network state, agent actions, rewards, LLM reports.
5. **Documentation** — formal write-up (POSG/Dec-POMDP model, algorithms, reward, references) + these notes.
6. **Demo** — a recorded or live episode: attack unfolds → agents respond → LLM report generated.

### 2.2 What the final output looks like (illustrative mock-ups)

**(a) Learning curve** — x: training steps, y: mean episode return. Expected shape: random < heuristic < PPO < MAPPO/hierarchical (🟡 *ILLUSTRATIVE ordering, to be verified by our runs*).

**(b) Comparison table** (fill with real numbers only):

| Method | Mean Return ↑ | Op-Server Compromised % ↓ | Restore Actions ↓ | Notes |
|---|---|---|---|---|
| Random Blue | _TBD_ | _TBD_ | _TBD_ | lower bound |
| Rule-based Blue (e.g., Monitor→Analyse→Remove/Restore) | _TBD_ | _TBD_ | _TBD_ | heuristic baseline |
| DQN (single-agent) | _TBD_ | _TBD_ | _TBD_ | value-based |
| PPO (single-agent) | _TBD_ | _TBD_ | _TBD_ | policy-gradient |
| Hierarchical / MAPPO (multi-agent) | _TBD_ | _TBD_ | _TBD_ | CTDE |
| MARL + LLM orchestrator | _TBD_ | _TBD_ | _TBD_ | full system |

**(c) Example LLM incident report** (🟡 *ILLUSTRATIVE format — real text is generated at runtime*):

```
INCIDENT REPORT  |  Episode 12, Step 18  |  Severity: HIGH
Summary : Red foothold established on User subnet host User2; lateral movement
          toward Enterprise subnet detected.
Evidence: Suspicious process alerts on User2 (Analyse), new connection to Enterprise1.
Actions : Blue-Agent-1 → Remove (User2); Blue-Agent-2 → DeployDecoy (Enterprise1).
Priority: Protect Operational subnet (mission-critical). Recommend Restore only if
          Red privilege level = Admin on a critical host (Restore is costly).
Status  : Contained / Ongoing / Escalate.
```

**(d) Dashboard screens:** network graph coloured by host state (clean / suspected / compromised / decoy), per-step action log, cumulative reward chart, LLM report panel, model-selector (baseline vs PPO vs MAPPO vs +LLM).

---

## 3. System Architecture

```
                      ┌───────────────────────────────────────────┐
                      │            LLM ORCHESTRATOR                │
                      │  (summarise → prioritise → advise → report)│
                      └───────▲───────────────┬────────────────────┘
        structured summary    │               │  priorities / advice /
        (alerts, host states, │               │  natural-language report
         recent actions)      │               ▼
   ┌──────────────────────────┴───────────────────────────────────┐
   │              ORCHESTRATION & SAFETY LAYER                     │
   │  state summariser · prompt builder · output parser ·          │
   │  action validator / mask · fallback (if LLM fails/timeouts)   │
   └──────▲─────────────────────────────────────────┬──────────────┘
          │ observations                            │ validated advice
   ┌──────┴──────────┐                    ┌─────────▼─────────┐
   │  CybORG / CAGE   │◄── Blue actions ───│  BLUE RL AGENTS   │
   │  Simulation Env  │                    │ (PPO / MAPPO /    │
   │  (network + Red  │── observations ───►│  Hierarchical)    │
   │   + Green agents)│── rewards ────────►│                   │
   └──────────────────┘                    └───────────────────┘
          │ logs / metrics
          ▼
   ┌────────────────────┐
   │ Evaluation + Dashboard │ (Streamlit, plots, tables)
   └────────────────────┘
```

### 3.1 The central design decision (VIVA FAVOURITE)
**Q: Why not let the LLM choose every action?**
- **Latency/cost:** RL step = microseconds–ms; LLM call = seconds. Thousands of steps × thousands of episodes is infeasible.
- **Determinism & safety:** LLM outputs are stochastic and can hallucinate; a wrong "Restore" on a critical server has real cost.
- **Training:** RL policies are *trained* against the reward; LLM isn't optimised for this reward.
- **Explainability:** LLM is excellent at *explaining* and *prioritising*, which RL is not.
- **Hence separation of concerns:** RL = fast reliable *execution*; LLM = slow, high-level *reasoning/reporting*, invoked every N steps or on events, with all outputs *validated* before influencing anything.

---

## 4. Proposed File Structure

> 🟡 *Align this tree with your actual repo before presenting; keep the folder purposes, adjust names.*

```
llm-marl-cyberdefense/
├── README.md                    # overview, setup, how to run
├── requirements.txt / pyproject.toml
├── configs/
│   ├── env_cage2.yaml           # episode length, red agent type, seeds
│   ├── env_cage4.yaml           # multi-agent environment settings
│   ├── ppo.yaml                 # lr, gamma, lambda, clip eps, batch size
│   ├── mappo.yaml
│   └── llm.yaml                 # model, temperature, call frequency
├── src/
│   ├── envs/
│   │   ├── cage2_wrapper.py     # Gym-style wrapper over CybORG (CAGE 2)
│   │   ├── cage4_wrapper.py     # PettingZoo/RLlib-style multi-agent wrapper
│   │   └── reward_shaping.py    # reward/ablation variants
│   ├── agents/
│   │   ├── baselines.py         # random, rule-based Blue
│   │   ├── dqn.py               # value-based baseline
│   │   ├── ppo.py               # single-agent PPO
│   │   ├── mappo.py             # centralised critic, decentralised actors
│   │   └── hierarchical.py      # meta-controller + sub-policies
│   ├── llm/
│   │   ├── summariser.py        # obs/alerts → structured text/JSON
│   │   ├── prompts.py           # system/user prompt templates
│   │   ├── orchestrator.py      # priority + advice generation
│   │   ├── reporter.py          # natural-language incident report
│   │   └── guardrails.py        # JSON schema validation, action masking, fallback
│   ├── training/
│   │   ├── train_single.py
│   │   ├── train_multi.py
│   │   └── callbacks.py         # logging, checkpointing
│   ├── evaluation/
│   │   ├── evaluate.py          # run N episodes per method, fixed seeds
│   │   ├── metrics.py           # return, compromise time, restore count…
│   │   └── ablation.py          # no-LLM vs LLM, reward variants, etc.
│   └── utils/ (seeding, logging, plotting)
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── results/
│   ├── checkpoints/
│   ├── logs/                    # TensorBoard / CSV
│   ├── figures/
│   └── reports/                 # saved LLM incident reports
├── notebooks/                   # exploration, plots
├── docs/                        # formal Word doc, these notes, references
└── tests/                       # wrapper, parser, guardrail unit tests
```

---

## 5. Tools & Technology Stack — What, How, Why

| Tool | What it is | How we use it | Why this (and not alternatives) |
|---|---|---|---|
| **CybORG** | Cyber Operations Research Gym — simulator of networks, attackers, defenders (from the Australian DSTG / TTCP CAGE effort) | Provides the environment: hosts, subnets, services, Red/Blue actions, observations, rewards | Standard academic benchmark (CAGE Challenges) → results are comparable and credible; building our own simulator is out of scope |
| **CAGE Challenge 2** | Single Blue agent defending a 13-host, 3-subnet network vs scripted Red | Phase 2 baseline & algorithm study | Simple, well-documented, many published baselines |
| **CAGE Challenge 4** | Multi-agent: several Blue agents, each responsible for a zone, plus Red and benign Green agents, mission phases, limited inter-agent messaging | Phase 3 multi-agent work (MAPPO / hierarchical) | Genuinely *multi-agent* & partially observable — matches Dec-POMDP theory |
| **Python 3.x** | Language | Everything | Ecosystem for RL/LLM |
| **PyTorch** | Deep learning framework | Policy/value networks | Flexible, standard in research |
| **Ray RLlib / Stable-Baselines3** | RL libraries with PPO etc. | Training baselines; RLlib for multi-agent | Tested implementations → we focus on the *research question*, not re-deriving PPO bugs. (SB3 = simple single-agent; RLlib = multi-agent scale.) |
| **Gymnasium / PettingZoo-style API** | Standard env interfaces | Wrap CybORG | Lets any RL library plug in |
| **LLM (API or local)** | Language model | Orchestrator + reporter. Options: free-tier hosted models or local via Ollama (Llama/Qwen class) | Free/low-cost constraint; local gives privacy + no quota; hosted gives quality. Choice is a config switch (`llm.yaml`) |
| **Prompt + JSON-schema output** | Structured prompting | LLM returns priorities/advice as JSON we can validate | Prevents free-form text from leaking into control path |
| **TensorBoard / CSV logging** | Experiment tracking | Learning curves | Reproducibility |
| **Matplotlib / Pandas** | Analysis | Plots & tables | Standard |
| **Streamlit** | Python dashboard framework | Live/replay visualisation | Fastest path from Python to UI; no frontend build |
| **Git/GitHub** | Version control | Public repo | Reproducibility, evidence of work |
| **`docx` (Node.js)** | Document generation | Produced the formal 9-page Phase 1–3 report | Programmatic, consistent formatting |

---

## 6. Theory Foundations (the "deep" part of viva)

### 6.1 Reinforcement Learning — core vocabulary
- **Agent, environment, state s, action a, reward r, policy π(a|s), return G = Σ γᵗ rₜ**, discount factor γ∈[0,1).
- **Value functions:** V^π(s) = E[G | s]; Q^π(s,a) = E[G | s,a]. **Advantage** A(s,a) = Q(s,a) − V(s) — "how much better than average was this action".
- **Exploration vs exploitation:** ε-greedy (DQN), entropy bonus (PPO).
- **Markov property:** future depends only on current state — *violated* in partial observability, which is why we need POMDPs/recurrence/history.

### 6.2 From MDP to Dec-POMDP to POSG (the formal model)

| Model | Tuple (informal) | What it adds |
|---|---|---|
| **MDP** | ⟨S, A, T, R, γ⟩ | Full observability, single agent |
| **POMDP** | ⟨S, A, T, R, Ω, O, γ⟩ | Agent sees **observations** o ~ O(·\|s,a), not state |
| **Dec-POMDP** | ⟨I, S, {Aᵢ}, T, R, {Ωᵢ}, O, γ⟩ | **Multiple cooperating agents**, each with own actions/observations, **one shared reward** |
| **POSG** | ⟨I, S, {Aᵢ}, T, {Rᵢ}, {Ωᵢ}, O, γ⟩ | Agents have **separate rewards** → allows competition |
| **Zero-sum POSG** | R_blue = −R_red | Pure adversarial game |

**Our modelling:** Blue agents form a *cooperative team* (Dec-POMDP among themselves) playing against Red (POSG overall). Because Red in CAGE is a **scripted** agent (fixed strategy), practically Red is part of the environment dynamics, but the formulation is a **zero-sum game** at the conceptual level — useful to explain payoff matrices and why attacker/defender strategies interact. *(Viva: "Is Red learning?" → In the CAGE benchmark Red is scripted/heuristic; learning Red = future work / self-play.)*

**Why partial observability?** Defender cannot see which hosts are compromised; it only sees alerts/scan results, which are **noisy** (false negatives possible). That is why "Analyse" and "Monitor" actions exist — they buy information.

### 6.3 Value-based RL — Q-learning & DQN
- **Q-learning update:** Q(s,a) ← Q(s,a) + α[r + γ maxₐ′ Q(s′,a′) − Q(s,a)] (off-policy, tabular).
- **DQN:** Q approximated by a neural net; loss L = (r + γ maxₐ′ Q_target(s′,a′) − Q(s,a))².
  - **Experience replay:** break correlation between consecutive samples.
  - **Target network:** stabilise the moving target.
- **Limits:** overestimation bias, discrete actions only, can be unstable/sample-inefficient on large action spaces.

### 6.4 Policy-gradient & Actor–Critic
- **Policy gradient:** ∇J(θ) = E[∇ log π_θ(a|s) · A(s,a)]. Directly optimises the policy; works with stochastic policies (good for hiding intent / exploration).
- **Actor–Critic:** *Actor* = policy network; *Critic* = value network estimating V(s) to compute advantage → **lower variance** than pure REINFORCE.
- **GAE (Generalised Advantage Estimation):** Aₜ = Σₗ (γλ)ˡ δₜ₊ₗ, with δₜ = rₜ + γV(sₜ₊₁) − V(sₜ). λ trades bias vs variance.

### 6.5 PPO (Proximal Policy Optimization)
- Ratio: rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)
- **Clipped objective:** L^CLIP = E[ min( rₜ·Aₜ , clip(rₜ, 1−ε, 1+ε)·Aₜ ) ]
- **Intuition:** don't let one update move the policy too far (trust-region-like behaviour, but first-order & simple). ε≈0.1–0.2.
- Total loss = −L^CLIP + c₁·(value loss) − c₂·(entropy bonus).
- **Why PPO for us:** stable, handles discrete actions, works on-policy with parallel envs, strong CAGE results in literature, easy to extend to multi-agent (MAPPO).

### 6.6 Multi-Agent RL (MARL)
**Core challenges:**
1. **Non-stationarity** — other agents are learning, so the environment looks like it keeps changing.
2. **Credit assignment** — a team reward: which agent deserved it?
3. **Partial observability** — each agent sees only its zone.
4. **Scalability** — joint action space grows exponentially.
5. **Coordination / communication.**

**Paradigms:**
| Paradigm | Idea | Pros | Cons |
|---|---|---|---|
| **Independent learners (IPPO / IQL)** | Each agent trains alone, treats others as environment | Simple, scalable | Non-stationarity, no explicit coordination |
| **Centralised training, decentralised execution (CTDE)** | Critic sees global info *during training*; actors use only local obs at run-time | Stable training; deployable | Needs global state at training time |
| **MAPPO** | PPO + centralised value function, shared/individual actors | Surprisingly strong baseline across MARL benchmarks; simple | Needs tuning; centralised critic input grows with agents |
| **Value decomposition (VDN/QMIX)** | Decompose team Q into per-agent Qs | Good credit assignment (cooperative) | Monotonicity assumptions |
| **Hierarchical RL** | Meta-controller picks sub-goal/sub-policy; sub-policies act | Temporal abstraction, specialisation (e.g., "contain" vs "recover") | Harder to train; design of hierarchy matters |

**Our Phase-3 question:** *Hierarchical vs MAPPO — which gives better defence?* (Hierarchical is a strong CAGE-2 pattern; MAPPO is the principled MARL baseline.)

### 6.7 Reward design & shaping
- CAGE-style reward is **negative-dominant** (penalties): Blue is penalised per step for Red presence on hosts (weighted by host importance), a large penalty for Red **Impact** on the operational server, and a cost for **Restore** (so Blue doesn't spam it).
- *(Confirm exact constants against your Word doc's term-by-term table before viva. Commonly cited: Red admin on a user host ≈ −0.1/step; on enterprise/operational hosts ≈ −1/step; Impact on operational server ≈ −10; Restore ≈ −1.)*
- **Why these terms:** encode *business priority* (operational > enterprise > user) and *cost of response* (restoring disrupts service).
- **Reward shaping:** adding auxiliary rewards to speed learning (e.g., small bonus for correct Analyse before Restore). **Risk:** reward hacking / misaligned incentives; shaping must not change the optimal policy (potential-based shaping preserves it).
- **Ablation on reward:** train with/without terms to show each one's contribution.

### 6.8 LLM concepts we rely on
- **Transformer / autoregressive LLM:** predicts next token; instruction-tuned for following prompts.
- **Prompt engineering:** system prompt (role + rules), few-shot examples, structured output (JSON).
- **Grounding:** feed the LLM *real* state summaries so it doesn't invent facts.
- **Hallucination:** confident but false output → mitigated by **schema validation, allow-lists of host/action names, low temperature, fallback**.
- **Retrieval-Augmented Generation (RAG) — optional extension:** retrieve MITRE ATT&CK-style knowledge / playbooks to ground recommendations.
- **Where LLM adds value:** summarisation, prioritisation, explanation, cross-zone coordination text, post-incident reporting. **Where it should not be trusted:** low-level, high-frequency control.

### 6.9 Cybersecurity concepts
- **Kill chain / attack stages:** reconnaissance → initial access → privilege escalation → lateral movement → impact. Maps to Red actions: *DiscoverRemoteSystems → DiscoverNetworkServices → ExploitRemoteService → PrivilegeEscalate → Impact*.
- **Blue actions (CAGE 2):** *Monitor* (passive/automatic info), *Analyse* (inspect a host for malware/compromise), *Remove* (kill Red user-level presence), *Restore* (rebuild host — removes admin-level Red, but costly), *DeployDecoy* (honeypot services that distract/detect Red).
- **Red agent types in CAGE 2:** e.g., *B-line* (knows the topology, heads straight for the operational server), *Meander* (explores broadly), plus *Sleep*. Evaluating against several attacker types tests **robustness/generalisation**.
- **Defence-in-depth, honeypots/decoys, containment vs eradication vs recovery** (NIST incident-response lifecycle: Preparation → Detection & Analysis → Containment/Eradication/Recovery → Post-incident).
- **CAGE 4 additions:** multiple Blue agents per zone, benign **Green** users creating noise (false positives), **mission phases** that change what is critical, and zone-level traffic blocking/allowing actions, limited message passing between Blue agents. *(Verify exact action set/version in the repo you cloned.)*

---

## 7. Phase & Batch Plan — What / How / Why / Output

> 🟡 *Phases 1–3 match the formal document already produced. The batch numbering and Phases 4–6 are laid out as the logical build order — rename to match your team's actual plan if it differs.*

### PHASE 1 — Foundations & Problem Formulation  ✅ (documented)
**Objective:** Define the problem rigorously before any code.

| Batch | Work | How | Why | Output |
|---|---|---|---|---|
| 1.1 Literature | Read & map 7+ papers (Landolt et al. 2025; Kiely et al. CAGE 2 & 4; Nguyen & Reddi, IEEE TNNLS 2023; Liu 2024; PPO/MAPPO sources) | Paper → project-component mapping table | Justify every design choice with a citation | Reference list + section-to-reference appendix |
| 1.2 Formal model | Write POSG / Dec-POMDP, zero-sum formulation | Tuple definitions, observation/state/action spaces | Gives mathematical grounding; viva-proof | Formal model section |
| 1.3 Env understanding | Network topology, Red/Blue action tables, reward formula | Read CybORG source + CAGE docs | You can't evaluate what you don't understand | Action-space tables, reward term-by-term rationale |
| 1.4 Algorithm theory | Q-learning/DQN, PPO, actor-critic, MAPPO + comparison table | Derive/compare | Choose algorithms with reasons | MARL algorithm comparison table |

**Key viva points:** why Dec-POMDP not MDP; why zero-sum; why PPO/MAPPO; what each reward term encodes.

---

### PHASE 2 — Single-Agent Baseline (CAGE 2)  ✅ methodology documented / 🟡 results as per runs
**Objective:** Establish a trustworthy baseline and a working training pipeline.

| Batch | Work | How | Why | Output |
|---|---|---|---|---|
| 2.1 Env setup & wrapper | Install CybORG, wrap as Gym env, fix seeds | `cage2_wrapper.py`; verify one random episode runs | Reproducible foundation | Working env + smoke test |
| 2.2 Non-learning baselines | Random Blue; rule-based Blue | Implement simple heuristics (e.g., Analyse suspicious host → Remove; Restore if admin) | **Lower bounds** — RL must beat these to be meaningful | Baseline return table |
| 2.3 DQN | Train value-based agent | Replay buffer + target net | Show value-based approach & its limits | DQN curve |
| 2.4 PPO | Train PPO agent | Tune lr, γ, λ, clip ε, entropy coef; multiple seeds | Main single-agent method | PPO curve + checkpoint |
| 2.5 Evaluation | N episodes × several Red types × several seeds | `evaluate.py`; mean ± std | Statistical credibility; robustness | Comparison table + plots |

**Key viva points:** why multiple seeds (RL variance); why evaluate vs different Red types (generalisation); overfitting to B-line; learning-curve interpretation; hyperparameter sensitivity.

---

### PHASE 3 — Multi-Agent Defence (CAGE 4 / MARL)  🟡 build phase
**Objective:** Defend with multiple coordinated agents; compare coordination strategies.

| Batch | Work | How | Why | Output |
|---|---|---|---|---|
| 3.1 Multi-agent env | Get CAGE 4 running; PettingZoo/RLlib-style wrapper | Per-agent obs/action spaces, parameter sharing option | Real multi-agent partial observability | Working multi-agent loop |
| 3.2 IPPO baseline | Independent PPO per agent (or shared weights) | Each agent trains on its local obs | Simple multi-agent baseline; exposes non-stationarity | IPPO results |
| 3.3 MAPPO | Centralised critic, decentralised actors | Critic gets global/concat state during training only | CTDE = stable + deployable | MAPPO results |
| 3.4 Hierarchical | Meta-controller selects mode/sub-policy (e.g., *investigate / contain / recover*) | Two-level policy; sub-policies pretrained or jointly trained | Temporal abstraction, specialisation | Hierarchical results |
| 3.5 Comparison | Hierarchical vs MAPPO vs IPPO | Same seeds, same Red/Green settings | Answer the core research question fairly | Comparison table + curves |

**Key viva points:** non-stationarity; credit assignment; why CTDE; parameter sharing pros/cons; communication limits in CAGE 4; why CAGE 4 over CAGE 2 for multi-agent (CAGE 2 is single-Blue).

---

### PHASE 4 — LLM Orchestrator  🟡 PLANNED
**Objective:** Add language-level reasoning above the RL agents *without* touching low-level control.

| Batch | Work | How | Why | Output |
|---|---|---|---|---|
| 4.1 State summariser | Convert observations/alerts/recent actions → compact structured text/JSON | Deterministic code (no LLM) | LLM must see *grounded, concise* facts; avoids token bloat | `summariser.py` |
| 4.2 Prompt design | System prompt (SOC analyst role, rules, output schema) + few-shot examples | Iterative prompt testing on recorded episodes | Reliability of structured output | `prompts.py` |
| 4.3 Prioritisation & advice | LLM outputs ranked zones/hosts + advisory per agent | JSON schema; invoked every N steps or on trigger events (e.g., new compromise) | High-level coordination at low frequency | `orchestrator.py` |
| 4.4 Guardrails | Validate JSON, allow-list hosts/actions, timeout, fallback to pure-RL | `guardrails.py` | LLM must never break the system | Safe integration |
| 4.5 Integration mode | Decide *how advice influences RL*: (a) **report-only** (zero influence), (b) **priority hints** (e.g., bias action masks / reward weights / meta-controller input), (c) **veto/suggest with validation** | Start with (a)→(b); keep (c) experimental | Incremental risk control; clear ablation ladder | Integrated pipeline |
| 4.6 Reporter | Episode/incident → analyst-style report | Template + LLM narrative | The explainability payoff | Sample reports |

**Key viva points:** why LLM isn't in the inner loop; hallucination mitigation; how advice is "injected"; what happens when LLM is wrong/slow; call frequency vs benefit; free/local model choice and its limits.

---

### PHASE 5 — Dashboard & System Integration  🟡 partly built
**Objective:** Make the system observable and demonstrable.

| Batch | Work | How | Why | Output |
|---|---|---|---|---|
| 5.1 Episode replay | Log full episode (states, actions, rewards) to JSON/CSV | Logger in the env loop | Replay without rerunning | Episode logs |
| 5.2 Streamlit UI | Network graph, host status colours, action timeline, reward chart | `dashboard/app.py` | Viva demo & intuition | Dashboard |
| 5.3 Model switcher | Select baseline / PPO / MAPPO / +LLM | Load checkpoints | Side-by-side comparison live | Interactive comparison |
| 5.4 Report panel | Show LLM report per step/episode | Read saved reports | Show explainability | Report view |

---

### PHASE 6 — Evaluation, Ablation & Final Deliverable  🟡 PLANNED
**Objective:** Prove each component matters, honestly.

**Ablation ladder (each row adds exactly one thing):**
1. Random → 2. Rule-based → 3. Single-agent PPO → 4. Multi-agent (IPPO) → 5. MAPPO → 6. Hierarchical → 7. + LLM report only → 8. + LLM priority hints.
Plus: **reward ablation** (remove terms), **Red-type generalisation** (train vs B-line, test vs Meander and vice-versa), **LLM call-frequency ablation**, **LLM-failure robustness** (inject bad outputs → fallback works?).

**Metrics:**
- Mean/median episode **return** (± std over seeds)
- **Time-to-containment**, % episodes where operational server impacted
- **Number of Restore actions** (response cost)
- **False-positive responses** (acting on clean hosts — esp. important with Green noise in CAGE 4)
- **Sample efficiency** (steps to reach a return threshold)
- LLM-side: **schema-valid output rate**, **latency per call**, **report quality** (small human rubric: accuracy, completeness, actionability)

**Statistics:** fixed seeds, ≥ several seeds per method, report mean ± std (or confidence intervals); don't cherry-pick best run.

**Final output:** results tables/figures, trained checkpoints, dashboard demo, report, README, viva-ready notes.

---

## 8. How Training Actually Works (step-by-step, for "explain PPO in practice")

1. Initialise actor π_θ and critic V_φ.
2. Collect rollouts: for T steps (parallel envs), store (sₜ, aₜ, rₜ, logπ_old(aₜ|sₜ), V(sₜ)).
3. Compute returns and **GAE advantages**; normalise advantages.
4. For K epochs over minibatches: compute ratio, clipped loss, value loss, entropy bonus → gradient step.
5. Replace old policy with new; repeat.
6. Log mean return, entropy, KL, value loss; checkpoint best & final.

**MAPPO delta:** critic input = global state (or concatenation of all agents' obs) during training; each actor still uses only its own observation at execution.

**Hierarchical delta:** meta-policy picks high-level option every k steps; low-level policy picks the concrete action within that option.

---

## 9. Design Justifications Cheat-Sheet ("why X not Y")

| Choice | Alternative | Why we chose ours |
|---|---|---|
| PPO | DQN, A2C, SAC, TRPO | Stable, simple, discrete-action friendly, strong in CAGE literature; TRPO is heavier; SAC is continuous-oriented |
| MAPPO | QMIX, MADDPG | Simple, strong empirical baseline, works with discrete actions & cooperative reward; MADDPG is continuous-action oriented |
| CTDE | Fully centralised | Fully centralised execution isn't realistic (agents have local views) |
| CybORG/CAGE | Custom simulator, real testbed | Benchmark credibility, no safety/legal issues of real attacks |
| LLM above RL | LLM-only agent, RL-only | LLM-only too slow/unreliable; RL-only is opaque and can't reason/explain |
| JSON-schema outputs | Free text | Validatable, safe to parse |
| Streamlit | React/Next.js frontend | Python-native, fastest to ship (note: team *can* do MERN/Next.js, but dashboard is not the research contribution) |
| Scripted Red | Learning Red | Benchmark standard; learning Red = self-play future work |

---

## 10. Viva Question Bank

### 10.1 Concept & motivation
1. **Explain your project in 1 minute.** → §0.
2. **How is this Soft Computing?** → Neural function approximation (policy/value nets), learning under uncertainty/partial observability, approximate/heuristic reasoning (LLM), optional fuzzy severity scoring; contrast with hard-coded exact rules.
3. **Why RL and not supervised learning?** No labelled "correct response" dataset; actions change future state; reward is delayed → sequential decision problem.
4. **Why simulate instead of real network?** Safety, legality, reproducibility, cost, ability to run millions of episodes.
5. **What is the novelty/contribution?** Combining MARL execution with an LLM reasoning/reporting layer under a strict separation-of-concerns + guardrails, and measuring its effect via ablation. (Don't overclaim — this is a course project building on published work.)

### 10.2 RL theory
6. Difference between **value-based and policy-based** methods? On-policy vs off-policy?
7. Why **replay buffer & target network** in DQN?
8. Write the **PPO objective** and explain clipping. What if ε is too large/small?
9. What is **advantage**, and why subtract a baseline? What does **GAE λ** do?
10. Why does PPO use **entropy bonus**?
11. **Discount factor γ:** what happens at γ→0 / γ→1 in our setting? (Short-sighted vs long-horizon: containing early prevents later impact → need high γ.)
12. What is the **exploration problem** in this environment? Sparse large penalty (Impact) vs small per-step penalties.
13. **Why did returns have high variance across seeds?** Stochastic env, policy randomness, initial conditions.

### 10.3 Formal models
14. MDP vs POMDP vs Dec-POMDP vs POSG — differences and which applies where.
15. Why **zero-sum**? What is the **payoff matrix** view? Nash equilibrium meaning here? (Note: scripted Red ⇒ not a true equilibrium computation; it's a modelling lens.)
16. What is a **belief state**? How do recurrent policies / frame-stacking approximate it?

### 10.4 MARL
17. **Non-stationarity** — explain and how CTDE helps.
18. **Credit assignment** — how do you handle a shared team reward?
19. MAPPO vs IPPO vs QMIX vs MADDPG.
20. **Parameter sharing** — when valid? (Homogeneous roles; breaks if agents differ in action/obs spaces — CAGE 4 zones differ, so padding/masking or per-agent heads may be needed.)
21. Hierarchical RL — what's an option/meta-controller? Why might it beat flat PPO? Failure modes?
22. How do agents **communicate** in CAGE 4? Limits and why it matters.

### 10.5 Environment & cyber
23. Describe the CAGE 2 network (13 hosts, 3 subnets: User / Enterprise / Operational) and the **attack path** to the operational server.
24. List Blue & Red actions and the **kill-chain mapping**.
25. **Why is Restore penalised?** Why are Decoys useful?
26. How does **observation noise** arise? What are false negatives?
27. Why evaluate against **multiple Red strategies**? What is **overfitting to the attacker**?
28. CAGE 2 vs CAGE 4 differences (single vs multi-agent, Green noise, mission phases, zone actions).

### 10.6 LLM layer
29. Why isn't the LLM controlling actions directly?
30. How do you **prevent hallucination** from affecting the system?
31. What happens if the LLM is **down, slow, or wrong**? (Fallback to RL-only; validated outputs; timeouts.)
32. How exactly does LLM advice **influence** the agents? (Be precise: report-only vs priority hints; state which one is implemented.)
33. **How do you evaluate the LLM's contribution** when it's non-deterministic? (Multiple runs, fixed temperature, ablation ladder, schema-valid rate, rubric for reports.)
34. **Prompt injection** risk: can attacker-controlled log text manipulate the LLM? (Yes — logs are untrusted input; mitigations: sanitise/summarise via code, never pass raw attacker strings as instructions, strict output schema, allow-lists.)
35. Why that specific model (free/local)? Trade-offs: quality, latency, privacy, quota.

### 10.7 Evaluation & ablation
36. What is an **ablation study** and which ones did you run/plan?
37. Why **mean ± std over seeds**? What is statistical significance in RL?
38. What would you conclude if LLM variant ≈ no-LLM variant? (Honest answer: LLM adds explainability but little return; that's still a valid finding.)
39. How do you ensure **reproducibility**? (Seeds, configs, pinned versions, public repo.)

### 10.8 Limitations, ethics, future work
40. **Sim-to-real gap** — what breaks in a real SOC? (Scale, noisy logs, unknown topology, adversarial adaptation.)
41. **Adversarial robustness** — Red could exploit a learned policy; adaptive/learning Red & self-play as future work.
42. **Safety/ethics** — automated response can disrupt business (false-positive restore); human-in-the-loop for high-impact actions; dual-use of attacker models.
43. **Scalability** — larger networks, graph neural network policies, transfer learning.
44. **Future work:** learning Red (self-play), RAG with ATT&CK/playbooks, graph-based observations, formal verification of guardrails, real log ingestion.

### 10.9 Trick questions (be ready)
- *"Your PPO beats rule-based but is it better than the CAGE winners?"* → We don't claim SOTA; goal is correct pipeline + comparative study.
- *"Isn't the LLM just a gimmick?"* → It targets what RL lacks (explanation, prioritisation); we measure it; worst case it's report-only with zero risk.
- *"If Red is scripted, why talk about game theory?"* → Modelling lens + future self-play; be upfront about the scripted-Red limitation.
- *"Why no real data?"* → Safety/legal; benchmark standard; simulated alerts approximate real telemetry only coarsely.

---

## 11. Handling "Is this Soft Computing?" (detailed answer)
Soft computing's pillars: **neural networks, fuzzy logic, evolutionary/probabilistic computing** — methods that tolerate imprecision and uncertainty.
- **Neural:** DQN/PPO/MAPPO networks approximate value and policy functions; the LLM is a large neural network.
- **Uncertainty:** POMDP/Dec-POMDP, noisy observations, stochastic policies, probabilistic reasoning.
- **Heuristic/approximate optimisation:** RL finds good (not provably optimal) policies in intractable spaces (Dec-POMDPs are NEXP-complete).
- **Optional fuzzy extension:** fuzzy severity scoring (e.g., IF host criticality *high* AND alert confidence *medium* THEN priority *high*) as a transparent complement to the LLM prioritiser. *(Mention as extension unless actually implemented.)*
- **Optional evolutionary extension:** evolutionary strategies / GA for hyperparameter search as an alternative to manual tuning.

---

## 12. Glossary (quick-fire)

| Term | Meaning |
|---|---|
| **CybORG** | Cyber Operations Research Gym — the simulation framework |
| **CAGE Challenge** | Cyber Autonomy Gym for Experimentation — competition series built on CybORG |
| **Blue / Red / Green** | Defender / attacker / benign user agents |
| **Dec-POMDP** | Decentralised POMDP — cooperative multi-agent, shared reward |
| **POSG** | Partially Observable Stochastic Game — general (possibly competitive) multi-agent model |
| **CTDE** | Centralised Training, Decentralised Execution |
| **PPO / MAPPO** | Proximal Policy Optimization / Multi-Agent PPO |
| **GAE** | Generalised Advantage Estimation |
| **Advantage** | Q − V; how much better an action is than average |
| **Reward shaping** | Adding auxiliary reward terms to guide learning |
| **Ablation study** | Removing/adding one component to measure its effect |
| **Kill chain** | Ordered stages of an attack |
| **Decoy / honeypot** | Fake service that lures/detects attacker |
| **Orchestrator** | Higher-level controller that coordinates lower-level agents |
| **Hallucination** | LLM producing plausible but false content |
| **Guardrail** | Validation/constraint layer around model output |
| **Sim-to-real gap** | Performance drop moving from simulation to real systems |
| **Prompt injection** | Attacker text manipulating an LLM's instructions |

---

## 13. References (map to your Word doc's numbered list)
- Kiely et al. — *CAGE Challenge 2* and *CAGE Challenge 4* reports (Autonomous Cyber Defence benchmark)
- Landolt et al. (2025) — multi-agent RL for cyber defence
- Nguyen & Reddi (IEEE TNNLS 2023) — RL for cybersecurity survey
- Liu (2024) — LLM/RL related work used in the project
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Schulman et al. (2016) — *High-Dimensional Continuous Control Using GAE*
- Yu et al. (2022) — *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO)
- Mnih et al. (2015) — *Human-level control through deep RL* (DQN)
- Oliehoek & Amato (2016) — *A Concise Introduction to Decentralized POMDPs*
- Standen et al. — *CybORG: An Autonomous Cyber Operations Research Gym*

> ⚠️ **Before viva:** open your Word doc's reference appendix and make the citation strings here match it exactly; verify the constants (reward values, action counts) against the CybORG version you actually installed.

---

## 14. Pre-Viva Checklist
- [ ] Can I draw the architecture (§3) from memory and defend the RL/LLM separation?
- [ ] Can I write the PPO clipped objective and explain every symbol?
- [ ] Can I explain MDP → POMDP → Dec-POMDP → POSG in 60 seconds?
- [ ] Do I know exactly which LLM-integration mode (report-only / hints) is *implemented*?
- [ ] Do I have **real numbers** for baseline vs PPO vs multi-agent (and none invented)?
- [ ] Can I run the demo end-to-end offline (cached LLM outputs as backup)?
- [ ] Do I know the limitations and future work cold?
