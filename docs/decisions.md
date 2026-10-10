# Architecture & Design Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-04 | Separate LLM from RL loop | RL agents require deterministic, high-throughput action execution. LLMs are non-deterministic and high latency; hence LLM operates as an out-of-loop orchestrator/advisor only. |
| 2026-10-04 | CAGE Challenge 4 for multi-agent experiments | CC4 provides five Blue agents, randomized enterprise topologies, mission phases, and native multi-agent wrappers; the old CAGE 2 path could not support an honest multi-agent benchmark. |
| 2026-10-04 | Keep the toy environment only for sanity tests | `ToySubnetEnv` helps validate MAPPO mechanics, but it is not CybORG and its returns are excluded from all reports and dashboard results. |
| 2026-10-04 | Use SB3 MaskablePPO for IPPO | CC4 provides a verified `TrainingSB3.py` action-masking path; it is more reliable than Ray/RLlib on native Windows. |
| 2026-10-04 | Validate baselines at 30 episodes and 100 steps during development | This reduces iteration time while preserving three-seed comparisons; final 500-step evaluation is deferred until the development results are reviewed. |
| 2026-10-04 | MAPPO selected over Independent DQN & QMIX for Phase 2 | Grounded in Nguyen & Reddi (IEEE TNNLS 2023) survey: CTDE solves multi-agent non-stationarity while decentralized actors respect partial observability in Dec-POMDP. |
| 2026-10-07 | Host state logged as red_session / no_red_session from get_agent_state | Uses documented CC4 debugging calls; it does not distinguish user from root privilege. |
| 2026-10-08 | Batch 4 results: IPPO -494, MAPPO -547, Sleep -534 on held-out seeds (20,000-step budget, 30 evaluation episodes, 3 seeds) | Learned policies are not distinguishable from Sleep at this budget; reported as-is. Longer training or 100 evaluation episodes would be needed to separate them. |
| 2026-10-08 | IPPO via sb3-contrib MaskablePPO following CC4's TrainingSB3 chain; MAPPO generalized in agents/mappo.py; both use identical episode budgets, held-out evaluation seeds (9,000,000+) and reward scaling 0.1 | Follows the Batch 4 plan; comparable budgets. Toy MAPPO moved to agents/mappo_toy.py. |