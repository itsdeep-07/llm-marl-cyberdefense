# Architecture & Design Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-04 | Separate LLM from RL loop | RL agents require deterministic, high-throughput action execution. LLMs are non-deterministic and high latency; hence LLM operates as an out-of-loop orchestrator/advisor only. |
| 2026-10-04 | CAGE Challenge 2 for single-agent baselines | Established CAGE Challenge 2 benchmark provides standardized B-line and Meander Red attacker agents. |
