# Real CC4 Migration and Experiment Plan

This document records the ordered implementation batches for moving the project
from fabricated CAGE 2/toy-environment demonstrations to reproducible
CAGE Challenge 4 (CC4) experiments.

## Core evidence rule

Every reward, action, host state, latency, cost, dashboard value, and report
number must come from a real CC4 rollout or a saved training run. Hand-written
simulation logic and invented bonuses are never valid evidence.

## Batch 0: Stop fabricated evidence

- Delete fabricated JSON episode logs.
- Move the hand-written evaluator to `legacy/evaluate_fake_sim.py` and mark it
  as unusable for results.
- Rename the toy MAPPO environment to `envs/toy_env.py::ToySubnetEnv`.
- Keep the toy environment only for implementation sanity tests; never report
  its returns as cyber-defense performance.
- Remove any LLM reward bonus.
- Ensure CSV evidence is trackable and ignore only raw/temporary artifacts.
- Update dashboard and plan references so the fake evaluator is not an active
  data path.

Acceptance: no active Python reference to the fake evaluator; no fabricated
episode files remain under `results/`.

## Batch 1: Run the real CC4 environment

- Clone and install CC4 from its upstream repository.
- Use a compatible Python runtime without mutating the project environment
  unexpectedly.
- Build `envs/cc4_env.py` from the verified CC4 example APIs:
  `EnterpriseScenarioGenerator`, `CybORG`, and `BlueFlatWrapper`.
- Use `SleepAgent`, `EnterpriseGreenAgent`, and `FiniteStateRedAgent`.
- Add a smoke test that prints real red activity, Blue actions, and rewards.
- Verify five-agent operation, action masks, non-positive penalties, and
  seed-dependent randomized topology.

Acceptance: five real Sleep-team episodes complete successfully.

## Batch 2: Establish real baselines

- Implement Sleep, action-mask-aware Random, and a state-observation-driven
  RuleBased policy.
- The rule policy should prioritize real CC4 actions such as Analyse, Remove,
  and Restore when observed alerts justify them.
- Run 100 episodes per baseline over three seeds.
- Aggregate team reward as the mean of the per-agent CybORG rewards at each
  step, matching the CC4 evaluator, and include an across-seed summary row.
- Save mean and standard deviation to `results/tables/baselines.csv`.
- Re-running with the same seed must reproduce the same result.

Acceptance: all policies run against the real CC4 environment and the rule
policy is compared honestly with Sleep and Random. Do not force a desired
ranking if the simulator does not produce it.

## Batch 3: Save real rollout logs

- Add `experiments/rollout.py`.
- Save one JSON file per episode under `results/episodes/`.
- Record only environment-provided values: step, red activity, Blue actions,
  host state, reward, cumulative reward, and termination information.
- Add provenance metadata: environment, agent, Red policy, seed, Git commit,
  source, and host count.
- Remove hardcoded host and subnet lists from result consumers.

Acceptance: replaying a saved episode agrees with the environment reward sum.

## Batch 4: Train IPPO and MAPPO honestly

- Add a real IPPO entry point using the CC4 `TrainingSB3.py` path and
  `sb3-contrib` `MaskablePPO`. Do not use the Ray/RLlib path on native Windows.
- Save per-episode training returns and checkpoints.
- Generalize MAPPO to five agents, CC4 observation dimensions, centralized
  joint state, and per-agent action masks.
- Train and evaluate in exactly the same environment and observation format.
- Use three seeds and held-out evaluation seeds.
- Report mean and standard deviation, environment steps, and wall-clock time.

Acceptance: IPPO and MAPPO use comparable budgets and are evaluated against the
real baseline table.

## Batch 5: Make the LLM layer measurable

- Add a CC4 state summarizer for alerts, zone counts, mission phase, and recent
  Blue actions.
- Validate all model output through `llm/schema.py`.
- Add an explicit output source: `gemini`, `cache`, or `fallback`.
- Remove silent exception swallowing and fake latency/reward values.
- Measure API wall time and compute cost from token usage and configured prices.
- Include model name and prompt version in a SHA-256 cache key.
- Keep API keys out of logs and disk.
- Evaluate advisory mode and, if budget permits, intervening mode.
- Compare RL-only, RL plus advisory, and RL plus intervention on identical seeds.

Acceptance: fallback and cached results are visibly labelled and excluded from
model-performance claims.

## Batch 6: Make the dashboard evidence-driven

- Read only saved files under `results/`.
- Remove hardcoded seeds, agent lists, host counts, returns, and canned outputs.
- Add a CC4 data-source badge with agent, seed, and Git commit.
- Show “No results yet” when real files are absent.
- Display measured LLM source, latency, and cost.
- Add real episode replay, topology/zone grouping, red-target highlighting,
  event logs, reward curves, and training mean +/- standard deviation.

Acceptance: deleting a result file removes its dashboard content rather than
revealing placeholder data.

## Batch 7: Documentation and repository hygiene

- Update README installation and experiment commands for CC4.
- Record dated design decisions in `docs/decisions.md`.
- Mark toy-only configs clearly and fail loudly on missing configuration.
- Move toy-trained checkpoints out of the active model path.
- Update project context and report tables for CC4.
- Document compute limits, simulator-only scope, LLM free-tier limits, and
  shorter training budgets.

Acceptance: a new user can reproduce each saved table from documented commands.

## Execution policy

Batches are completed in order. Each batch must pass its acceptance check and
be committed before the next batch starts. If CC4 cannot run, use a clearly
documented real-environment fallback; never replace it with fabricated data.
