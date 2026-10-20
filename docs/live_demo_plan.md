# Live CC4 Demonstration Plan

## Purpose

This plan defines how to add an **in-process live execution mode** to the
Streamlit dashboard. The live mode will execute a fresh CC4 episode in memory
and render the environment as it progresses, instead of reading only from
saved replay files.

The goal is a reliable, explainable demonstration of the real simulator:

```text
seeded CC4 scenario
        ↓
Red policy advances
        ↓
Blue policy selects actions
        ↓
CC4 environment steps
        ↓
dashboard renders the current state
```

The live mode is a simulation demonstration, not a real network attack. The UI
must label scripted attack scenarios as controlled simulated attacks.

## Scope decision

### Included in this plan

- In-process CC4 environment execution
- Reproducible seed selection
- A controlled scripted Red scenario
- Live Sleep, IPPO, and MAPPO execution where their verified runtime paths and
  checkpoints are available
- Same-seed, same-scenario comparisons
- Live zone-level network visualization
- Current-step Blue actions and Red activity
- Live reward and operational metrics
- Pause, resume, stop, and restart behavior
- Saved replay fallback
- Optional saving of a completed live run as a normal episode artifact

### Explicitly excluded from this phase

- LLM approval, rejection, replacement, or modification of Blue actions
- Any claim that advisory LLM output changes reward or behavior
- Mixing live-demo results with the existing held-out evaluation tables
- Replacing the current saved replay and evidence workflow
- Inventing host states, rewards, actions, latency, or cost values

The existing LLM remains advisory-only. Live LLM incident reporting can be
added as a later phase, but it must not intervene in action selection in this
plan.

## Demo modes

The dashboard should expose three clearly separated modes:

### 1. Live Demo

Creates a fresh CC4 environment and advances it one step at a time. This is
the primary presentation mode.

### 2. Same-Seed Comparison

Creates independent environment instances using the same scenario seed and
controlled Red strategy. Each instance runs one Blue policy:

- Sleep
- IPPO
- MAPPO

The comparison must use separate environment instances. A run must never share
mutable state with another run.

### 3. Saved Replay Fallback

Loads an existing artifact from `results/episodes/` when live execution is
unavailable or when the presenter wants an exact recorded result. The UI must
show that it is replay mode and must not silently switch from live mode after
an error.

## Controlled Red scenario

The live demonstration needs a predictable scenario that creates visible
state changes without pretending to model a real intrusion.

### Scenario controls

- Scenario seed
- Red strategy
- Attack start step
- Episode length
- Optional pause between rendered steps

### Initial Red strategies

1. **Native CC4 Red policy**
   - Uses the verified CC4 Red implementation.
   - Best for demonstrating the actual benchmark behavior.

2. **Controlled scripted chain**
   - A small deterministic Red policy used only for demonstration.
   - The default chain is:

     ```text
     discover → exploit → privilege escalation → impact
     ```

   - The chain begins at a configured step.
   - It selects targets from the current CC4 observation and valid action
     space; it must not assume fixed host IDs or a fixed topology.
   - Invalid or unavailable actions must be handled through the environment's
     documented action and mask interfaces.

The UI should label this option:

> Controlled simulated attack — deterministic demo scenario.

The scripted policy is a demonstration tool, not an evaluation baseline. Its
results must not be added to the benchmark comparison tables unless a separate
experiment formally defines and evaluates it.

## Blue policies

The live runner should use the same policy interfaces and action semantics as
the existing experiment code:

- **Sleep**: verified no-op baseline
- **IPPO**: verified checkpoint-aware policy path
- **MAPPO**: verified checkpoint-aware multi-agent policy path

The live dashboard must fail explicitly when a selected checkpoint or required
runtime dependency is unavailable. It must not substitute a saved replay or a
different policy without showing that choice to the presenter.

Each live frame should record:

- Current step
- Blue actions by agent
- Red activity
- Per-agent and team reward
- Cumulative reward
- Host state summary
- Zone summary
- Mission phase
- Termination status
- Seed, policy, Red strategy, and checkpoint provenance

## Live comparison design

The strongest presentation is a matched scenario comparison:

```text
Scenario seed: selected seed
Red strategy: selected strategy
Blue runs:
  1. Sleep
  2. IPPO
  3. MAPPO
```

Every run starts from an independently reset environment configured with the
same scenario seed and Red strategy. The dashboard advances the runs to a
shared comparison step and displays:

### Per-policy metrics

- Current step
- Cumulative team reward
- Compromised-host count
- Restored-host count
- Active alert count
- Impacted-host count, if exposed by CC4
- Operational-zone downtime
- Current mission phase
- Episode completion state

### Comparison charts

- Cumulative reward by step
- Compromised hosts by step
- Restored hosts by step
- Operational-zone downtime by step

The dashboard should distinguish:

- **Live demonstration metrics**: generated during the current run
- **Saved evaluation metrics**: loaded from reproducible result tables

Live curves are for explaining behavior. They must not be presented as the
held-out statistical result unless the run is deliberately saved and evaluated
through the normal experiment pipeline.

## Live network visualization

The default graph should remain zone-level so it is readable during a
presentation.

### Zone graph

Each node represents one CC4 zone and displays or exposes:

- Number of hosts in the zone
- Number of hosts with Red sessions
- Number of restored hosts
- Current alert or impact state

Suggested state colors:

- Neutral: no active incident
- Red: one or more compromised hosts
- Amber: alert or suspicious activity
- Green: restoration activity or recovered state

### Zone expansion

Clicking or selecting a zone should reveal its current hosts and show:

- Host identifier from the live observation
- Red-session state
- Relevant alert state
- Most recent Blue action affecting the host

The graph must be rebuilt from the current environment observation. It must not
use a fixed host list or hardcoded “94-host” topology.

## Current-step activity panel

The live view should show only the current step by default.

### Blue actions

Display compact rows such as:

```text
Blue-1   REMOVE   host-17
Blue-2   RESTORE  host-04
Blue-3   DECOY    host-22
```

The action column must remain visible when agent names or action parameters
are long. Historical activity can be available in a collapsible event log,
but it must not replace the current-step view.

### Red activity

Display the current Red event, target, and phase from the environment
observation. If CC4 does not expose a field, show an explicit unavailable
state rather than inferring one.

## Execution and state model

The first implementation should use a small live-run controller stored in
Streamlit session state. Conceptually:

```text
LiveRun
  mode
  seed
  red_strategy
  blue_policy
  checkpoint
  environment
  current_step
  episode_done
  current_observation
  current_actions
  current_reward
  cumulative_reward
  frame_history
  error_state
```

The controller should expose operations equivalent to:

- Create/reset run
- Execute one environment step
- Advance all comparison runs to a target step
- Pause
- Resume
- Stop
- Restart
- Save completed run

The implementation should execute one environment step per controlled
Streamlit rerun rather than relying on one long blocking loop. This keeps the
current frame visible and makes pause/stop behavior reliable.

## Playback controls

Required controls:

- Start
- Pause
- Resume
- Stop
- Restart
- Step forward
- Playback speed

Playback speed should control rendering cadence only. It must not alter the
simulator transition sequence or skip environment steps without recording them.

If an external LLM call is later added, it must not block the environment loop
in this phase. Live execution must remain functional without an LLM service.

## Failure handling and fallback

### Environment failure

Preserve the last valid frame and show the error. Offer an explicit button:

> Switch to saved replay

Do not silently change modes.

### Missing checkpoint

Show the missing checkpoint path and policy name. Offer Sleep or saved replay
only as an explicit user choice.

### Invalid scripted action

Reject the action through a visible controlled error or choose another valid
action using the verified CC4 action-mask mechanism. Never inject an action
that is not valid for the current observation.

### Partial run

Allow the presenter to inspect the last valid frame and save it as a partial
diagnostic artifact only if the artifact is marked incomplete.

### Restart

Reset the environment from the selected seed and clear all prior frame state.
The UI must show the new run identifier or reset timestamp so the presenter
knows the scenario restarted.

## Provenance and saved live runs

When requested, a completed live run should be saved using the same schema as
the existing episode files, with additional live-demo metadata:

- `source: live_demo`
- Environment and wrapper versions
- Seed
- Red strategy
- Blue policy
- Checkpoint path and hash, when applicable
- Git commit
- Completion status
- Scenario configuration

Saved live runs must be distinguishable from held-out evaluation artifacts.
They should not automatically change the benchmark tables or statistical
results.

## Implementation phases

### Phase 1: Live Sleep smoke path

- Add a live-run controller
- Reset a real CC4 environment
- Execute one step at a time
- Render current reward, phase, Red activity, and zone graph
- Add start, step, stop, and restart

**Acceptance:** a fresh seeded Sleep episode advances in the dashboard without
reading an episode replay file.

### Phase 2: Controlled scripted Red

- Implement the deterministic demo Red policy
- Validate actions against the current CC4 action space
- Add attack start step and strategy controls
- Label the scenario as simulated

**Acceptance:** the same seed and configuration produce the same visible attack
sequence, subject to verified CC4 determinism.

### Phase 3: Playback quality

- Add pause and resume
- Add speed control
- Add step-forward control
- Preserve the last valid frame
- Prevent long blocking loops

**Acceptance:** the presenter can pause, resume, stop, and restart without
losing or corrupting the current live state.

### Phase 4: Live trained policies

- Connect IPPO checkpoint loading
- Connect MAPPO checkpoint loading
- Reuse existing policy and mask handling
- Display checkpoint provenance
- Compare live actions against Sleep

**Acceptance:** each selected policy runs through the same live controller and
does not silently fall back to another policy.

### Phase 5: Same-seed comparison

- Create independent runs for Sleep, IPPO, and MAPPO
- Synchronize them at a selected step
- Add cumulative reward and compromise/downtime charts
- Add side-by-side zone graphs

**Acceptance:** all policies use the same seed and Red scenario while retaining
independent environment state.

### Phase 6: Replay fallback and saving

- Add explicit replay fallback toggle
- Preserve clear live/replay badges
- Save completed live runs using the episode schema
- Add incomplete-run handling

**Acceptance:** the demo remains usable when live dependencies fail, without
misrepresenting replay data as live execution.

## Validation plan

Before presenting the live mode:

1. Run a short Sleep episode with a fixed seed.
2. Repeat it and compare actions, rewards, and termination.
3. Run the scripted Red scenario with the same seed twice.
4. Confirm the graph uses the current topology and zone membership.
5. Confirm the current-step panel never displays future actions.
6. Run Sleep and IPPO from the same scenario seed in independent environments.
7. Stop and restart during playback.
8. Simulate a missing checkpoint and confirm the error is visible.
9. Simulate a live-run failure and confirm replay requires an explicit choice.
10. Save a completed live run and verify it can be loaded by the existing
    replay viewer.

## Definition of done

The live demo is ready when:

- A real CC4 episode can be launched from the dashboard without pre-existing
  replay files.
- The selected seed and Red strategy are visible.
- The zone graph, current actions, Red activity, reward, and mission phase
  update from the live environment.
- Sleep and at least one trained policy can run through the same controller, or
  the unavailable policy is clearly reported.
- Same-seed comparisons use independent environments.
- Pause, resume, stop, restart, and step-forward work predictably.
- Saved replay remains available as an explicit fallback.
- Live-demo outputs are not mixed into benchmark evidence automatically.
- No live result contains fabricated topology, reward, action, latency, or cost
  data.
- No LLM action intervention is implied or implemented.

## Later plan: LLM intervention

LLM intervention is intentionally deferred. A future plan may define a
separate **RL + LLM gating** mode in which the LLM approves, rejects, or
modifies proposed actions. That work requires its own action schema, timeout
fallback, reproducibility rules, latency budget, safety constraints, and
evaluation tables. Its results must not be mixed with this advisory-only live
demo or with the current RL evaluation.
