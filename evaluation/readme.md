# 📊 Evaluation Module (`evaluation/`)

The **Evaluation Module** forms the mechanics layer of the comparison pipeline. It contains the reusable components responsible for:

* Running agents through the environment
* Recording performance metrics
* Producing rollout histories
* Generating visual comparisons

This module is intentionally independent of:

❌ Scenarios

❌ Seeds

❌ Dashboards

❌ Multi-run orchestration

Those responsibilities belong to the `tests/` module, which imports and reuses the components defined here.

---

## 📖 Table of Contents

* [Module Structure](#-module-structure)
* [Baseline Policy](#-baseline-policy--baselinepy)
* [Rollout Engine](#-rollout-loop--run_episodepy)
* [Time-Series Visualization](#-time-series-plot--plot_resultspy)
* [Design Philosophy](#-design-philosophy)
* [Summary](#-summary)

---

# 📂 Module Structure

| File              | Responsibility                           |
| ----------------- | ---------------------------------------- |
| `baseline.py`     | Rule-based threshold controller          |
| `run_episode.py`  | Universal rollout loop                   |
| `plot_results.py` | Single-episode time-series visualization |

---

## Evaluation Pipeline

```text id="5kns9k"
Agent
   │
   ▼
Cloud Environment
   │
   ▼
Rollout Loop
   │
   ▼
Metrics + History
   │
   ▼
Plots / Aggregation
```

The module does not know:

* Which scenarios are being tested.
* How many seeds are used.
* Which agent is being evaluated.

It simply provides the mechanics needed to perform evaluations consistently.

---

# ⚙️ Baseline Policy — `baseline.py`

The baseline is a **hand-written threshold controller** that makes scaling decisions entirely from current utilization.

Unlike PPO and DQN:

❌ No learning

❌ No experience replay

❌ No policy optimization

❌ No adaptation

It serves as the primary non-learned comparison point.

---

# 🎯 Scaling Logic

```python id="tfln0x"
if utilization > 0.8:
    action = 2      # scale up

elif utilization < 0.3:
    action = 0      # scale down

else:
    action = 1      # maintain
```

---

## Decision Boundaries

| Utilization | Action     |
| ----------- | ---------- |
| `> 0.8`     | Scale Up   |
| `0.3 – 0.8` | Maintain   |
| `< 0.3`     | Scale Down |

---

## State Dependency

The controller reads:

```python id="7nzyfy"
state[2]
```

which corresponds to:

```text id="v6ok0z"
U_cpu = λ / (N · μ)
```

clipped to:

```text id="k88ggf"
[0, 1]
```

---

# Why These Thresholds?

## Upper Threshold — 0.8

```text id="sg3u0u"
80% utilization
```

Provides approximately:

```text id="im8sye"
20% spare capacity
```

This reduces the probability of SLA violations when sudden workload spikes occur.

---

## Lower Threshold — 0.3

```text id="3wrhws"
30% utilization
```

Allows aggressive scale-down behavior during periods of low demand.

This minimizes operational cost when resources are underutilized.

---

# Fundamental Limitation

The thresholds are fixed.

The baseline cannot:

❌ Adapt to workload characteristics

❌ Learn from past experience

❌ Optimize long-term objectives

❌ Change behavior over time

These limitations are precisely what reinforcement learning is designed to address.

---

# Interfaces

`baseline.py` exposes two interfaces.

---

## Raw Policy Function

```python id="1id65t"
threshold_policy(state)
```

Returns:

```python id="ubn3nm"
action
```

Useful for custom evaluation loops.

---

## BaselineAgent Class

```python id="f9jshz"
BaselineAgent
```

Implements:

```python id="7eymft"
.predict(
    obs,
    deterministic=True
)
```

This matches the Stable-Baselines3 API.

---

## Why Mirror the SB3 Interface?

Because it allows:

```text id="04h3r3"
PPO
DQN
Baseline
```

to be evaluated using exactly the same code.

No special-case logic is required.

---

# 🔄 Rollout Loop — `run_episode.py`

This file is the **single source of truth** for evaluating agents.

Every experiment, benchmark, and dashboard ultimately relies on this rollout engine.

---

# `run_episode()`

```python id="2df2om"
run_episode(
    env,
    model=None,
    policy_fn=None,
    deterministic=True
)
```

Runs one complete episode:

```text id="tb1ydq"
env.reset()
      ↓
step()
      ↓
step()
      ↓
...
      ↓
truncated=True
```

---

## Accepted Inputs

### SB3-Compatible Model

```python id="c30zv8"
model.predict()
```

Examples:

* PPO
* DQN
* BaselineAgent

---

### Raw Policy Function

```python id="ucm8ri"
policy_fn(state)
```

Used when an SB3 interface is unnecessary.

---

# Return Value

```python id="9i2gva"
{
    "cost": float,
    "error": float,
    "response": float,
    "history": list
}
```

---

## Metric Definitions

| Key        | Meaning                 |
| ---------- | ----------------------- |
| `cost`     | Mean cost per timestep  |
| `error`    | Mean SLA violation rate |
| `response` | Mean response time      |
| `history`  | Full per-step metrics   |

---

# History Structure

Every timestep stores:

```python id="q2jql1"
{
    "requests": ...,
    "instances": ...,
    "utilization": ...,
    "response_time": ...,
    "error_rate": ...,
    "cost": ...,
    "action": ...,
    "reward": ...
}
```

---

## Why Keep Full History?

Different consumers require different levels of detail.

### Aggregated Metrics

```text id="4c4vsv"
cost
error
response
```

Used for tables and comparisons.

---

### Full Time Series

```text id="8nxyvo"
history
```

Used for:

* Visualization
* Debugging
* Per-step analysis

---

# 🌱 Multi-Seed Evaluation

```python id="4v7gc1"
run_episode_multi_seed(
    env_fn,
    model,
    policy_fn,
    seeds
)
```

---

## Process

```text id="5df6ui"
Seed 1
   │
   ▼
run_episode()

Seed 2
   │
   ▼
run_episode()

Seed 3
   │
   ▼
run_episode()
```

A fresh environment is constructed for each seed:

```python id="t7k7sq"
env = env_fn(seed)
```

---

## Return Value

```python id="5rt6sq"
[
    result_seed_1,
    result_seed_2,
    result_seed_3,
    ...
]
```

This output is consumed by:

```text id="drm6ce"
tests/compare.py
```

which computes:

* Mean
* Standard Deviation
* Final comparison tables

---

# 📈 Time-Series Plot — `plot_results.py`

## `plot_instances_vs_workload()`

```python id="y76frg"
plot_instances_vs_workload(
    ppo_result,
    dqn_result,
    baseline_result
)
```

Generates the primary visualization used throughout the project.

---

# Plot Design

The figure uses a **twin y-axis layout**.

---

## Left Axis

```text id="wkryln"
Workload λ
(requests/sec)
```

Plotted as:

```text id="6ux4ld"
Blue dashed line
```

---

## Right Axis

```text id="d5e55q"
Instance Count N
```

Plotted for:

* PPO (orange)
* DQN (green)
* Baseline (red)

---

# Why Twin Axes?

The scales are fundamentally different.

| Metric    | Typical Range |
| --------- | ------------- |
| Workload  | `0 – 365+`    |
| Instances | `1 – 20`      |

Plotting them on the same axis would flatten instance-count behavior and make scaling decisions impossible to interpret visually.

---

# Data Source

All plotted values come directly from:

```python id="ll8rlt"
result["history"]
```

Specifically:

```python id="1nkc5e"
step["requests"]
step["instances"]
```

---

## Why Not Use the State Vector?

The state contains:

```text id="b4ezlc"
λ / 1000
```

which is normalized.

Plotting normalized workload caused earlier versions of the dashboard to display a nearly flat workload line.

Using raw metrics ensures:

✅ Correct scaling

✅ Accurate visualization

✅ Easier debugging

---

# Primary Purpose of the Plot

This visualization answers a simple but important question:

> **Is the agent genuinely tracking workload changes?**

The figure immediately reveals:

* Over-provisioning
* Under-provisioning
* Delayed reactions
* Oscillatory behavior
* Differences between PPO, DQN, and Baseline

---

# 🏗️ Design Philosophy

The `evaluation/` module intentionally separates:

```text id="wwk9mz"
How evaluations work
```

from:

```text id="8n2bnh"
What experiments are run
```

This separation provides:

✅ Reusability

✅ Consistency

✅ Cleaner testing code

✅ Easier debugging

✅ Reproducible comparisons

---

# 🔑 Summary

The `evaluation/` module provides the reusable infrastructure needed to compare cloud auto-scaling policies by:

* Implementing a rule-based baseline controller
* Providing a universal rollout engine
* Supporting multi-seed experiments
* Recording detailed per-step histories
* Generating workload-versus-scaling visualizations
* Remaining completely independent of scenarios and dashboards

This separation makes evaluation logic simple, reusable, and consistent across every experiment in the project.
