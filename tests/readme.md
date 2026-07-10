# 🧪 Tests Module (`tests/`)

The **Tests Module** is the experiment orchestration layer of the project. It determines:

* **What** to compare
* **Which scenarios** to evaluate
* **How many seeds** to run
* **How metrics are aggregated**
* **How results are visualized**
* **How trained policies are diagnosed**

Unlike `evaluation/`, which only provides reusable evaluation mechanics, this module designs and executes the actual experiments.

---

## 📖 Table of Contents

* [Purpose](#-purpose)
* [Role in the Project Pipeline](#-role-in-the-project-pipeline)
* [Module Structure](#-module-structure)
* [Experiment Workflow](#-experiment-workflow)
* [File Descriptions](#-file-descriptions)
* [CLI Commands](#-cli-commands)
* [Design Philosophy](#-design-philosophy)
* [Summary](#-summary)

---

# 🎯 Purpose

The `tests/` module is responsible for:

✅ Plotting training progress

✅ Running multi-scenario experiments

✅ Running multi-seed evaluations

✅ Aggregating metrics

✅ Producing dashboards and tables

✅ Visualizing policy behavior

✅ Diagnosing reward-function issues

This module represents the **final stage** of the RL pipeline, transforming trained models into reproducible experimental results and visual analyses.

---

# 🏗️ Role in the Project Pipeline

```text
agents/train.py
        │
        ├── models/
        └── logs/
               │
               ▼
tests/plot_training_curve.py
               │
               ▼
tests/run_experiments.py
               │
               ▼
tests/results_summary.py
               │
               ▼
tests/run_single_episode.py
               │
               ▼
tests/diagnose_reward.py
```

The `tests/` module consumes:

* Trained models (`models/`)
* Training logs (`logs/`)
* Evaluation utilities (`evaluation/`)

and produces all final metrics, plots, and comparisons.

---

# 📂 Module Structure

| File                     | Responsibility                   |
| ------------------------ | -------------------------------- |
| `scenarios.py`           | Defines evaluation scenarios     |
| `metrics.py`             | Computes and aggregates metrics  |
| `compare.py`             | Runs multi-agent comparisons     |
| `plot_dashboard.py`      | Creates dashboard visualizations |
| `plot_training_curve.py` | Visualizes learning progress     |
| `results_summary.py`     | Produces formatted result tables |
| `diagnose_reward.py`     | Investigates reward calibration  |
| `run_experiments.py`     | Full experiment runner           |
| `run_single_episode.py`  | Single-episode visualization     |

---

# 🔄 Experiment Workflow

```text
Load Models
      │
      ▼
Select Scenarios
      │
      ▼
Run Agents
      │
      ▼
Collect Metrics
      │
      ▼
Aggregate Results
      │
      ▼
Generate Plots & Tables
```

Every experiment in the project follows this workflow.

---

# 📁 File Descriptions

---

# 🌊 `scenarios.py`

Provides the single source of truth for workload scenarios.

```python
SCENARIOS = [
    "spike",
    "linear",
    "noisy",
    "periodic"
]
```

---

## Why Centralize Scenario Names?

Without this file:

❌ Scenario names become hardcoded.

❌ Adding new scenarios requires changes across multiple files.

❌ Typos become difficult to detect.

Centralizing scenarios ensures consistency throughout the project.

---

# 📊 `metrics.py`

Responsible for computing and aggregating evaluation metrics.

Provides two functions:

---

## `compute_metrics(data)`

Input:

```python
list[info_dict]
```

Output:

```python
{
    "cost": ...,
    "error": ...,
    "response": ...,
    "utilization": ...
}
```

---

## Metric Definitions

| Metric        | Aggregation |
| ------------- | ----------- |
| Cost          | Summed      |
| Error Rate    | Mean        |
| Response Time | Mean        |
| Utilization   | Mean        |

---

## `aggregate(results)`

Input:

```python
[
    metric_dict_seed1,
    metric_dict_seed2,
    ...
]
```

Output:

```python
{
    "mean": ...,
    "std": ...
}
```

Used to compute:

* Dashboard values
* Error bars
* Summary tables

---

# ⚔️ `compare.py`

The central comparison engine.

Provides:

```python
run_comparison()
```

---

## Responsibilities

1. Load latest PPO model
2. Load latest DQN model
3. Create baseline agent
4. Run all scenarios
5. Run all seeds
6. Aggregate metrics
7. Return results

---

## Data Flow

```text
PPO
DQN
Baseline
      │
      ▼
All Scenarios
      │
      ▼
All Seeds
      │
      ▼
Aggregated Results
```

---

## Return Structure

```python
results[
    scenario
][
    agent
][
    metric
]
```

Example:

```python
results["spike"]["ppo"]["cost"]
```

---

# 📈 `plot_dashboard.py`

Responsible for generating the project's primary evaluation figure.

Produces:

```text
2 × 2 Metrics Dashboard
```

---

## Dashboard Layout

| Subplot      | Metric        |
| ------------ | ------------- |
| Top Left     | Cost          |
| Top Right    | SLA Violation |
| Bottom Left  | Latency       |
| Bottom Right | Utilization   |

---

## Plot Structure

```text
Scenario
     │
     ├── PPO
     ├── DQN
     └── Baseline
```

Each bar displays:

```text
Mean ± Standard Deviation
```

computed across evaluation seeds.

---

# 📈 `plot_training_curve.py`

Visualizes learning progression during training.

Reads monitor logs from:

```text
logs/
```

and generates:

```text
Episode Reward
        vs
Cumulative Timesteps
```

for both PPO and DQN.

---

## Plot Contents

For each algorithm:

* Raw episode rewards
* Smoothed rolling average

---

## Why Plot Learning Curves?

Training curves reveal:

✅ Convergence

✅ Instability

✅ Reward plateaus

✅ Exploration behavior

✅ Relative learning speed

---

## Smoothing Window

```bash
python -m tests.plot_training_curve --window 10
python -m tests.plot_training_curve --window 50
```

---

# 📋 `results_summary.py`

Produces formatted terminal tables of evaluation results.

Runs the complete comparison pipeline and prints:

```text
Scenario
    ↓
Agent
    ↓
Metric ± Std
```

---

## Metrics Displayed

* Cost
* SLA Violation
* Latency
* Utilization

---

## Additional Analysis

After the tables, the script prints:

```text
Cost vs Baseline Highlights
```

showing where PPO or DQN outperform the threshold policy.

---

# 🔍 `diagnose_reward.py`

A diagnostic tool used during reward tuning.

Its purpose is to determine whether poor policy behavior originates from:

* Insufficient training
* Poor exploration
* Reward miscalibration

---

## Method

Runs:

```text
Learned Policy
```

versus

```text
Forced Scale-Up Policy
```

on the identical workload.

---

## Metrics Compared

* Mean reward
* Mean utilization
* Mean instance count

---

## Interpretation

### Forced Policy Performs Better

```text
Training Problem
```

Possible causes:

* Insufficient timesteps
* Exploration issues
* Hyperparameter problems

---

### Forced Policy Performs Worse

```text
Reward Calibration Problem
```

Possible causes:

* Cost penalties too large
* SLA penalties too weak
* Idle penalties improperly balanced

---

# 🚀 CLI Commands

---

# 📈 Training Curve Plot

```bash
python -m tests.plot_training_curve
```

Reads:

```text
logs/
```

and displays a two-panel learning progress plot for PPO and DQN.

---

## Custom Smoothing Window

```bash
python -m tests.plot_training_curve --window 10
python -m tests.plot_training_curve --window 50
```

---

# 📊 Full Multi-Scenario Comparison

```bash
python -m tests.run_experiments
```

---

## What It Does

1. Load latest PPO model
2. Load latest DQN model
3. Evaluate:

```text
PPO
DQN
Baseline
```

4. Run all scenarios
5. Run multiple seeds
6. Aggregate metrics
7. Display dashboard

---

## Default Scenarios

```text
spike
linear
noisy
periodic
```

---

## Custom Scenarios

```bash
python -m tests.run_experiments \
    --scenarios spike noisy \
    --seeds 1 2 3
```

---

## Save Results

```bash
python -m tests.run_experiments --save
```

Creates:

```text
results.json
```

---

## Custom Episode Length

```bash
python -m tests.run_experiments --steps 500
```

---

# 📉 Single-Episode Visualization

```bash
python -m tests.run_single_episode
```

---

## What It Does

1. Load latest models
2. Create shared workload
3. Run:

```text
PPO
DQN
Baseline
```

4. Print metrics
5. Display time-series plot

---

## Custom Scenario

```bash
python -m tests.run_single_episode \
    --scenario spike \
    --steps 500 \
    --seed 0
```

```bash
python -m tests.run_single_episode \
    --scenario periodic \
    --seed 42
```

---

# 📋 Results Summary Table

```bash
python -m tests.results_summary
```

Runs the full comparison and prints:

* Mean ± Standard Deviation tables
* Cost highlights versus baseline

---

## Custom Options

```bash
python -m tests.results_summary --save
```

```bash
python -m tests.results_summary \
    --seeds 1 7 23 42 100
```

```bash
python -m tests.results_summary \
    --scenarios spike linear
```

---

# 🔬 Reward Diagnostic

```bash
python -m tests.diagnose_reward
```

Runs:

```text
Learned Policy
vs
Forced Scale-Up Policy
```

and prints:

* Mean reward
* Mean utilization
* Mean instance count
* Plain-language verdict

---

## Example Output

```text
Policy Reward:  -0.42
Forced Reward: -0.28

Verdict:
Training problem.
```

or

```text
Policy Reward:  -0.42
Forced Reward: -0.55

Verdict:
Reward calibration problem.
```

---

# 🏗️ Design Philosophy

The project intentionally separates:

```text
Evaluation Mechanics
```

from:

```text
Experiment Design
```

This separation provides:

✅ Reusability

✅ Cleaner code

✅ Easier debugging

✅ Reproducibility

✅ Independent extension of testing logic

---

# 🔑 Summary

The `tests/` module represents the experiment orchestration layer of the project. It is responsible for:

* Designing evaluation experiments
* Plotting training progression
* Running multi-scenario comparisons
* Running multi-seed benchmarks
* Aggregating metrics
* Producing dashboards and tables
* Visualizing policy behavior
* Diagnosing reward-function issues

Together with `evaluation/`, it forms the complete benchmarking framework used to validate RL-based cloud auto-scaling policies.