# 🤖 Agents Module (`agents/`)

The **Agents Module** contains the reinforcement learning algorithms responsible for learning intelligent cloud auto-scaling policies. Two algorithms are implemented:

* **PPO (Proximal Policy Optimization)** – an on-policy policy-gradient method.
* **DQN (Deep Q-Network)** – an off-policy value-based method.

Both agents are trained against the same `CloudEnv` Gymnasium environment, enabling a fair and reproducible comparison of their scaling behavior and performance.

---

## 📖 Table of Contents

* [Module Structure](#-module-structure)
* [PPO Agent](#-ppo--proximal-policy-optimisation)
* [DQN Agent](#-dqn--deep-q-network)
* [Mixed-Scenario Training](#-mixed-scenario-training)
* [Model Naming Convention](#-model-naming-convention)
* [CLI Training Interface](#-cli--trainpy)

---

# 📂 Module Structure

| File           | Responsibility                          |
| -------------- | --------------------------------------- |
| `ppo_agent.py` | Configures and trains a PPO model       |
| `dqn_agent.py` | Configures and trains a DQN model       |
| `train.py`     | CLI entrypoint that dispatches training |

---

## Training Flow

```text
CloudEnv
    │
    ▼
PPO / DQN Agent
    │
    ▼
Stable-Baselines3
    │
    ▼
Trained Model (.zip)
```

The `train.py` file acts as the only executable entrypoint for this module.

---

# 🧠 PPO — Proximal Policy Optimisation

**File:** `ppo_agent.py`

PPO is an **on-policy policy-gradient algorithm** that learns a direct mapping:

```text
State → Action Probabilities
```

The policy is updated using experience collected from the current policy itself.

A clipping mechanism constrains policy updates, preventing excessively large parameter changes that can destabilize learning.

---

## Why PPO?

PPO is particularly suitable for this project because:

✅ The reward function has multiple competing objectives.

✅ Workload patterns are stochastic.

✅ Conservative updates improve training stability.

✅ Entropy regularization encourages exploration.

---

## PPO Training Pipeline

```text
Environment
      │
      ▼
Collect Rollouts
      │
      ▼
Estimate Advantages
      │
      ▼
Policy Update
      │
      ▼
Repeat
```

---

## PPO Hyperparameters

| Parameter       | Value                            | Description                    |
| --------------- | -------------------------------- | ------------------------------ |
| `policy`        | `MlpPolicy`                      | Fully connected policy network |
| `learning_rate` | `3e-4`                           | Gradient descent step size     |
| `n_steps`       | `min(2048, max(256, steps//20))` | Rollout buffer size            |
| `batch_size`    | `64`                             | Mini-batch size                |
| `gamma`         | `0.99`                           | Discount factor                |
| `gae_lambda`    | `0.95`                           | Advantage estimation smoothing |
| `clip_range`    | `0.2`                            | Maximum policy update size     |
| `ent_coef`      | `0.05`                           | Exploration bonus coefficient  |

---

## Hyperparameter Rationale

### Learning Rate

```python
learning_rate = 3e-4
```

Provides stable convergence without excessively slow learning.

---

### Rollout Buffer

```python
n_steps = min(
    2048,
    max(256, steps//20)
)
```

Ensures:

* At least 20 policy updates.
* Rollout size scales with training budget.

---

### Entropy Coefficient

```python
ent_coef = 0.05
```

Higher than the SB3 default (`0.01`) to encourage exploration when training budgets are relatively short.

---

## PPO Training Process

### Step 1

Wrap environment:

```python
Monitor(env)
```

---

### Step 2

Determine workload mode:

* Fixed workload
* Mixed-scenario workload

---

### Step 3

Compute adaptive rollout size.

---

### Step 4

Train:

```python
model.learn(
    total_timesteps
)
```

Stable-Baselines3 internally performs:

* Rollout collection
* Advantage estimation
* Gradient updates

---

### Step 5

Save model:

```text
models/ppo_{steps}_{timestamp}.zip
```

---

# ⚡ DQN — Deep Q-Network

**File:** `dqn_agent.py`

DQN is an **off-policy value-based algorithm**.

Instead of learning a policy directly, it learns:

```text
Q(state, action)
```

which estimates the expected cumulative reward for each action.

Actions are selected using:

```text
ε-greedy exploration
```

---

## Why DQN?

DQN is particularly well suited because:

✅ Action space is very small (`3` actions).

✅ Value estimation is straightforward.

✅ Replay buffers improve sample efficiency.

✅ Past experiences can be reused.

---

## DQN Training Pipeline

```text
Environment
      │
      ▼
Replay Buffer
      │
      ▼
Q-Network Updates
      │
      ▼
Target Network Sync
      │
      ▼
Repeat
```

---

## DQN Hyperparameters

| Parameter                | Value                            | Description                    |
| ------------------------ | -------------------------------- | ------------------------------ |
| `policy`                 | `MlpPolicy`                      | Fully connected Q-network      |
| `learning_rate`          | `5e-4`                           | Gradient step size             |
| `buffer_size`            | `min(100000, max(2000, steps))`  | Replay buffer capacity         |
| `learning_starts`        | `min(5000, max(200, steps//20))` | Initial exploration period     |
| `batch_size`             | `32`                             | Training batch size            |
| `gamma`                  | `0.99`                           | Discount factor                |
| `train_freq`             | `4`                              | Update frequency               |
| `target_update_interval` | `500`                            | Target network synchronization |
| `exploration_fraction`   | `0.5`                            | Exploration duration           |
| `exploration_final_eps`  | `0.05`                           | Minimum exploration rate       |

---

## Hyperparameter Rationale

### Replay Buffer

```python
buffer_size = min(
    100000,
    max(2000, steps)
)
```

Ensures:

* Efficient memory usage.
* Buffer remains well utilized regardless of training budget.

---

### Learning Starts

```python
learning_starts = min(
    5000,
    max(200, steps//20)
)
```

Prevents learning from an insufficient amount of experience.

---

### Exploration Fraction

```python
exploration_fraction = 0.5
```

Significantly larger than the SB3 default (`0.1`).

Short training budgets require more relative exploration to discover useful policies.

---

## DQN Training Process

### Step 1

Wrap environment:

```python
Monitor(env)
```

---

### Step 2

Compute adaptive parameters:

* `buffer_size`
* `learning_starts`

---

### Step 3

Populate replay buffer via random exploration.

---

### Step 4

Update Q-network:

```python
train_freq = 4
```

---

### Step 5

Synchronize target network:

```python
target_update_interval = 500
```

---

### Step 6

Save model:

```text
models/dqn_{steps}_{timestamp}.zip
```

---

# 🌊 Mixed-Scenario Training

When:

```bash
python -m agents.train --algo both
```

both agents are trained on a weighted mixture of workload scenarios.

---

## Scenario Distribution

| Scenario | Probability | Description                              |
| -------- | ----------- | ---------------------------------------- |
| Default  | 40%         | Sinusoidal traffic with bursts and noise |
| Spike    | 15%         | Sudden jump to λ=500                     |
| Linear   | 15%         | Steadily increasing traffic              |
| Noisy    | 15%         | Pure random demand                       |
| Periodic | 15%         | Regular oscillatory traffic              |

---

## Scenario Selection Process

```text
Episode Reset
       │
       ▼
Random Scenario Draw
       │
       ▼
Generate Workload
       │
       ▼
Train Agent
```

---

## Why Mixed Training?

Training on a single workload causes agents to memorize traffic patterns.

Mixed-scenario training encourages:

✅ Robustness

✅ Generalization

✅ Adaptability

✅ Better evaluation performance

---

## Reproducibility

Both agents use:

```python
seed = 42
```

which guarantees:

* Identical scenario sequences
* Reproducible experiments
* Fair comparisons

---

# 💾 Model Naming Convention

Saved checkpoints follow:

```text
models/{agent}_{total_timesteps}_{timestamp}.zip
```

Examples:

```text
models/ppo_200000_20260621_143022.zip
models/dqn_200000_20260621_143041.zip
```

Benefits:

* Encodes training budget.
* Encodes training time.
* Easy checkpoint identification.
* No accidental overwrites.

---

# Training Logs
 
Every training run writes a Monitor log CSV to `logs/`, mirroring the model filename:
 
```
logs/ppo_200000_20260621_143022.monitor.csv
logs/dqn_200000_20260621_143041.monitor.csv
```
 
Each CSV contains one row per completed episode recording total episode reward (`r`), episode length in steps (`l`), and wall-clock time elapsed (`t`). These logs are read by `tests/plot_training_curve.py` to produce the learning progress plot showing how each agent's policy improved over the course of training.
 
---

# 🖥️ CLI — `train.py`

The `train.py` file is the sole executable entrypoint for training.

All commands should be run from the project root.

---

# 🚀 Commands

## Train PPO

```bash
python -m agents.train \
    --algo ppo \
    --steps 200000
```

Produces:

```text
models/ppo_200000_{timestamp}.zip
```

---

## Train DQN

```bash
python -m agents.train \
    --algo dqn \
    --steps 200000
```

Produces:

```text
models/dqn_200000_{timestamp}.zip
```

---

## Train Both Agents

```bash
python -m agents.train \
    --algo both \
    --steps 200000
```

Training order:

```text
DQN
 ↓
PPO
```

Each agent saves its own checkpoint.

---

## Custom Episode Length

```bash
python -m agents.train \
    --algo both \
    --steps 200000 \
    --episode-length 500
```

The `episode-length` parameter controls:

```text
Timesteps per workload sequence
```

Default:

```text
500 steps
```

Only applicable to:

```text
--algo both
```

---

# ⚙️ CLI Arguments

| Argument           | Default  | Description                               |
| ------------------ | -------- | ----------------------------------------- |
| `--algo`           | `ppo`    | Algorithm to train (`ppo`, `dqn`, `both`) |
| `--steps`          | `100000` | Total training timesteps                  |
| `--episode-length` | `500`    | Timesteps per episode                     |

---

# 🔑 Summary

The `agents/` module provides:

* PPO and DQN implementations
* Fair algorithm comparison on a shared environment
* Mixed-scenario training for generalization
* Adaptive hyperparameter scaling
* Reproducible experiments
* Timestamped checkpoint management
* Simple CLI-based training workflows

Together, these components form the learning engine of the RL-based cloud auto-scaling system, enabling intelligent policies that balance cost, latency, and SLA compliance under highly dynamic workloads.
