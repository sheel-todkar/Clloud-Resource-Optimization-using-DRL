# Environment — `envs/cloud_env.py`

This module defines the custom Gymnasium environment used to train and evaluate reinforcement learning agents for cloud auto-scaling. It wraps the cloud simulator (`sim/cloud_sim.py`) and workload generator (`sim/workload.py`) in a standard OpenAI Gymnasium interface, making it compatible with any SB3-based RL algorithm without modification.

---

## Libraries

| Library | Role |
|---|---|
| `gymnasium` | Base `Env` class, action/observation space definitions |
| `gymnasium.spaces` | `Discrete` (action space), `Box` (observation space) |
| `numpy` | State vector construction, normalisation arithmetic |
| `sim.cloud_sim.CloudSimulator` | Computes system metrics from action + workload |

---

## Environment Parameters

| Parameter | Default | Description |
|---|---|---|
| `workload` | `None` | Fixed `WorkloadGenerator` instance — same sequence replayed every episode (used for evaluation and testing) |
| `workload_factory` | `None` | Callable returning a fresh `WorkloadGenerator` — called once per `reset()` so each training episode gets a different scenario |
| `mu` | `50` | Maximum requests per second each instance can serve |
| `cost_per_instance` | `1` | Cost unit charged per active instance per timestep |
| `N_min` | `1` | Minimum allowed instance count |
| `N_max` | `20` | Maximum allowed instance count |

Either `workload` or `workload_factory` must be provided — not both, not neither.

---

## Resources Modelled

The following cloud system resources are actively computed and used within the environment at every timestep. Resources referenced in theoretical cloud models (memory, network bandwidth, pod restart probability) are not included — the simulator focuses on the compute and capacity resources that directly govern scaling decisions and SLA outcomes.

### Instance Count — `N_t`

The primary control variable. Represents the number of active server instances currently running. The agent's entire job is to manage this number — every action either increments, decrements, or holds it. Bounded between `N_min=1` and `N_max=20` at all times.

```
N_{t+1} = clip(N_t + (action - 1), N_min, N_max)
```

Instance count determines total system capacity and therefore directly drives every other resource metric below.

### CPU Utilisation — `U_cpu`

Represents how much of the total compute capacity is currently in use. Derived directly from incoming workload and current instance count:

```
U_cpu = λ_t / (N_t · μ)
```

Where `μ = 50` requests/sec is the per-instance service rate. Values above `1.0` indicate overload — more requests arriving than the system can handle. Included in the state vector (index 2), normalised and clipped to `[0, 1]`.

### System Capacity — `C`

Total requests per second the current fleet can handle:

```
C = N_t · μ
```

Not exposed directly in the state vector but used internally to compute utilisation and error rate at every step.

### Response Time — `R`

Per-request mean response time computed using the M/M/1 queueing formula:

```
R = 1 / (μ - λ_i)     where λ_i = λ_t / N_t  (per-instance arrival rate)
```

Response time grows non-linearly as per-instance load approaches the service rate `μ` — near-flat at low utilisation, diverging sharply as the system approaches saturation. Capped at `10.0` seconds under overload conditions. Included in the state vector (index 4) as `response_norm = R / 10`.

### Error Rate

Fraction of incoming requests that cannot be served due to capacity overload:

```
Error = max(0, (λ_t - C) / λ_t)    when λ_t > C
Error = 0                            when λ_t ≤ C
```

Zero under normal operating conditions. Rises when demand exceeds capacity, reaching `1.0` only if instances are completely removed under load. Included in the state vector (index 3) and carries the highest reward weight (`β = 0.6`) since SLA violation is the primary failure mode this system is designed to prevent.

### Operational Cost

Total cost incurred per timestep for running the current number of instances:

```
Cost = N_t · cost_per_instance
```

With `cost_per_instance = 1`, cost is numerically equal to instance count, making it easy to reason about directly. Used in the reward function as both a flat cost term (`γ · norm_cost`) and as the basis for the idle-capacity penalty (`δ · norm_idle`). Not included in the state vector — the agent infers cost implications from `N_norm` and `utilization`.

---

```
Discrete(3)
  0 → scale down  (N_t -= 1)
  1 → maintain    (N_t unchanged)
  2 → scale up    (N_t += 1)
```

Scaling is instant — the new instance count takes effect in the same timestep the action is chosen. Instance count is clipped to `[N_min, N_max]` after every action.

---

## Observation Space (State Vector)

```
Box(low=0.0, high=1.0, shape=(5,), dtype=float32)
```

All five features are normalised to `[0, 1]` before being passed to the agent. This prevents any single feature from dominating gradient updates due to scale differences — raw `λ` can reach 500+ while utilisation and error rate are naturally in `[0, 1]`.

### State Features

| Index | Feature | Raw Value | Normalisation | Description |
|---|---|---|---|---|
| 0 | `lambda_norm` | `λ_t` (requests/sec) | `λ / 1000` | Incoming request rate — the primary demand signal |
| 1 | `N_norm` | `N_t` (1–20) | `(N - N_min) / (N_max - N_min)` | Current active instance count, scaled to unit range |
| 2 | `utilization` | `λ / (N · μ)` | Clipped to `[0, 1]` | Fraction of total capacity currently in use |
| 3 | `error_rate` | `max(0, (λ - C) / λ)` | Already `[0, 1]` | Fraction of requests dropped due to overload |
| 4 | `response_norm` | M/M/1 response time | `R / 10` | Per-request response time, normalised to max cap |

### Why These Five Features

Together these features give the agent both the demand signal (`lambda_norm`) and the consequences of the current provisioning level (`utilization`, `error_rate`, `response_norm`), plus the current resource state (`N_norm`). Utilisation, error rate, and response time are all deterministic functions of `λ` and `N` given fixed `μ` — including them explicitly means the agent does not need to learn the underlying queueing relationships from scratch, which speeds up policy learning under limited training budgets.

---

## State Building — `_build_state(metrics)`

Called after every `reset()` and `step()`. Takes the raw metrics dict from `CloudSimulator` and returns the normalised numpy state vector passed to the agent.

```python
lambda_norm    = min(lambda_t / 1000.0, 1.0)
N_norm         = (N_t - N_min) / (N_max - N_min)
utilization    = min(U_cpu, 1.0)
error_rate     = error_rate               # already [0,1]
response_norm  = min(response_time / 10.0, 1.0)

state = [lambda_norm, N_norm, utilization, error_rate, response_norm]
```

The `info` dict returned by `step()` always carries the **raw, unnormalised** values (`requests`, `instances`, `utilization`, `response_time`, `error_rate`, `cost`) for use by evaluation and plotting code — decoupling metric logging from state representation so normalisation changes never affect downstream analysis.

---

## Reward Function — `compute_reward(metrics, action)`

The reward signal guides the agent toward the project objective: **lower operational cost with marginal SLA compromise**.

### Formula

```
reward = -(α · norm_response + β · norm_error + γ · norm_cost + δ · norm_idle)
         - scaling_penalty
```

### Cost Split Design

Cost is deliberately split into two terms rather than one flat penalty:

**`γ · norm_cost`** penalises the total running cost — instances multiplied by cost per instance, normalised to `[0, 1]`. Kept moderate so scaling up during high load is not made too expensive.

**`δ · norm_idle`** penalises idle/wasted capacity specifically — instances running above what the current workload actually requires (`ceil(λ / μ)`), normalised by `N_max`. This directly incentivises scale-down during low-load periods without penalising necessary scale-up during spikes.

This split was introduced after empirical testing showed a flat cost penalty caused agents to converge to either persistent under-provisioning (high SLA violation) or persistent over-provisioning (high cost, low SLA) depending on the weight — neither of which matched the project objective.

### Weight Table

| Weight | Value | Term | Role |
|---|---|---|---|
| `α` (alpha) | 0.3 | Response time | Secondary priority — penalises high latency |
| `β` (beta) | 0.6 | Error rate (SLA) | Primary driver of scale-up behavior — penalises dropped requests |
| `γ` (gamma) | 0.1 | Necessary cost | Moderate cost signal — allows scale-up when needed |
| `δ` (delta) | 0.3 | Idle capacity | Primary driver of scale-down — penalises over-provisioning during low load |

### Scaling Penalty

```python
scaling_penalty = 0.02 * abs(action - 1)
```

A small fixed penalty for any action other than `maintain` (action=1). Discourages unnecessary instance thrashing — changing `N_t` every single step even when not warranted — without dominating the reward or preventing genuine reactive scaling.

### Weight Rationale

`β > γ` and `β > δ` ensures that SLA violation is always more costly than the savings from under-provisioning. `δ > γ` ensures idle capacity is penalised more strongly than necessary capacity, which is what produces the scale-down incentive during troughs. The 12x asymmetry between one step of SLA violation and one step of idle capacity is intentional — it reflects the practical reality that dropped requests are far more damaging than a marginally higher instance count.

---

## Workload Modes

**Fixed workload (evaluation mode):**
```python
env = CloudEnv(workload=WorkloadGenerator(steps=500, scenario="periodic", seed=0))
```
The same 500-step sequence replays identically every episode. Used in `tests/` and `evaluation/` to ensure fair, reproducible comparisons across agents.

**Factory workload (training mode):**
```python
env = CloudEnv(workload_factory=lambda: WorkloadGenerator(steps=500, scenario=chosen))
```
`reset()` calls the factory each episode, generating a fresh sequence from a potentially different scenario. Used during training to expose agents to varied traffic patterns and prevent memorisation of a single workload.
