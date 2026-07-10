# 🚀 Cloud Resource Optimization using Deep Reinforcement Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reinforcement learning system for intelligent and cost-efficient cloud auto-scaling. The project trains **PPO** and **DQN** agents to dynamically adjust the number of active server instances in response to changing workloads, aiming to minimize infrastructure costs while maintaining acceptable service-level performance.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Project Structure](#-project-structure)
- [How It Works](#️-how-it-works)
- [Design Rationale](#-design-rationale)
- [Advantages of RL-Based Auto-Scaling](#-advantages-of-rl-based-auto-scaling)
- [Technology Stack](#️-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Evaluation Scenarios](#-evaluation-scenarios)
- [Project Goal](#-project-goal)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

Cloud environments experience highly dynamic and unpredictable workloads. Choosing the correct number of active instances is a continuous balancing act:

- **Under-provisioning** leads to SLA violations, increased latency, and dropped requests.
- **Over-provisioning** wastes infrastructure resources and increases operational costs.

Traditional threshold-based auto-scalers react to current utilization but cannot systematically optimize multiple competing objectives.

This project models cloud auto-scaling as a **Markov Decision Process (MDP)** and uses **Reinforcement Learning (RL)** to learn adaptive scaling policies that outperform conventional rule-based approaches.

---

## 🏗️ Architecture

### High-Level Flow

```text
Workload Generator
        │
        ▼
 Cloud Simulator
        │
        ▼
 Gymnasium Environment
        │
        ▼
 PPO / DQN Agents
        │
        ▼
 Evaluation & Testing
```

---

## 📂 Project Structure

```text
Cloud-Resource-Optimization-using-DRL/
│
├── sim/                    Cloud simulator and workload generator
│   ├── cloud_sim.py            M/M/1 queueing-based cloud simulator
│   └── workload.py             Multi-scenario workload generator
│
├── envs/                   Gymnasium environment wrapper
│   └── cloud_env.py            RL-compatible Gym interface with reward shaping
│
├── agents/                 PPO and DQN training pipelines
│   ├── ppo_agent.py            PPO agent configuration and training
│   ├── dqn_agent.py            DQN agent configuration and training
│   └── train.py                Unified CLI for training (single or both agents)
│
├── evaluation/             Baseline policy and evaluation utilities
│   ├── baseline.py             Threshold-based baseline policy
│   ├── load_models.py          Model loading and evaluation helpers
│   ├── run_episode.py          Single-episode runner
│   └── plot_results.py         Result visualization utilities
│
├── tests/                  Experiment runner and dashboard generation
│   ├── run_experiments.py      Multi-scenario experiment runner
│   ├── run_single_episode.py   Single-episode visualization
│   ├── plot_training_curve.py  Training curve plotter
│   ├── plot_dashboard.py       Dashboard generation
│   ├── results_summary.py      Summary table generator
│   ├── compare.py              Agent comparison utilities
│   ├── diagnose_reward.py      Reward diagnostics
│   ├── metrics.py              Metric computation utilities
│   └── scenarios.py            Scenario definitions
│
├── models/                 Saved trained model checkpoints
├── logs/                   Monitor CSV logs written during training
├── results/                Generated evaluation outputs
│
├── requirements.txt        Python dependencies
├── .gitignore              Git ignore rules
└── README.md               This file
```

### Dependency Flow

```text
sim/ ← envs/ ← agents/
                  │
                  ▼
             evaluation/ ← tests/
```

**Module Responsibilities**

| Module | Responsibility |
|---|---|
| `sim/` | Cloud dynamics, M/M/1 queueing model, workload generation |
| `envs/` | RL-compatible Gymnasium interface with reward shaping |
| `agents/` | PPO and DQN training with mixed-scenario cycling |
| `evaluation/` | Baseline comparison and model evaluation |
| `tests/` | Multi-scenario experiments, dashboards, and diagnostics |

---

## ⚙️ How It Works

At each discrete timestep:

1. The **workload generator** produces the incoming request rate `λₜ`.
2. The RL agent **observes** the current cloud state (5 normalized features):
   - `λ_norm` — normalized request rate
   - `N_norm` — normalized instance count
   - `U_cpu` — CPU utilization
   - `Error` — error rate
   - `R_norm` — normalized response time
3. The agent selects one of **three actions**:
   | Action | Effect |
   |---|---|
   | `0` | Scale Down (remove 1 instance) |
   | `1` | Maintain Current Capacity |
   | `2` | Scale Up (add 1 instance) |
4. The simulator updates the number of active instances `Nₜ`.
5. Performance metrics are computed using an **M/M/1 queueing model**:
   - **CPU Utilization**: `U = λ / (N × μ)`
   - **Response Time**: `R = 1 / (μ − λᵢ)` where `λᵢ = λ / N`
   - **Error Rate**: `max(0, (λ − C) / λ)` where `C = N × μ`
   - **Infrastructure Cost**: `N × cost_per_instance`
6. A **shaped reward** is generated: `R = −(α·response + β·error + γ·cost + δ·idle) − scaling_penalty`
7. The agent **updates its policy** to maximize long-term cumulative reward.

---

## 🧠 Design Rationale

### Markov Decision Process (MDP)

Auto-scaling naturally fits an MDP framework because it contains:

- Observable system state
- Discrete control actions
- Stochastic workload transitions
- A measurable optimization objective

This formulation enables the use of modern reinforcement learning algorithms without requiring custom optimization techniques.

### Simulator–Environment Separation

The simulator remains completely independent of RL:

```text
CloudSimulator
      ↓
Gym Environment
      ↓
RL Agent
```

**Benefits:**

- Easier testing and debugging
- Cleaner, modular architecture
- Future extensibility (swap simulators, add real cloud APIs)
- Independent simulator improvements

### Mixed-Scenario Training

Agents are trained across multiple workload patterns with weighted random cycling:

| Scenario | Training Weight |
|---|---|
| Default | 40% |
| Spike | 15% |
| Linear Growth | 15% |
| Noisy | 15% |
| Periodic | 15% |

This improves policy robustness and prevents overfitting to a single traffic pattern.

### Idle-Capacity Penalty

The reward function explicitly penalizes unnecessary idle resources, encouraging:

- ✅ Cost-efficient scaling
- ✅ Aggressive scale-down during low demand
- ✅ Sufficient capacity during workload spikes

---

## ✅ Advantages of RL-Based Auto-Scaling

### 💰 Lower Operational Cost

RL agents learn when resources are genuinely required, reducing unnecessary instance usage.

### 📈 Better Generalization

A single trained policy can handle multiple workload types without manual retuning.

### ⚖️ Multi-Objective Optimization

The reward function simultaneously considers:

- Cost
- Latency
- SLA Violations
- Resource Efficiency

### 🔮 Proactive Scaling

Unlike threshold policies that only react after utilization changes, RL agents can learn workload patterns and scale ahead of demand.

### 📊 Reproducible Evaluation

All approaches are evaluated under identical conditions:

- Same workload traces
- Same metrics
- Same evaluation procedures

This ensures fair comparison between PPO, DQN, and baseline policies.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Reinforcement Learning | Stable-Baselines3 |
| Algorithms | PPO, DQN |
| Environment Interface | Gymnasium |
| Numerical Computing | NumPy |
| Queueing Model | M/M/1 |
| Visualization | Matplotlib |
| Language | Python 3.11+ |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/sheel-todkar/Clloud-Resource-Optimization-using-DRL.git
cd Clloud-Resource-Optimization-using-DRL
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train Agents

Train both PPO and DQN:

```bash
python -m agents.train --algo both --steps 200000
```

Train a specific algorithm:

```bash
python -m agents.train --algo ppo --steps 200000
python -m agents.train --algo dqn --steps 200000
```

---

### 2️⃣ Plot Training Curves

```bash
python -m tests.plot_training_curve
```

---

### 3️⃣ Visualize a Single Episode

```bash
python -m tests.run_single_episode
```

---

### 4️⃣ Run Evaluation

Generate metrics and comparison dashboard:

```bash
python -m tests.run_experiments
```

---

### 5️⃣ Results Summary Table

```bash
python -m tests.results_summary --save
```

---

### 6️⃣ Evaluate Specific Scenarios (Optional)

```bash
python -m tests.run_single_episode --scenario spike --steps 500 --seed 0
```

```bash
python -m tests.run_experiments --scenarios spike noisy --seeds 1 2 3 --save
```

---

## 🌊 Evaluation Scenarios

The system is tested against several workload patterns:

| Scenario | Description | Characteristics |
|---|---|---|
| Default | Mixed realistic workload | Sinusoidal base + noise + random bursts |
| Spike | Sudden traffic bursts | Low traffic → sudden jump to 500 req/s |
| Linear | Gradually increasing demand | Starts at 10, grows by 2 per timestep |
| Noisy | Random fluctuations | Uniform random between 20–500 req/s |
| Periodic | Repeating cyclic patterns | Sinusoidal with amplitude 150 around 200 |

These scenarios help evaluate both adaptability and robustness.

---

## 🎯 Project Goal

Develop an intelligent cloud resource management system capable of:

- Minimizing infrastructure cost
- Maintaining SLA compliance
- Reducing response latency
- Adapting to diverse workload patterns
- Demonstrating the effectiveness of reinforcement learning for cloud resource optimization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/sheel-todkar">Sheel Todkar</a>
</p>