# Usage:
#   python -m agents.train --algo ppo  --steps 100000
#   python -m agents.train --algo dqn  --steps 100000
#   python -m agents.train --algo both --steps 100000

import os
import argparse
import numpy as np

from agents.ppo_agent import train_ppo
from agents.dqn_agent import train_dqn
from sim.workload import WorkloadGenerator


def train_both(total_steps=100_000, episode_length=500):

    """Trains DQN and PPO using weighted random scenario cycling.

    Each episode reset() draws a scenario from a weighted distribution
    (default 40%, others 15% each) and generates a fresh WorkloadGenerator
    for that scenario -- so agents learn to handle spike, linear, noisy,
    periodic, and default traffic patterns proportionally, rather than
    memorizing one fixed sequence.

    """

    os.makedirs("models", exist_ok=True)

    scenarios = ["default", "spike", "linear", "noisy", "periodic"]

    # Weights: default appears 40% of time, others 15% each.
    # Reflects that normal traffic is more common than extreme patterns, and keeps training 
    # anchored to the baseline workload shape while still exposing agents to all scenario types.
    weights = [0.40, 0.15, 0.15, 0.15, 0.15]

    # Seeded RNG for reproducibility -- same scenario sequence across multiple training runs with the same seed.
    rng = np.random.default_rng(42)

    def workload_factory():
        # Called once per episode via CloudEnv.reset().
        scenario = rng.choice(scenarios, p=weights)
        return WorkloadGenerator(steps=episode_length, scenario=scenario)

    print("Training DQN...")
    dqn_model = train_dqn(
        total_timesteps=total_steps,
        workload_factory=workload_factory,
    )

    print("\nTraining PPO...")
    ppo_model = train_ppo(
        total_timesteps=total_steps,
        workload_factory=workload_factory,
    )

    return dqn_model, ppo_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="ppo")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--episode-length", type=int, default=500)

    args = parser.parse_args()

    print(f"Starting training: algo={args.algo}, steps={args.steps}\n")

    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps)

    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps)

    elif args.algo == "both":
        train_both(total_steps=args.steps, episode_length=args.episode_length)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()