# Diagnostic: is the agent's "stay under-provisioned" behavior because
# (a) scaling up genuinely isn't worth it under the current reward weights
#     (a reward-design problem), or
# (b) the agent hasn't learned that scaling up helps yet
#     (a training/exploration problem)?

# Method: run the SAME workload sequence through TWO independent simulators in lockstep 
# -- one driven by the trained model's actual chosen actions, one where action is FORCED to scale-up 
# (2) whenever utilization is high (>0.7), otherwise maintain. Compare per-step and total reward between
# the two. If forced-scale-up gives clearly better (less negative) reward, the policy just hasn't learned 
# this yet -> training problem. If forced scale-up gives similar/worse reward despite fixing error/latency,
# the cost term is outweighing error/latency in the reward math itself -> reward-design problem.

import numpy as np
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator
from evaluation.load_models import load_latest_models, ModelLoadError


def run_with_forced_scaling(workload, steps, utilization_trigger=0.7):
    """
    Steps through CloudEnv, but OVERRIDES the action: scale up whenever
    utilization is above utilization_trigger, otherwise maintain.
    This bypasses any learned policy entirely -- it's a hand-forced
    "always react to load" rule, used purely to probe the reward surface.
    """
    env = CloudEnv(workload)
    state, _ = env.reset()

    total_reward = 0.0
    rewards = []
    instances = []
    utilizations = []

    done = False
    # Track utilization from the PREVIOUS step's info to decide this
    # step's forced action (current step's utilization isn't known until
    # after we've already chosen the action -- this mirrors how a real
    # reactive controller would only know last step's reading).
    last_utilization = 0.0

    while not done:
        action = 2 if last_utilization > utilization_trigger else 1  # scale up or maintain

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        instances.append(info["instances"])
        utilizations.append(info["utilization"])

        last_utilization = info["utilization"]
        done = terminated or truncated

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / len(rewards),
        "rewards": rewards,
        "instances": instances,
        "utilizations": utilizations,
    }


def run_with_policy(workload, steps, model):
    """Steps through CloudEnv using the trained model's actual actions."""
    env = CloudEnv(workload)
    state, _ = env.reset()

    total_reward = 0.0
    rewards = []
    instances = []
    utilizations = []

    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        instances.append(info["instances"])
        utilizations.append(info["utilization"])

        done = terminated or truncated

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / len(rewards),
        "rewards": rewards,
        "instances": instances,
        "utilizations": utilizations,
    }


def main():
    steps = 500

    print("Loading latest PPO/DQN models...\n")
    try:
        ppo_model, dqn_model = load_latest_models()
    except ModelLoadError:
        return

    # Same workload sequence for every comparison -- fair test.
    base_workload = WorkloadGenerator(steps=steps, seed=0)

    for label, model in [("PPO", ppo_model), ("DQN", dqn_model)]:

        policy_workload = WorkloadGenerator(steps, sequence=base_workload.sequence)
        forced_workload = WorkloadGenerator(steps, sequence=base_workload.sequence)

        policy_run = run_with_policy(policy_workload, steps, model)
        forced_run = run_with_forced_scaling(forced_workload, steps)

        print(f"--- {label} ---")
        print(f"Policy's own actions   -> mean reward/step: {policy_run['mean_reward']:.4f}, "
              f"mean instances: {np.mean(policy_run['instances']):.2f}, "
              f"mean utilization: {np.mean(policy_run['utilizations']):.3f}")
        print(f"Forced scale-up (>0.7) -> mean reward/step: {forced_run['mean_reward']:.4f}, "
              f"mean instances: {np.mean(forced_run['instances']):.2f}, "
              f"mean utilization: {np.mean(forced_run['utilizations']):.3f}")

        diff = forced_run["mean_reward"] - policy_run["mean_reward"]
        print(f"Forced-scaling reward advantage: {diff:+.4f} per step")

        if diff > 0.02:
            print(f"=> Forced scale-up is CLEARLY better. {label} has NOT learned to "
                  f"exploit this -- points to a training/exploration problem.\n")
        elif diff < -0.02:
            print(f"=> Forced scale-up is WORSE despite fixing overload -- the cost term "
                  f"is outweighing error/latency. Points to a reward-design problem.\n")
        else:
            print(f"=> Roughly a wash -- {label}'s policy may already be near a local "
                  f"optimum given current reward weights either way.\n")


if __name__ == "__main__":
    main()