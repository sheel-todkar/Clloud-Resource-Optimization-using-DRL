# Single-command entrypoint for the OTHER graph: one episode's raw workload (lambda) plotted against
# PPO/DQN/Baseline's instance-count decisions over time. This is different from run_experiments.py's
# dashboard -- that one aggregates stats across scenarios/seeds, this one shows a single run's full time 
# series.

# Usage:
#   python -m tests.run_single_episode
#   python -m tests.run_single_episode --scenario spike --steps 500 --seed 0
#   python -m tests.run_single_episode --ppo-model models/ppo_200000_20260621_143022.zip --dqn-model models/dqn_200000_20260621_143041.zip

import argparse
 
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator
from evaluation.baseline import BaselineAgent
from evaluation.run_episode import run_episode
from evaluation.plot_results import plot_instances_vs_workload
from evaluation.load_models import load_models_auto, ModelLoadError
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="default",
                         help="Workload scenario: default/spike/linear/noisy/periodic")
    parser.add_argument("--steps", type=int, default=500,
                         help="Episode length (workload steps)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Seed for the shared workload sequence")
    parser.add_argument("--ppo-model", type=str, default=None,
                         help="Explicit PPO model path (overrides config.json and latest)")
    parser.add_argument("--dqn-model", type=str, default=None,
                         help="Explicit DQN model path (overrides config.json and latest)")
 
    args = parser.parse_args()
 
    try:
        ppo_model, dqn_model, _, _ = load_models_auto(
            ppo_path_override=args.ppo_model,
            dqn_path_override=args.dqn_model,
        )
    except ModelLoadError:
        return
 
    # ONE shared workload sequence -> all three agents face identical traffic, so differences 
    # in their plotted instance curves come from the agent's policy, not from luck of the draw in random workload.
    base_workload = WorkloadGenerator(steps=args.steps, scenario=args.scenario, seed=args.seed)
 
    ppo_env      = CloudEnv(WorkloadGenerator(args.steps, sequence=base_workload.sequence))
    dqn_env      = CloudEnv(WorkloadGenerator(args.steps, sequence=base_workload.sequence))
    baseline_env = CloudEnv(WorkloadGenerator(args.steps, sequence=base_workload.sequence))
 
    print("Running PPO...")
    ppo_result = run_episode(ppo_env, model=ppo_model)
 
    print("Running DQN...")
    dqn_result = run_episode(dqn_env, model=dqn_model)
 
    print("Running Baseline...")
    baseline_result = run_episode(baseline_env, model=BaselineAgent())
 
    print("\n--- RESULTS ---")
    print(f"PPO      -> Cost: {ppo_result['cost']:.2f}, Error: {ppo_result['error']:.3f}, Response: {ppo_result['response']:.3f}")
    print(f"DQN      -> Cost: {dqn_result['cost']:.2f}, Error: {dqn_result['error']:.3f}, Response: {dqn_result['response']:.3f}")
    print(f"Baseline -> Cost: {baseline_result['cost']:.2f}, Error: {baseline_result['error']:.3f}, Response: {baseline_result['response']:.3f}")
 
    plot_instances_vs_workload(ppo_result, dqn_result, baseline_result)
 
 
if __name__ == "__main__":
    main()