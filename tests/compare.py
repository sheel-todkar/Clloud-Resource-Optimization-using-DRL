# The single comparison entrypoint: runs PPO, DQN, and Baseline across a list of scenarios,
# each with multiple seeds, and returns aggregated metrics ready for plotting.


from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator
from evaluation.baseline import BaselineAgent
from evaluation.run_episode import run_episode_multi_seed
from tests.metrics import compute_metrics, aggregate
from tests.scenarios import SCENARIOS


def env_fn(scenario, seed, steps=500):
    """Builds a fresh CloudEnv for a given scenario+seed combination."""
    workload = WorkloadGenerator(steps=steps, scenario=scenario, seed=seed)
    return CloudEnv(workload)


def evaluate_agent(agent_label, model, scenario, seeds=(1, 42, 100), steps=500):
    """
    Runs one agent (PPO/DQN model, or BaselineAgent) across multiple seeds
    for a single scenario, returning aggregated mean/std metrics.

    model must expose .predict(obs, deterministic=...) -- both SB3 models
    and BaselineAgent satisfy this, so no special-casing is needed here.
    """
    per_seed_results = run_episode_multi_seed(
        env_fn=lambda seed: env_fn(scenario, seed, steps=steps),
        model=model,
        seeds=seeds,
    )

    # Each result already has "history" (full step list) -- compute_metrics
    # needs that history, not the averaged cost/error/response shortcuts.
    per_seed_metrics = [compute_metrics(r["history"]) for r in per_seed_results]

    return aggregate(per_seed_metrics)


def run_comparison(scenarios=None, seeds=(1, 42, 100), steps=500,
                   ppo_model=None, dqn_model=None):
    """
    Runs PPO, DQN, and Baseline across every scenario in `scenarios`
    (defaults to tests.scenarios.SCENARIOS), returning:

        {
            scenario_name: {
                "PPO":      {metric: {"mean":..., "std":...}, ...},
                "DQN":      {...},
                "Baseline": {...},
            },
            ...
        }

    Models can be passed in directly (ppo_model, dqn_model) -- useful when
    loading a specific pinned checkpoint via load_specific_models(). If not
    provided, load_latest_models() is used as the default.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    if ppo_model is None or dqn_model is None:
        from evaluation.load_models import load_latest_models
        ppo_model, dqn_model = load_latest_models()

    baseline = BaselineAgent()

    results = {}

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")

        results[scenario] = {
            "PPO": evaluate_agent("PPO", ppo_model, scenario, seeds=seeds, steps=steps),
            "DQN": evaluate_agent("DQN", dqn_model, scenario, seeds=seeds, steps=steps),
            "Baseline": evaluate_agent("Baseline", baseline, scenario, seeds=seeds, steps=steps),
        }

    return results