# Single-command pipeline: run the full PPO/DQN/Baseline comparison across
# all scenarios AND immediately plot the dashboard, in one process.
#
# Model selection priority (highest to lowest):
#   1) --ppo-model / --dqn-model CLI flags
#   2) config.json in project root (set paths there to pin a specific pair)
#   3) Most recently modified .zip files in models/ (default)
#
# Usage:
#   python -m tests.run_experiments
#   python -m tests.run_experiments --save
#   python -m tests.run_experiments --scenarios spike noisy --seeds 1 2 3
#   python -m tests.run_experiments --ppo-model models/ppo_200000_20260621_143022.zip --dqn-model models/dqn_200000_20260621_143041.zip

import argparse
import json

from tests.compare import run_comparison
from tests.scenarios import SCENARIOS
from tests.plot_dashboard import plot_all_metrics
from evaluation.load_models import load_models_auto, ModelLoadError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 100])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save", action="store_true",
                         help="Also write results to results.json")
    parser.add_argument("--ppo-model", type=str, default=None,
                         help="Explicit PPO model path (overrides config.json and latest)")
    parser.add_argument("--dqn-model", type=str, default=None,
                         help="Explicit DQN model path (overrides config.json and latest)")

    args = parser.parse_args()

    print("\nRunning experiments...\n")

    try:
        ppo_model, dqn_model, _, _ = load_models_auto(
            ppo_path_override=args.ppo_model,
            dqn_path_override=args.dqn_model,
        )

        results = run_comparison(
            scenarios=args.scenarios,
            seeds=tuple(args.seeds),
            steps=args.steps,
            ppo_model=ppo_model,
            dqn_model=dqn_model,
        )
    except ModelLoadError:
        return

    print("\nExperiments complete. Plotting dashboard...\n")

    plot_all_metrics(results)

    if args.save:
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\nResults also saved to results.json\n")


if __name__ == "__main__":
    main()