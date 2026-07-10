# Formats the full PPO/DQN/Baseline comparison as a clean results table,
# printed to the terminal and optionally saved as a CSV.
#
# Runs the same comparison pipeline as run_experiments.py but outputs a
# structured table instead of (or in addition to) the bar chart dashboard,
# giving examiners and reviewers a concrete set of numbers to reference.
#
# Each row in the table is one (scenario, agent) combination.
# Each column is one metric: cost, SLA violation, latency, utilisation.
# Mean ± std is shown for each cell, computed across the evaluation seeds.
#
# Usage:
#   python -m tests.results_summary
#   python -m tests.results_summary --save          (also writes results_summary.csv)
#   python -m tests.results_summary --scenarios spike periodic

import argparse
import csv
import sys

from tests.compare import run_comparison
from tests.scenarios import SCENARIOS
from evaluation.load_models import load_models_auto, ModelLoadError


METRICS = ["cost", "sla_violation", "latency", "utilization"]
AGENTS  = ["PPO", "DQN", "Baseline"]
 
COL_SCENARIO = 10
COL_AGENT    = 10
COL_METRIC   = 18
 
 
def format_cell(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"
 
 
def print_table(results):
    header = (
        f"{'Scenario':<{COL_SCENARIO}}"
        f"{'Agent':<{COL_AGENT}}"
        + "".join(f"{m.upper():<{COL_METRIC}}" for m in METRICS)
    )
    separator = "-" * len(header)
 
    print("\n" + separator)
    print(header)
    print(separator)
 
    for scenario in results:
        first_agent = True
        for agent in AGENTS:
            metrics = results[scenario][agent]
            scenario_label = scenario if first_agent else ""
            first_agent = False
 
            row = (
                f"{scenario_label:<{COL_SCENARIO}}"
                f"{agent:<{COL_AGENT}}"
                + "".join(
                    f"{format_cell(metrics[m]['mean'], metrics[m]['std']):<{COL_METRIC}}"
                    for m in METRICS
                )
            )
            print(row)
 
        print(separator)
 
    print()
 
 
def save_csv(results, path="results_summary.csv"):
    rows = []
 
    for scenario in results:
        for agent in AGENTS:
            metrics = results[scenario][agent]
            row = {"scenario": scenario, "agent": agent}
 
            for m in METRICS:
                row[f"{m}_mean"] = round(metrics[m]["mean"], 4)
                row[f"{m}_std"]  = round(metrics[m]["std"],  4)
 
            rows.append(row)
 
    fieldnames = (
        ["scenario", "agent"]
        + [f"{m}_mean" for m in METRICS]
        + [f"{m}_std"  for m in METRICS]
    )
 
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"Results saved to {path}")
 
 
def print_highlights(results):
    print("--- Cost vs Baseline ---")
 
    for scenario in results:
        baseline_cost = results[scenario]["Baseline"]["cost"]["mean"]
 
        for agent in ["PPO", "DQN"]:
            agent_cost = results[scenario][agent]["cost"]["mean"]
            diff = agent_cost - baseline_cost
            pct  = (diff / baseline_cost) * 100
 
            symbol    = "+" if diff <= 0 else "x"
            direction = "lower" if diff <= 0 else "higher"
 
            print(
                f"  {symbol} {agent} vs Baseline | {scenario:>8}: "
                f"{agent_cost:.0f} vs {baseline_cost:.0f}  "
                f"({abs(pct):.1f}% {direction})"
            )
 
    print()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 100, 7, 23])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save", action="store_true",
                        help="Save results to results_summary.csv")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="Explicit PPO model path (overrides config.json and latest)")
    parser.add_argument("--dqn-model", type=str, default=None,
                        help="Explicit DQN model path (overrides config.json and latest)")
 
    args = parser.parse_args()
 
    print("\nRunning comparison...\n")
 
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
 
    print_table(results)
    print_highlights(results)
 
    if args.save:
        save_csv(results)
 
 
if __name__ == "__main__":
    main()