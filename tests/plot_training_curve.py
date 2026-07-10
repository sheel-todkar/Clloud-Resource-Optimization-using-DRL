# Plots episode reward vs timesteps for the latest PPO and DQN training
# runs, showing how each agent's policy improved over the course of training.
#
# Reads the Monitor CSV logs written to logs/ during training.
# SB3's Monitor wrapper writes one row per completed episode:
#   r  -> total episode reward
#   l  -> episode length (steps)
#   t  -> wall-clock time elapsed
#
# The cumulative timestep for each episode is reconstructed by summing
# episode lengths -- this gives the x-axis ("training progress") that
# maps directly to the --steps budget used during training.
#
# Usage:
#   python -m tests.plot_training_curve
#   python -m tests.plot_training_curve --window 20

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_monitor_csv(path):
    """
    Reads a Monitor CSV file and returns a DataFrame with columns:
      timestep  -- cumulative environment steps at episode end
      reward    -- total undiscounted episode reward
    """
    # Monitor CSV has a comment header line starting with #, then column names,
    # then data -- pd.read_csv needs to skip the comment line.
    df = pd.read_csv(path, comment="#")

    # Reconstruct cumulative timesteps from episode lengths
    df["timestep"] = df["l"].cumsum()

    return df[["timestep", "r"]].rename(columns={"r": "reward"})


def smooth(values, window):
    """Rolling mean over `window` episodes -- reduces noise for readability."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def find_latest_log(pattern):
    """Returns the most recently modified log file matching pattern."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20,
                        help="Smoothing window size in episodes (default: 20)")
    parser.add_argument("--ppo-log", type=str, default=None,
                        help="Explicit PPO monitor CSV path -- overrides "
                             "config.json and latest")
    parser.add_argument("--dqn-log", type=str, default=None,
                        help="Explicit DQN monitor CSV path -- overrides "
                             "config.json and latest")
    args = parser.parse_args()

    # Priority: explicit --ppo-log/--dqn-log > config.json derived log > latest log
    if args.ppo_log and args.dqn_log:
        ppo_log = args.ppo_log
        dqn_log = args.dqn_log
    else:
        # Load models via the same priority chain as all other scripts --
        # this ensures the training curve shown always corresponds to the
        # same model that run_experiments/run_single_episode would use.
        from evaluation.load_models import load_models_auto, ModelLoadError
        try:
            _, _, ppo_log, dqn_log = load_models_auto(
                ppo_path_override=None,
                dqn_path_override=None,
            )
        except ModelLoadError:
            return

        # Fall back to latest log file if load_models_auto returned None logs
        # (e.g. model was trained before Monitor CSV logging was added)
        if ppo_log is None:
            ppo_log = find_latest_log("logs/ppo_*.monitor.csv")
        if dqn_log is None:
            dqn_log = find_latest_log("logs/dqn_*.monitor.csv")

    if ppo_log is None and dqn_log is None:
        print("No training logs found in logs/. Train a model first with:")
        print("  python -m agents.train --algo both --steps 200000")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress — Episode Reward vs Timesteps", fontsize=13)

    for ax, log_path, label, color in [
        (axes[0], ppo_log, "PPO", "tab:orange"),
        (axes[1], dqn_log, "DQN", "tab:green"),
    ]:
        if log_path is None:
            ax.set_title(f"{label} — no log found")
            ax.axis("off")
            continue

        df = load_monitor_csv(log_path)

        # Raw episode rewards (faint) + smoothed line (solid)
        ax.plot(df["timestep"], df["reward"],
                color=color, alpha=0.2, linewidth=0.8, label="Raw episode reward")

        if len(df) >= args.window:
            smoothed = smooth(df["reward"].values, args.window)
            # smoothed array is shorter by (window-1) -- align to the END
            # of the timestep series so the smoothed line ends at the same
            # timestep as the raw data.
            smoothed_steps = df["timestep"].values[args.window - 1:]
            ax.plot(smoothed_steps, smoothed,
                    color=color, linewidth=2,
                    label=f"Smoothed (window={args.window})")

        ax.set_title(f"{label} Training Curve")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate total episodes and final smoothed reward
        n_episodes = len(df)
        final_reward = smoothed[-1] if len(df) >= args.window else df["reward"].iloc[-1]
        ax.annotate(
            f"Episodes: {n_episodes}\nFinal avg reward: {final_reward:.2f}",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()