# Reusable helper for loading trained PPO/DQN models into the evaluation/testing pipeline.

# Model selection priority (highest to lowest):
#   1) CLI flags (--ppo-model / --dqn-model) passed by the caller
#   2) config.json in the project root -- set paths here to pin a specific
#      model pair without touching CLI or source code
#   3) load_latest_models() -- auto-picks most recently modified .zip files
#
# To pin specific models, edit config.json:
#   {
#       "ppo_model": "models/ppo_200000_20260621_143022.zip",
#       "dqn_model": "models/dqn_200000_20260621_143041.zip"
#   }
# To revert to latest, set both values back to "".
#
# Filenames follow agent_steps_timestamp.zip (e.g. ppo_10000_20260617_132143.zip),
# set by ppo_agent.py / dqn_agent.py. "Latest" is determined by file
# modification time, NOT filename sort.
#
# Log files follow the same naming convention as models:
#   models/ppo_200000_20260621_143022.zip
#     -> logs/ppo_200000_20260621_143022.monitor.csv

import os
import glob
import json
from stable_baselines3 import PPO, DQN


CONFIG_PATH = "config.json"


class ModelLoadError(Exception):
    """Raised when a requested model file can't be found or fails to load."""
    pass


def _latest_file(pattern):
    """Returns the path of the most recently MODIFIED file matching pattern."""
    matches = glob.glob(pattern)
    if not matches:
        raise ModelLoadError(f"No files found matching pattern: {pattern}")
    return max(matches, key=os.path.getmtime)


def _derive_log_path(model_path):
    """
    Derives the Monitor CSV log path from a model path using the
    filename convention:
      models/ppo_200000_20260621_143022.zip
        -> logs/ppo_200000_20260621_143022.monitor.csv

    Returns the log path if the file exists, None if no log was found.
    """
    base = os.path.splitext(os.path.basename(model_path))[0]
    log_path = os.path.join("logs", base + ".monitor.csv")
    return log_path if os.path.exists(log_path) else None


def _read_config():
    """
    Reads config.json from the project root.
    Returns (ppo_path, dqn_path) if both are set to non-empty strings,
    otherwise returns (None, None) so the caller falls through to latest.
    """
    if not os.path.exists(CONFIG_PATH):
        return None, None

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

        ppo_path = config.get("ppo_model", "").strip()
        dqn_path = config.get("dqn_model", "").strip()

        if ppo_path and dqn_path:
            return ppo_path, dqn_path

    except (json.JSONDecodeError, KeyError):
        print("Warning: config.json is malformed -- falling back to latest models")

    return None, None


def load_latest_models(models_dir="models"):
    """
    Loads the most recently created ppo_*.zip and dqn_*.zip from models_dir.

    Returns: (ppo_model, dqn_model)
    Raises: ModelLoadError if either file is missing or fails to load.
    """
    try:
        ppo_path = _latest_file(os.path.join(models_dir, "ppo_*.zip"))
        dqn_path = _latest_file(os.path.join(models_dir, "dqn_*.zip"))

        ppo_model = PPO.load(ppo_path)
        dqn_model = DQN.load(dqn_path)

        print(f"Loaded latest PPO model: {ppo_path}")
        print(f"Loaded latest DQN model: {dqn_path}")

        return ppo_model, dqn_model

    except Exception:
        print("Failed to load models")
        raise ModelLoadError("Failed to load models") from None


def load_specific_models(ppo_path, dqn_path):
    """
    Loads exact, pinned model files rather than auto-selecting the latest.
    Also resolves the corresponding Monitor CSV log for each model.

    Returns: (ppo_model, dqn_model, ppo_log_path, dqn_log_path)
             ppo_log_path / dqn_log_path are None if no log file exists.
    Raises:  ModelLoadError if either model file is missing or fails to load.
    """
    try:
        ppo_model = PPO.load(ppo_path)
        dqn_model = DQN.load(dqn_path)

        print(f"Loaded PPO model: {ppo_path}")
        print(f"Loaded DQN model: {dqn_path}")

        ppo_log = _derive_log_path(ppo_path)
        dqn_log = _derive_log_path(dqn_path)

        if ppo_log:
            print(f"Corresponding PPO log: {ppo_log}")
        else:
            print(f"Warning: no log found for {ppo_path}")

        if dqn_log:
            print(f"Corresponding DQN log: {dqn_log}")
        else:
            print(f"Warning: no log found for {dqn_path}")

        return ppo_model, dqn_model, ppo_log, dqn_log

    except Exception:
        print("Failed to load models")
        raise ModelLoadError("Failed to load models") from None


def load_models_auto(ppo_path_override=None, dqn_path_override=None):
    """
    Main entry point for all scripts that need to load models.
    Implements the full priority chain:

        1) CLI override (ppo_path_override / dqn_path_override if provided)
        2) config.json  (if both paths are set to non-empty strings)
        3) Latest model (most recently modified .zip in models/)

    Returns: (ppo_model, dqn_model, ppo_log, dqn_log)
             log paths are None if no corresponding CSV exists.
    Raises:  ModelLoadError on any load failure.
    """
    # Priority 1: explicit CLI override
    if ppo_path_override and dqn_path_override:
        print("Loading models from CLI arguments...")
        return load_specific_models(ppo_path_override, dqn_path_override)

    # Priority 2: config.json
    ppo_path, dqn_path = _read_config()
    if ppo_path and dqn_path:
        print("Loading models from config.json...")
        return load_specific_models(ppo_path, dqn_path)

    # Priority 3: latest
    print("Loading latest models...")
    ppo_model, dqn_model = load_latest_models()
    ppo_log = _derive_log_path(
        _latest_file(os.path.join("models", "ppo_*.zip"))
    )
    dqn_log = _derive_log_path(
        _latest_file(os.path.join("models", "dqn_*.zip"))
    )
    return ppo_model, dqn_model, ppo_log, dqn_log


# --------------------------------------------------------------------
# To pin specific models, edit config.json in the project root:
#
#   {
#       "ppo_model": "models/ppo_200000_20260621_143022.zip",
#       "dqn_model": "models/dqn_200000_20260621_143041.zip"
#   }
#
# To revert to auto-latest, set both values back to "".
#
# All scripts (run_experiments, run_single_episode, results_summary,
# plot_training_curve, diagnose_reward) call load_models_auto() and
# respect this priority chain automatically.
# --------------------------------------------------------------------