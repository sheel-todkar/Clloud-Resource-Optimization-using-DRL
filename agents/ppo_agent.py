import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator


def train_ppo(total_timesteps=100_000, model_path=None,
              workload=None, workload_factory=None):

    os.makedirs("models", exist_ok=True)

    # Default naming: ppo_<steps>_<timestamp>.zip  e.g. ppo_10000_20260617_132143.zip
    if model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/ppo_{total_timesteps}_{timestamp}"

    # Save Monitor logs to logs/ -- this CSV is what plot_training_curve.py reads for the learning progress plot.
    log_path = model_path.replace("models/", "logs/")
    os.makedirs("logs", exist_ok=True)

    # workload_factory takes priority -- if provided, CloudEnv regenerates a fresh scenario each episode. 
    # If only workload is provided, the same fixed sequence replays every episode.
    if workload_factory is not None:
        env = Monitor(CloudEnv(workload_factory=workload_factory),filename=log_path)
    else:
        if workload is None:
            workload = WorkloadGenerator(steps=1000)
        env = Monitor(CloudEnv(workload=workload),filename=log_path)

    # n_steps scaled to the training budget: a fixed 2048 with only 10k total steps gives ~4 policy updates 
    # -- too few to learn anything. This keeps ~20 updates regardless of how short the run is.
    n_steps = min(2048, max(256, total_timesteps // 20))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    env.close()

    print(f"PPO model saved to {model_path}.zip")

    return model