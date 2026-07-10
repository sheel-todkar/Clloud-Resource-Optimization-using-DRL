def run_episode(env, model=None, policy_fn=None, deterministic=True):
    """
    Runs one full episode in env using either a trained model (must expose
    .predict(obs, deterministic=...)) or a raw policy_fn(state) -> action,
    e.g. evaluation.baseline.threshold_policy.

    Returns a dict with both per-episode averages AND full step history:
        {
            "cost":      average cost per step,
            "error":     average error_rate per step,
            "response":  average response_time per step,
            "history":   list of the raw info dict from every step
                         (so callers can pull "instances", "requests",
                         "utilization", etc. for plotting or per-step
                         metric computation)
        }
    """
    if model is None and policy_fn is None:
        raise ValueError("run_episode requires either model or policy_fn")

    state, _ = env.reset()
    done = False

    total_cost = 0.0
    total_error = 0.0
    total_response = 0.0
    steps = 0

    history = []

    while not done:

        if model is not None:
            action, _ = model.predict(state, deterministic=deterministic)
        else:
            action = policy_fn(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        total_cost += info["cost"]
        total_error += info["error_rate"]
        total_response += info["response_time"]

        history.append(info)

        state = next_state
        steps += 1
        done = terminated or truncated

    return {
        "cost": total_cost / steps,
        "error": total_error / steps,
        "response": total_response / steps,
        "history": history,
    }


def run_episode_multi_seed(env_fn, model=None, policy_fn=None, seeds=(1, 42, 100)):
    """
    Runs run_episode() once per seed, building a fresh env each time via
    env_fn(seed) -> env. Returns the list of per-seed result dicts (each
    with the same shape as run_episode's return value), so callers can
    feed this directly into tests/metrics.py's compute_metrics/aggregate,
    or just inspect each seed's run individually.
    """
    results = []

    for seed in seeds:
        env = env_fn(seed)
        result = run_episode(env, model=model, policy_fn=policy_fn)
        results.append(result)

    return results