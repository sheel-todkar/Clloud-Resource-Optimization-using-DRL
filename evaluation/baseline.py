def threshold_policy(state):
    """
    state = [lambda_norm, N_norm, utilization, error_rate, response_norm]
    (matches CloudEnv._build_state in Approach A)
    """
    utilization = state[2]

    if utilization > 0.8:
        return 2
    elif utilization < 0.3:
        return 0
    else:
        return 1


class BaselineAgent:
    """
    Wraps threshold_policy with the same .predict(obs, deterministic=True)
    interface SB3 models expose, so it can be dropped into evaluation code
    that expects a model-like object (e.g. evaluate_model in tests/).
    """

    def predict(self, obs, deterministic=True):
        action = threshold_policy(obs)
        return action, None