# Approach: full 5-feature state, normalized to [0,1] so no single feature dominates gradients during training.
#   state = [lambda_norm, N_norm, U_cpu, Error, R_norm]

# Reward and step/reset logic read from the metrics dict returned by the
# simulator -- this keeps reward correct regardless of how state normalization is done.

# Supports two workload modes:
#   workload:         fixed WorkloadGenerator instance -- used for evaluation,
#                     testing, and single-scenario training (same sequence
#                     replayed every episode).
#   workload_factory: callable -> WorkloadGenerator -- used for mixed-scenario
#                     training (Option C). Called once per reset() so each
#                     episode gets a freshly generated workload sequence,
#                     allowing the agent to generalize across scenario types.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim.cloud_sim import CloudSimulator


class CloudEnv(gym.Env):

    # Normalization caps -- chosen to comfortably cover the workload generator's range (max ~ base + amp
    # + burst ~= 100+80+200 = 380,plus headroom for "spike" scenario which goes to 500).
    LAMBDA_MAX = 1000.0
    RESPONSE_MAX = 10.0

    def __init__(self, workload=None, workload_factory=None,
                 mu=50, cost_per_instance=1, N_min=1, N_max=20):
        super().__init__()

        if workload is None and workload_factory is None:
            raise ValueError("CloudEnv requires either workload or workload_factory")

        self.sim = CloudSimulator(mu=mu, cost_per_instance=cost_per_instance,
                                   N_min=N_min, N_max=N_max)

        self.workload_factory = workload_factory
        self.workload = workload if workload_factory is None else workload_factory()
        self.t = 0

        # 0 = scale down, 1 = maintain, 2 = scale up
        self.action_space = spaces.Discrete(3)

        # All 5 features normalized to [0,1]. Features: [lambda, N, utilization, error, response time]
        low = np.zeros(5, dtype=np.float32)
        high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        metrics = self.sim.reset()
        self.t = 0

        if self.workload_factory is not None:
            self.workload = self.workload_factory()

        metrics["lambda"] = self.workload.get(self.t)

        state = self._build_state(metrics)

        return state, {}

    def step(self, action):
        lambda_t = self.workload.get(self.t)

        metrics = self.sim.step(action, lambda_t)

        state = self._build_state(metrics)
        reward = self.compute_reward(metrics, action)

        self.t += 1

        terminated = False
        truncated = self.t >= len(self.workload.sequence)

        # Raw Values before Normalization for logging/plotting/evaluation
        info = {
            "requests": metrics["lambda"],
            "served_requests": metrics["lambda"] * (1 - metrics["error_rate"]),
            "utilization": min(metrics["utilization"], 1.0),
            "response_time": metrics["response_time"],
            "error_rate": metrics["error_rate"],
            "cost": metrics["cost"],
            "instances": metrics["instances"],
            "action": action,
            "reward": reward,
        }

        return state, reward, terminated, truncated, info

    def _build_state(self, metrics):
        """Normalize all 5 features to [0,1] to avoid scale imbalance."""

        lambda_norm = min(metrics["lambda"] / self.LAMBDA_MAX, 1.0)
        N_norm = (metrics["instances"] - self.sim.N_min) / (self.sim.N_max - self.sim.N_min)
        utilization_norm = min(metrics["utilization"], 1.0)
        error_norm = metrics["error_rate"]
        response_norm = min(metrics["response_time"] / self.RESPONSE_MAX, 1.0)

        return np.array([
            lambda_norm,
            N_norm,
            utilization_norm,
            error_norm,
            response_norm,
        ], dtype=np.float32)

    def compute_reward(self, metrics, action):

        """
        Reward = -(alpha * response + beta * error + gamma * cost + delta * idle) - scaling_penalty

        Now cost is split into two terms:
          gamma * norm_cost  -- penalizes necessary cost (running instances
                                 to serve current load), kept moderate so
                                 scale-up during spikes is still worthwhile
          delta * norm_idle  -- penalizes IDLE/WASTED capacity specifically
                                 (instances above what current load requires),
                                 giving a strong direct incentive to scale DOWN
                                 during low-load periods


        alpha   response time -- secondary priority
        beta    SLA/error -- highest priority
        gamma   necessary cost -- moderate, allows scale-up when needed
        delta   idle/wasted capacity -- high, directly drives scale-down

        """        

        max_response = self.RESPONSE_MAX
        max_cost = self.sim.N_max * self.sim.cost_per_instance

        norm_response = metrics["response_time"] / max_response
        norm_error = metrics["error_rate"]
        norm_cost = metrics["cost"] / max_cost

        required_instances = np.ceil(metrics["lambda"] / self.sim.mu)
        required_instances = np.clip(
            required_instances, self.sim.N_min, self.sim.N_max
        )
        idle_instances = max(0.0, metrics["instances"] - required_instances)
        norm_idle = idle_instances / self.sim.N_max

        alpha = 0.3
        beta  = 0.6
        gamma = 0.1
        delta = 0.3

        reward = -(
            alpha * norm_response
            + beta  * norm_error
            + gamma * norm_cost
            + delta * norm_idle
        )

        # Small penalty for any (action != 1), discourages pointless thrashing without dominating the reward.
        scaling_penalty = 0.02 * abs(action - 1)
        reward -= scaling_penalty

        return reward