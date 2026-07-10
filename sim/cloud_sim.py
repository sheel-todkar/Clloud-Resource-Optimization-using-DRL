# Implements the physics of the cloud system as derived in the MDP spec:
#   U_cpu = lambda / (N * mu)
#   R     = 1 / (mu - lambda_i)   where lambda_i = lambda / N   (M/M/1 per-instance)
#   Error = max(0, (lambda - C) / lambda)   where C = N * mu
#   Cost  = N * cost_per_instance

import numpy as np

class CloudSimulator:

    def __init__(self,
                 mu=50,                 # max requests/sec served per instance
                 cost_per_instance=1,   # cost unit per active instance
                 N_min=1,               # lower bound on instance count
                 N_max=20):             # upper bound on instance count

        self.mu = mu
        self.cost_per_instance = cost_per_instance
        self.N_min = N_min
        self.N_max = N_max

        # State is fully (re)initialized in reset()
        self.N_t = None
        self.t = None

        self.reset()

    def reset(self, N_init=5):
        """Reset instance count and internal clock. Returns initial metrics."""
        self.N_t = N_init
        self.t = 0

        return {
            "lambda": 0.0,
            "instances": self.N_t,
            "utilization": 0.0,
            "response_time": 0.0,
            "error_rate": 0.0,
            "cost": self.N_t * self.cost_per_instance,
        }

    def step(self, action, lambda_t):
        """
        Apply a scaling action and the current workload, return resulting metrics.

        action: 0 = scale down, 1 = maintain, 2 = scale up
        lambda_t: incoming request rate for this timestep
        """

        if action == 0:
            self.N_t -= 1
        elif action == 2:
            self.N_t += 1

        self.N_t = int(np.clip(self.N_t, self.N_min, self.N_max))

        capacity = self.N_t * self.mu
        utilization = lambda_t / capacity if capacity > 0 else 1.0

        # Per-instance arrival rate and M/M/1 response time
        lambda_i = lambda_t / self.N_t

        if lambda_i < self.mu:
            epsilon = 1e-6
            response_time = 1.0 / max(self.mu - lambda_i, epsilon)
            error_rate = 0.0
        else:
            response_time = 10.0
            error_rate = max(0.0, (lambda_t - capacity) / lambda_t) if lambda_t > 0 else 0.0

        cost = self.N_t * self.cost_per_instance

        self.t += 1

        return {
            "lambda": lambda_t,
            "instances": self.N_t,
            "utilization": utilization,
            "response_time": response_time,
            "error_rate": error_rate,
            "cost": cost,
        }