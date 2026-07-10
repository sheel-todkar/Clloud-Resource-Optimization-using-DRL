# Generates lambda_t (incoming request rate) sequences for different scenarios.
# Matches the PDF's stochastic/sinusoidal workload model:
#   lambda_{t+1} = lambda_base + A*sin(omega*t) + noise

import numpy as np

class WorkloadGenerator:

    def __init__(self, steps, scenario="default", seed=None, sequence=None):
        self.steps = steps
        self.scenario = scenario

        # Independent RNG instance -> no global seed pollution, so multiple generators in the same process don't interfere.
        self.rng = np.random.default_rng(seed)

        if sequence is not None:
            # Reuse an existing sequence (e.g. for fair cross-agent comparison)
            self.sequence = sequence
        else:
            self.sequence = self._generate_sequence()

    def _generate_sequence(self):
        """Builds a lambda_t sequence according to the chosen scenario."""

        seq = []
        phase = self.rng.uniform(0, 2 * np.pi)
        freq = self.rng.uniform(0.04, 0.07)
        amp = self.rng.uniform(50, 80)

        for t in range(self.steps):

            if self.scenario == "spike":
                val = 50 if t < 80 else 500

            elif self.scenario == "linear":
                val = 10 + t * 2

            elif self.scenario == "noisy":
                val = self.rng.integers(20, 500)

            elif self.scenario == "periodic":
                val = 200 + 150 * np.sin((2 * np.pi * t / 50) + phase)

            else:
                base = 100
                periodic = amp * np.sin((freq * t) + phase)
                noise = self.rng.normal(0, 10)
                burst = self.rng.uniform(100, 200) if self.rng.random() < 0.05 else 0

                val = max(0, base + periodic + noise + burst)

            seq.append(val)

        return seq

    def get(self, t):
        """Returns lambda at timestep t, clamped to the last value if t overflows."""
        if t < len(self.sequence):
            return self.sequence[t]
        return self.sequence[-1]