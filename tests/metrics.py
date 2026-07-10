# Turns raw per-step info history into per-episode summary metrics, then
# aggregates those summaries across multiple seeds (mean + std).


import numpy as np

def compute_metrics(data):
    """
    data: list of per-step info dicts (each from CloudEnv.step()).

    cost is summed (total resource cost over the episode), while sla_violation/latency/utilization are averaged per step 
    -- this matches how the original metrics were defined, but note cost will scale with episode length,
    so only compare cost across equal-length episodes (all scenarios here use the same workload length).
    """
    cost = np.sum([d.get("cost", 0) for d in data])

    error_rates = [d.get("error_rate", 0) for d in data]
    sla_violation = np.mean(error_rates)

    latency = np.mean([d.get("response_time", 0) for d in data])
    utilization = np.mean([d.get("utilization", 0) for d in data])

    return {
        "cost": cost,
        "sla_violation": sla_violation,
        "latency": latency,
        "utilization": utilization,
    }


def aggregate(results):
    """
    results: list of compute_metrics() outputs (one per seed).
    Returns mean/std for each metric across all seeds.
    """
    final = {}

    for key in results[0]:
        values = [r[key] for r in results]

        final[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return final