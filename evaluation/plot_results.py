import matplotlib.pyplot as plt


def plot_instances_vs_workload(ppo_result, dqn_result, baseline_result):
    """
    Plots one episode's workload (raw lambda) against each agent's
    instance-count decisions over time.

    """

    workload = [step["requests"] for step in ppo_result["history"]]

    ppo_instances = [step["instances"] for step in ppo_result["history"]]
    dqn_instances = [step["instances"] for step in dqn_result["history"]]
    baseline_instances = [step["instances"] for step in baseline_result["history"]]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(workload, label="Workload (\u03bb)", linestyle="--",
              linewidth=2, color="tab:blue", alpha=0.6)
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Workload (\u03bb)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ppo_instances, label="PPO", linewidth=2, color="tab:orange")
    ax2.plot(dqn_instances, label="DQN", linewidth=2, color="tab:green")
    ax2.plot(baseline_instances, label="Baseline", linewidth=2, color="tab:red")
    ax2.set_ylabel("Instances (N)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Workload vs Scaling Decisions")
    ax1.grid()

    plt.tight_layout()
    plt.show()