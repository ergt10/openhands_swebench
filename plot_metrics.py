#!/usr/bin/env python3
"""
Plot throughput metrics for continuum and vllm runs across batch sizes.

Generates a 2x2 grid: Tasks/min and Steps/min for continuum and vllm.
"""
import matplotlib.pyplot as plt

# Data
batch_sizes = [32, 48, 64, 96]
continuum_tasks = [4.000, 4.758, 3.725, 2.049]
continuum_steps = [205.851, 238.758, 192.800, 112.083]
vllm_tasks = [4.569, 5.843, 2.085, 1.719]
vllm_steps = [227.231, 280.510, 110.366, 83.883]


def make_axes(title, ylabel):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Batch size", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


def plot_lines(ax, series, label):
    ax.plot(batch_sizes, series, marker="o", linewidth=2.2, markersize=7, label=label)
    ax.fill_between(batch_sizes, series, color=ax.lines[-1].get_color(), alpha=0.1)
    ax.set_xticks(batch_sizes)
    ax.legend()


def main():
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # Continuum Tasks/min
    axes[0, 0].set_title("Continuum: Tasks/min", fontsize=12, weight="bold")
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("Tasks/min")
    plot_lines(axes[0, 0], continuum_tasks, "Tasks/min")

    # Continuum Steps/min
    axes[0, 1].set_title("Continuum: Steps/min", fontsize=12, weight="bold")
    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("Steps/min")
    plot_lines(axes[0, 1], continuum_steps, "Steps/min")

    # vLLM Tasks/min
    axes[1, 0].set_title("vLLM: Tasks/min", fontsize=12, weight="bold")
    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("Tasks/min")
    plot_lines(axes[1, 0], vllm_tasks, "Tasks/min")

    # vLLM Steps/min
    axes[1, 1].set_title("vLLM: Steps/min", fontsize=12, weight="bold")
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("Steps/min")
    plot_lines(axes[1, 1], vllm_steps, "Steps/min")

    fig.suptitle("Throughput vs Batch Size", fontsize=14, weight="bold")
    fig.savefig("metrics.png", dpi=180)
    print("Saved metrics.png")


if __name__ == "__main__":
    main()
