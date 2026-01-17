#!/usr/bin/env python3
"""
Plot throughput metrics for continuum and vllm runs across batch sizes.

Generates a 2x2 grid: Tasks/min and Steps/min for continuum and vllm.
"""
import matplotlib.pyplot as plt

# Data
batch_sizes = [24, 32, 48, 64, 72, 96, 120]
tasks = [154.7, 175.2, 190.4, 194.0, 187.8, 186.6, 189.8]



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
    span = max(batch_sizes) - min(batch_sizes)
    pad = span * 0.05
    ax.set_xlim(min(batch_sizes) - pad, max(batch_sizes) + pad)
    ax.legend()


def main():
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

    """# Continuum Tasks/min
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
    plot_lines(axes[1, 1], vllm_steps, "Steps/min")"""


    axes.set_title("ThunderReact: Steps/min", fontsize=12, weight="bold")
    axes.set_xlabel("Batch size")
    axes.set_ylabel("Steps/min")
    plot_lines(axes, tasks, "Steps/min")

    fig.suptitle("Throughput vs Batch Size", fontsize=14, weight="bold")
    fig.savefig("metrics.png", dpi=180)
    print("Saved metrics.png")


if __name__ == "__main__":
    main()
