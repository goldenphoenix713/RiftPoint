"""Plotting and visualization generator for RiftPoint benchmark results.

Generates high-resolution publication-quality PNG and SVG benchmark figures
for documentation and performance reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _apply_plot_theme() -> None:
    """Apply modern high-contrast design styling to matplotlib plots."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.dpi": 300,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )


def generate_forking_speedup_chart(
    fork_results: list[dict[str, Any]],
    output_path: Path | str = "docs/images/benchmark_forking_speedup.png",
) -> Path:
    """Generate side-by-side plots of branch forking latency and speedup multipliers.

    Args:
        fork_results: List of result dicts across payload sizes.
        output_path: Destination image path.

    Returns:
        Path to saved figure.
    """
    _apply_plot_theme()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    payload_labels = [f"{r['payload_kb']} KB" for r in fork_results]
    std_times = [r["std_ms"] for r in fork_results]
    rift_times = [r["rift_ms"] for r in fork_results]
    speedups = [r["speedup"] for r in fork_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Subplot 1: Latency Comparison (Log-scale) ---
    x_indices = range(len(payload_labels))
    width = 0.35

    bars1 = ax1.bar(
        [x - width / 2 for x in x_indices],
        std_times,
        width,
        label="Standard LangGraph (deepcopy)",
        color="#f43f5e",
        edgecolor="#be123c",
        alpha=0.9,
    )
    bars2 = ax1.bar(
        [x + width / 2 for x in x_indices],
        rift_times,
        width,
        label="RiftPoint (CoW Pointer Sharing)",
        color="#6366f1",
        edgecolor="#4338ca",
        alpha=0.95,
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("State Payload Size")
    ax1.set_ylabel("Forking Latency for 100 Branches (ms, Log Scale)")
    ax1.set_title(
        "Branch Forking Latency: O(1) CoW vs O(N·S) Deepcopy", fontweight="bold"
    )
    ax1.set_xticks(list(x_indices))
    ax1.set_xticklabels(payload_labels)
    ax1.legend(loc="upper left", frameon=True)

    # Annotate values
    for bar, val in zip(bars1, std_times, strict=True):
        ax1.annotate(
            f"{val:.1f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#be123c",
        )
    for bar, val in zip(bars2, rift_times, strict=True):
        ax1.annotate(
            f"{val:.2f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#4338ca",
        )

    # --- Subplot 2: Speedup Factor Curve ---
    bar_colors = ["#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"]
    bars_speed = ax2.bar(
        payload_labels,
        speedups,
        color=bar_colors,
        edgecolor="#3730a3",
        width=0.55,
    )

    ax2.set_xlabel("State Payload Size")
    ax2.set_ylabel("RiftPoint Speedup Multiplier (x)")
    ax2.set_title("RiftPoint Speedup Factor vs Context Size", fontweight="bold")

    for bar, sp in zip(bars_speed, speedups, strict=True):
        ax2.annotate(
            f"{sp:.1f}x\nfaster",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#312e81",
        )

    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p


def generate_micro_latency_chart(
    mem_put: tuple[float, dict[str, float]],
    rift_put: tuple[float, dict[str, float]],
    mem_get: tuple[float, dict[str, float]],
    rift_get: tuple[float, dict[str, float]],
    output_path: Path | str = "docs/images/benchmark_micro_latency.png",
) -> Path:
    """Generate bar chart comparing Put and Get latency percentiles in microseconds.

    Args:
        mem_put: MemorySaver put stats.
        rift_put: RiftPoint put stats.
        mem_get: MemorySaver get stats.
        rift_get: RiftPoint get stats.
        output_path: Destination image path.

    Returns:
        Path to saved figure.
    """
    _apply_plot_theme()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Mean", "p50 (Median)", "p95 Percentile"]
    mem_put_stats = [mem_put[1]["mean_us"], mem_put[1]["p50_us"], mem_put[1]["p95_us"]]
    rift_put_stats = [
        rift_put[1]["mean_us"],
        rift_put[1]["p50_us"],
        rift_put[1]["p95_us"],
    ]

    mem_get_stats = [mem_get[1]["mean_us"], mem_get[1]["p50_us"], mem_get[1]["p95_us"]]
    rift_get_stats = [
        rift_get[1]["mean_us"],
        rift_get[1]["p50_us"],
        rift_get[1]["p95_us"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(metrics))
    width = 0.35

    # Put Operations
    ax1.bar(
        [i - width / 2 for i in x],
        mem_put_stats,
        width,
        label="Standard MemorySaver",
        color="#94a3b8",
    )
    ax1.bar(
        [i + width / 2 for i in x],
        rift_put_stats,
        width,
        label="RiftCheckpointSaver",
        color="#6366f1",
    )
    ax1.set_ylabel("Latency (µs)")
    ax1.set_title("Checkpoint Put / Write Latency (N=1,000)", fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics)
    ax1.legend(loc="upper left")

    # Get Operations
    ax2.bar(
        [i - width / 2 for i in x],
        mem_get_stats,
        width,
        label="Standard MemorySaver",
        color="#94a3b8",
    )
    ax2.bar(
        [i + width / 2 for i in x],
        rift_get_stats,
        width,
        label="RiftCheckpointSaver",
        color="#6366f1",
    )
    ax2.set_ylabel("Latency (µs)")
    ax2.set_title("Checkpoint Get / Lookup Latency (N=1,000)", fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(metrics)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p


def generate_async_performance_chart(
    async_micro: tuple[
        tuple[float, dict[str, float]],
        tuple[float, dict[str, float]],
        tuple[float, dict[str, float]],
        tuple[float, dict[str, float]],
    ],
    async_scaling: list[dict[str, Any]],
    output_path: Path | str = "docs/images/benchmark_async_performance.png",
) -> Path:
    """Generate plots comparing asynchronous checkpoint throughput and branch scaling.

    Args:
        async_micro: Tuple of (mem_put, rift_put, mem_get, rift_get) stats.
        async_scaling: List of async parallel branch scaling result dicts.
        output_path: Destination image path.

    Returns:
        Path to saved figure.
    """
    _apply_plot_theme()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    mem_put, rift_put, mem_get, rift_get = async_micro
    ops_labels = ["Async Put\n(ops/sec)", "Async Get\n(ops/sec)"]
    mem_ops = [mem_put[0], mem_get[0]]
    rift_ops = [rift_put[0], rift_get[0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x_indices = range(len(ops_labels))
    width = 0.35

    # Subplot 1: Async Read/Write Throughput
    bars1 = ax1.bar(
        [x - width / 2 for x in x_indices],
        mem_ops,
        width,
        label="Standard MemorySaver (Async)",
        color="#94a3b8",
        edgecolor="#64748b",
        alpha=0.9,
    )
    bars2 = ax1.bar(
        [x + width / 2 for x in x_indices],
        rift_ops,
        width,
        label="AsyncRiftCheckpointSaver",
        color="#06b6d4",
        edgecolor="#0891b2",
        alpha=0.95,
    )

    ax1.set_ylabel("Throughput (Operations / Second)")
    ax1.set_title("Async Micro-Op Throughput (N=1,000 Ops)", fontweight="bold")
    ax1.set_xticks(list(x_indices))
    ax1.set_xticklabels(ops_labels)
    ax1.legend(loc="upper left")

    for bar, val in zip(bars1, mem_ops, strict=True):
        ax1.annotate(
            f"{val:,.0f} op/s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#475569",
        )
    for bar, val in zip(bars2, rift_ops, strict=True):
        ax1.annotate(
            f"{val:,.0f} op/s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0e7490",
        )

    # Subplot 2: Parallel Branch Scaling
    counts = [f"B={r['count']}\nBranches" for r in async_scaling]
    latencies = [r["ms"] for r in async_scaling]
    colors = ["#67e8f9", "#22d3ee", "#06b6d4", "#0891b2"]

    bars_sc = ax2.bar(
        counts,
        latencies,
        color=colors,
        edgecolor="#0e7490",
        width=0.55,
    )
    ax2.set_xlabel("Speculative Concurrency Level")
    ax2.set_ylabel("Total Latency (ms): Run + Score + Collapse + Prune")
    ax2.set_title(
        "Async Speculative Pipeline Latency (RiftRunner + Resolver)",
        fontweight="bold",
    )

    for bar, val in zip(bars_sc, latencies, strict=True):
        ax2.annotate(
            f"{val:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#155e75",
        )

    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p
