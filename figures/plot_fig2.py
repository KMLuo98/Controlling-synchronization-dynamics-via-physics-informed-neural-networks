from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the paper-style summary figure from saved simulation outputs.")
    parser.add_argument("--config", default="configs/kuramoto_n10.json", help="Path to the JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    data_dir = Path("results/fig2")
    control_dir = Path("results/fig3")
    target_start = float(config["target_start"])
    target_order = float(config["target_order"])

    theta_no_control = np.loadtxt(data_dir / "theta_no_control.txt")
    theta_with_control = np.loadtxt(data_dir / "theta_with_control.txt")
    r_no_control = np.loadtxt(data_dir / "R_no_control.txt")
    r_with_control = np.loadtxt(data_dir / "R_with_control.txt")
    time_series = np.loadtxt(data_dir / "time_series.txt")
    control = np.loadtxt(control_dir / "control_signals.txt")

    sin_theta_no = np.sin(theta_no_control)
    sin_theta_ctrl = np.sin(theta_with_control)
    n = sin_theta_no.shape[1]
    colors = cm.viridis(np.linspace(0.05, 0.95, n))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.2,
            "axes.linewidth": 1.0,
        }
    )

    fig = plt.figure(figsize=(6.8, 4.6))
    grid = gridspec.GridSpec(3, 2, width_ratios=[0.9, 1.05], height_ratios=[1.0, 1.0, 0.9], wspace=0.28, hspace=0.18)

    ax1 = fig.add_subplot(grid[0, 0])
    for i in range(n):
        ax1.plot(time_series, sin_theta_no[:, i], color=colors[i], alpha=0.7)
    ax1.set_ylabel(r"$\sin(\theta_i)$")
    ax1.set_ylim([-1.2, 1.2])
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.text(-0.18, 0.88, "(a)", transform=ax1.transAxes)

    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    for i in range(n):
        ax2.plot(time_series, sin_theta_ctrl[:, i], color=colors[i], alpha=0.7)
    ax2.set_ylabel(r"$\sin(\theta_i)$")
    ax2.set_ylim([-1.2, 1.2])
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.text(-0.18, 0.88, "(b)", transform=ax2.transAxes)

    ax3 = fig.add_subplot(grid[2, 0], sharex=ax1)
    for i in range(control.shape[1]):
        ax3.plot(time_series, control[:, i], color=colors[i], alpha=0.7)
    ax3.set_ylabel(r"$u_i$")
    ax3.set_xlabel(r"$t$")
    ax3.text(-0.18, 0.84, "(c)", transform=ax3.transAxes)

    ax4 = fig.add_subplot(grid[0:2, 1])
    ax4.plot(time_series, r_with_control, color="#b22222", linewidth=1.8, label="With control")
    ax4.plot(time_series, r_no_control, color="#1f4e79", linewidth=1.8, label="Without control")
    ax4.axvspan(target_start, time_series[-1], color="#f0c808", alpha=0.12)
    ax4.axhline(target_order, color="gray", linestyle="--", linewidth=1.0)
    ax4.set_ylabel(r"$R(t)$")
    ax4.set_ylim([-0.02, 1.05])
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.legend(frameon=False, loc="lower center")
    ax4.text(-0.14, 0.94, "(d)", transform=ax4.transAxes)

    ax5 = fig.add_subplot(grid[2, 1], sharex=ax4)
    delta = np.abs(r_with_control - r_no_control)
    ax5.plot(time_series, delta, color="black", linewidth=1.4)
    ax5.set_xlabel(r"$t$")
    ax5.set_ylabel(r"$|R_c-R_f|$")
    ax5.text(-0.14, 0.94, "(e)", transform=ax5.transAxes)

    output_dir = Path("figures/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure2_paper_style.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure2_paper_style.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_dir}")


if __name__ == "__main__":
    main()
