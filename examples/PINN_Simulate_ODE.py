from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def make_mlp(hidden: int = 48, depth: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


class ScalarPINN(nn.Module):
    def __init__(self, t_end: float, y0: float, hidden: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.t_end = float(t_end)
        self.y0 = float(y0)
        self.net = make_mlp(hidden=hidden, depth=depth)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tau = 2.0 * t / self.t_end - 1.0
        return self.y0 + t * self.net(tau)


def rhs_numpy(y: float, t: float) -> float:
    return -0.8 * y + 0.5 * np.sin(2.0 * t)


def rhs_torch(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -0.8 * y + 0.5 * torch.sin(2.0 * t)


def rk4(y0: float, time_grid: np.ndarray) -> np.ndarray:
    y = np.zeros_like(time_grid)
    y[0] = y0
    for i in range(time_grid.size - 1):
        t = time_grid[i]
        h = time_grid[i + 1] - time_grid[i]
        yi = y[i]
        k1 = rhs_numpy(yi, t)
        k2 = rhs_numpy(yi + 0.5 * h * k1, t + 0.5 * h)
        k3 = rhs_numpy(yi + 0.5 * h * k2, t + 0.5 * h)
        k4 = rhs_numpy(yi + h * k3, t + h)
        y[i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results" / "pinn_simulate_ode"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_end = 8.0
    dt = 0.02
    y0 = 1.2
    time_grid = np.arange(0.0, t_end + dt, dt, dtype=np.float32)
    y_true = rk4(y0, time_grid).astype(np.float32)

    model = ScalarPINN(t_end=t_end, y0=y0, hidden=48, depth=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    steps = 1200
    n_phys = 256
    n_data = 64
    w_phys = 1.0
    w_data = 20.0
    w_ic = 50.0

    data_idx = np.linspace(0, len(time_grid) - 1, n_data, dtype=int)
    t_data_np = time_grid[data_idx][:, None]
    y_data_np = y_true[data_idx][:, None]
    t_data = torch.tensor(t_data_np, dtype=torch.float32, device=device)
    y_data = torch.tensor(y_data_np, dtype=torch.float32, device=device)

    history = {"step": [], "loss": [], "physics": [], "data": [], "initial": []}

    for step in range(steps + 1):
        optimizer.zero_grad()

        t_phys = torch.rand(n_phys, 1, device=device) * t_end
        t_phys.requires_grad_(True)
        y_phys = model(t_phys)
        dy_dt = torch.autograd.grad(y_phys.sum(), t_phys, create_graph=True)[0]
        loss_phys = torch.mean((dy_dt - rhs_torch(y_phys, t_phys)) ** 2)

        y_pred_data = model(t_data)
        loss_data = torch.mean((y_pred_data - y_data) ** 2)

        t0 = torch.zeros(64, 1, device=device)
        y0_pred = model(t0)
        loss_ic = torch.mean((y0_pred - y0) ** 2)

        loss = w_phys * loss_phys + w_data * loss_data + w_ic * loss_ic
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == steps:
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["physics"].append(float(loss_phys.item()))
            history["data"].append(float(loss_data.item()))
            history["initial"].append(float(loss_ic.item()))

    t_eval = torch.tensor(time_grid[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pinn = model(t_eval).cpu().numpy().reshape(-1)

    mse = float(np.mean((y_pinn - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pinn - y_true)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    axes[0].plot(time_grid, y_true, color="#1f4e79", linewidth=1.8, label="RK4")
    axes[0].plot(time_grid, y_pinn, color="#b22222", linewidth=1.5, linestyle="--", label="PINN")
    axes[0].scatter(t_data_np[:, 0], y_data_np[:, 0], s=10, color="black", alpha=0.45, label="Data")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$y(t)$")
    axes[0].legend(frameon=False)
    axes[0].text(-0.15, 0.92, "(a)", transform=axes[0].transAxes)

    axes[1].plot(history["step"], history["loss"], color="black", label="Total")
    axes[1].plot(history["step"], history["physics"], color="#377eb8", label="Physics")
    axes[1].plot(history["step"], history["data"], color="#e41a1c", label="Data")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].text(-0.15, 0.92, "(b)", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(output_dir / "ode_fit_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ode_fit_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez(output_dir / "ode_fit_results.npz", time=time_grid, y_true=y_true, y_pinn=y_pinn)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"mse": mse, "mae": mae}, handle, indent=2)

    print(f"Saved ODE demo results to {output_dir}")
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")


if __name__ == "__main__":
    main()
