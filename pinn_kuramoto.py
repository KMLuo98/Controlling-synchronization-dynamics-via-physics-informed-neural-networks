from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


def _order_parameter_np(theta_np: np.ndarray) -> np.ndarray:
    c = np.mean(np.cos(theta_np), axis=1)
    s = np.mean(np.sin(theta_np), axis=1)
    return np.sqrt(c * c + s * s + 1e-12)


def _make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


def run_pinn_kuramoto(config: dict) -> dict[str, float]:
    n = int(config["N"])
    horizon = float(config["T"])
    coupling = float(config["K"])
    seed = int(config["SEED"])
    dt = float(config["dt"])

    hidden = int(config["HIDDEN"])
    depth = int(config["DEPTH"])
    umax = float(config["UMAX"])

    steps = int(config["steps"])
    lr = float(config["lr"])
    n_f = int(config["n_f"])

    target_start = float(config["target_start"])
    target_order = float(config["target_order"])

    w_r = float(config["w_R"])
    w_u = float(config["w_u"])
    w_phys = float(config["w_phys"])

    output_dir_fig2 = Path(config["output_dir_fig2"])
    output_dir_fig3 = Path(config["output_dir_fig3"])

    np.random.seed(seed)
    torch.manual_seed(seed)

    omega = np.random.uniform(-np.pi, np.pi, n).astype(np.float32)
    adjacency = np.asarray(nx.adjacency_matrix(nx.complete_graph(n)).todense(), dtype=np.float32)
    theta0 = np.random.uniform(-np.pi / 2, np.pi / 2, n).astype(np.float32)
    time_grid = np.arange(0.0, horizon + dt, dt, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"N={n}, T={horizon}, K={coupling}, target window=[{target_start}, {horizon}], target R={target_order}")

    omega_t = torch.tensor(omega, dtype=torch.float32, device=device)
    theta0_t = torch.tensor(theta0, dtype=torch.float32, device=device)

    def shaping_function(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-t)

    def order_parameter_torch(theta: torch.Tensor) -> torch.Tensor:
        c = torch.mean(torch.cos(theta), dim=1)
        s = torch.mean(torch.sin(theta), dim=1)
        return torch.sqrt(c * c + s * s + 1e-12)

    class ThetaNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = _make_mlp(1, n, hidden, depth)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            tau = 2.0 * t / horizon - 1.0
            h = shaping_function(t)
            return theta0_t + h * self.net(tau)

    class ControlNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = _make_mlp(1, n, hidden, depth)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            tau = 2.0 * t / horizon - 1.0
            h = shaping_function(t)
            return h * umax * torch.tanh(self.net(tau))

    theta_model = ThetaNet().to(device)
    u_model = ControlNet().to(device)
    optimizer = torch.optim.Adam(list(theta_model.parameters()) + list(u_model.parameters()), lr=lr)

    def physics_residual(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t.requires_grad_(True)
        theta = theta_model(t)
        u = u_model(t)

        grads = []
        for i in range(n):
            grad_i = torch.autograd.grad(theta[:, i].sum(), t, create_graph=True)[0]
            grads.append(grad_i)
        dtheta = torch.cat(grads, dim=1)

        rhs_terms = []
        for i in range(n):
            coupling_sum = 0.0
            for j in range(n):
                if adjacency[i, j]:
                    coupling_sum = coupling_sum + torch.sin(theta[:, j] - theta[:, i])
            rhs_terms.append(omega_t[i] + coupling * coupling_sum)
        rhs = torch.stack(rhs_terms, dim=1)
        return dtheta - (rhs + u), u

    def target_window_loss(t_samples: torch.Tensor) -> torch.Tensor:
        theta = theta_model(t_samples)
        order = order_parameter_torch(theta)
        mask = t_samples[:, 0] >= target_start
        if torch.any(mask):
            return torch.mean((order[mask] - target_order) ** 2)
        return torch.tensor(0.0, device=device)

    history: dict[str, list[float]] = {"step": [], "loss": [], "physics": [], "target": [], "control": []}

    for it in range(steps):
        optimizer.zero_grad()

        tf = torch.rand(n_f, 1, device=device) * horizon
        residual, u_f = physics_residual(tf)
        l_phys = torch.mean(residual ** 2)

        te = torch.rand(n_f, 1, device=device) * horizon
        l_r = target_window_loss(te)

        l_u = torch.mean(torch.mean(u_f ** 2, dim=1))

        loss = w_phys * l_phys + w_r * l_r + w_u * l_u
        loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == steps - 1:
            with torch.no_grad():
                probe_t = torch.tensor([[target_start], [0.5 * (target_start + horizon)], [horizon]], dtype=torch.float32, device=device)
                probe_order = order_parameter_torch(theta_model(probe_t)).cpu().numpy()
                print(
                    f"[{it:4d}] loss={loss.item():.3e} "
                    f"L_phys={l_phys.item():.2e} L_R={l_r.item():.2e} "
                    f"u_rms={torch.sqrt(l_u).item():.4f} "
                    f"R(t*)={probe_order[0]:.3f} R(mid)={probe_order[1]:.3f} R(T)={probe_order[2]:.3f}"
                )
                history["step"].append(float(it))
                history["loss"].append(float(loss.item()))
                history["physics"].append(float(l_phys.item()))
                history["target"].append(float(l_r.item()))
                history["control"].append(float(l_u.item()))

    t_eval = torch.tensor(time_grid[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        theta_pinn = theta_model(t_eval).cpu().numpy()
        u_pinn = u_model(t_eval).cpu().numpy()

    def u_interp(t: float) -> np.ndarray:
        if t <= time_grid[0]:
            return u_pinn[0]
        if t >= time_grid[-1]:
            return u_pinn[-1]
        k = int((t - time_grid[0]) / dt)
        k = min(k, len(time_grid) - 2)
        t0_ = time_grid[k]
        t1_ = time_grid[k + 1]
        weight = (t - t0_) / (t1_ - t0_)
        return (1.0 - weight) * u_pinn[k] + weight * u_pinn[k + 1]

    def kuramoto_rhs(theta: np.ndarray, t: float, use_control: bool = True) -> np.ndarray:
        rhs = np.zeros_like(theta, dtype=np.float64)
        for i in range(n):
            coupling_sum = 0.0
            for j in range(n):
                if adjacency[i, j]:
                    coupling_sum += np.sin(theta[j] - theta[i])
            rhs[i] = omega[i] + coupling * coupling_sum
        if use_control:
            rhs = rhs + u_interp(t)
        return rhs

    def rk4_integrate(theta_init: np.ndarray, use_control: bool = True) -> np.ndarray:
        theta = np.zeros((len(time_grid), n), dtype=np.float64)
        theta[0] = theta_init.astype(np.float64)
        for idx in range(len(time_grid) - 1):
            t = time_grid[idx]
            h = dt
            y = theta[idx]
            k1 = kuramoto_rhs(y, t, use_control=use_control)
            k2 = kuramoto_rhs(y + 0.5 * h * k1, t + 0.5 * h, use_control=use_control)
            k3 = kuramoto_rhs(y + 0.5 * h * k2, t + 0.5 * h, use_control=use_control)
            k4 = kuramoto_rhs(y + h * k3, t + h, use_control=use_control)
            theta[idx + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return theta

    theta_rk4_ctrl = rk4_integrate(theta0, use_control=True)
    theta_rk4_free = rk4_integrate(theta0, use_control=False)

    r_pinn = _order_parameter_np(theta_pinn)
    r_rk4_ctrl = _order_parameter_np(theta_rk4_ctrl)
    r_rk4_free = _order_parameter_np(theta_rk4_free)

    window_mask = time_grid >= target_start
    print("\n===== Target-window statistics =====")
    print(
        f"PINN target-window mean R={r_pinn[window_mask].mean():.4f}, "
        f"RK4+u mean R={r_rk4_ctrl[window_mask].mean():.4f}, "
        f"RK4 free mean R={r_rk4_free[window_mask].mean():.4f}"
    )

    output_dir_fig2.mkdir(parents=True, exist_ok=True)
    output_dir_fig3.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir_fig2 / "theta_no_control.txt", theta_rk4_free)
    np.savetxt(output_dir_fig2 / "theta_with_control.txt", theta_rk4_ctrl)
    np.savetxt(output_dir_fig2 / "R_no_control.txt", r_rk4_free)
    np.savetxt(output_dir_fig2 / "R_with_control.txt", r_rk4_ctrl)
    np.savetxt(output_dir_fig2 / "R_pinn.txt", r_pinn)
    np.savetxt(output_dir_fig2 / "time_series.txt", time_grid)

    np.savetxt(output_dir_fig3 / "control_signals.txt", u_pinn)

    u_squared_sum = np.sum(u_pinn ** 2, axis=1)
    e_cumulative = np.zeros_like(time_grid)
    for idx in range(1, len(time_grid)):
        e_cumulative[idx] = np.trapezoid(u_squared_sum[: idx + 1], time_grid[: idx + 1])
    e_total = float(e_cumulative[-1])

    np.savetxt(output_dir_fig3 / "energy_time_series.txt", e_cumulative)
    np.savetxt(output_dir_fig3 / "energy_density.txt", u_squared_sum)
    with (output_dir_fig3 / "total_energy.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"E = {e_total:.10f}\n")

    metrics = {
        "target_window_mean_R_pinn": float(r_pinn[window_mask].mean()),
        "target_window_mean_R_rk4_controlled": float(r_rk4_ctrl[window_mask].mean()),
        "target_window_mean_R_rk4_free": float(r_rk4_free[window_mask].mean()),
        "final_R_controlled": float(r_rk4_ctrl[-1]),
        "final_R_free": float(r_rk4_free[-1]),
        "total_control_energy": e_total,
    }

    with (output_dir_fig3 / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Fig. 2 data saved to {output_dir_fig2}")
    print(f"Fig. 3 data saved to {output_dir_fig3}")
    print(f"Total control energy: E = {e_total:.10f}")
    return metrics
