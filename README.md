# Controlling Synchronization Dynamics via Physics-Informed Neural Networks

Code release for the paper **Controlling synchronization dynamics via physics-informed neural networks**.

This repository contains:

- a PINN-based control example for the Kuramoto model
- scripts to export the data used for paper-style figures
- a standalone PINN dynamics-fitting demo in [examples/PINN_Simulate_ODE.py](/C:/Users/Administrator/Desktop/PINN/examples/PINN_Simulate_ODE.py)

## Repository Layout

- [pinn_kuramoto.py](/C:/Users/Administrator/Desktop/PINN/pinn_kuramoto.py): core training and validation routine
- [run_with_para.py](/C:/Users/Administrator/Desktop/PINN/run_with_para.py): command-line entry point
- [configs/kuramoto_n10.json](/C:/Users/Administrator/Desktop/PINN/configs/kuramoto_n10.json): default configuration
- [figures/plot_fig2.py](/C:/Users/Administrator/Desktop/PINN/figures/plot_fig2.py): plotting script for a publication-style summary figure
- [examples/PINN_Simulate_ODE.py](/C:/Users/Administrator/Desktop/PINN/examples/PINN_Simulate_ODE.py): PINN fitting demo for a scalar ODE
- [docs/hyperparameter_recommendations.md](/C:/Users/Administrator/Desktop/PINN/docs/hyperparameter_recommendations.md): practical parameter suggestions

## Environment

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the Main Example

```bash
python run_with_para.py --config configs/kuramoto_n10.json
```

This writes:

- `results/fig2`
- `results/fig3`

Then generate the summary figure:

```bash
python figures/plot_fig2.py --config configs/kuramoto_n10.json
```

## Run the ODE Fitting Demo

```bash
python examples/PINN_Simulate_ODE.py
```

This writes results under `examples/results/pinn_simulate_ode`.

## Notes for Reproducibility

- The synchronization target is specified only on the interval `[t*, T]`.
- The default release uses `N=10`.
- The code is intended as a concise research reproduction package rather than a full software framework.

## Citation

If you use this code, please cite the paper and this software release. A machine-readable citation file is provided in [CITATION.cff](/C:/Users/Administrator/Desktop/PINN/CITATION.cff).

