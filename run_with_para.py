from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinn_kuramoto import run_pinn_kuramoto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PINN synchronization-control example for the Kuramoto model."
    )
    parser.add_argument(
        "--config",
        default="configs/kuramoto_n10.json",
        help="Path to a JSON configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    run_pinn_kuramoto(config)


if __name__ == "__main__":
    main()
