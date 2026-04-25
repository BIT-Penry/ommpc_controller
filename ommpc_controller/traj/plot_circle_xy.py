#!/usr/bin/env python3
"""Plot the x-y trajectory stored in a txt reference file."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_txt = script_dir / "circle.txt"

    parser = argparse.ArgumentParser(
        description="Visualize the x-y trajectory from a txt reference file."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=default_txt,
        help=f"Trajectory txt file to plot (default: {default_txt})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of only showing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = args.file.expanduser().resolve()

    data = np.loadtxt(traj_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"{traj_path} does not contain x/y columns.")

    x = data[:, 0]
    y = data[:, 1]

    center_x = 0.5 * (np.min(x) + np.max(x))
    center_y = 0.5 * (np.min(y) + np.max(y))

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="trajectory", linewidth=2)
    plt.scatter(x[0], y[0], color="green", label="start", zorder=3)
    plt.scatter(x[-1], y[-1], color="red", label="end", zorder=3)
    plt.scatter(center_x, center_y, color="orange", marker="x", s=80, label="center")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"x-y trajectory: {traj_path.name}")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if args.save is not None:
        save_path = args.save.expanduser().resolve()
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
