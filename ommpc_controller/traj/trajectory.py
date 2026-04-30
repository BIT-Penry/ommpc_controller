#!/usr/bin/env python3

from pathlib import Path
import argparse
import math


WAYPOINT_TAUS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a waypoint-constrained polynomial trajectory txt file."
    )
    parser.add_argument("--max-speed", type=float, default=1.0, help="Maximum speed in m/s.")
    parser.add_argument("--height", type=float, default=1.0, help="Constant flight height in m.")
    parser.add_argument("--sample-time", type=float, default=0.01, help="Sampling time in s.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Start x position in m.")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start y position in m.")
    parser.add_argument("--via1-x", type=float, default=1.0, help="First waypoint x position in m.")
    parser.add_argument("--via1-y", type=float, default=0.25, help="First waypoint y position in m.")
    parser.add_argument("--via2-x", type=float, default=2.0, help="Second waypoint x position in m.")
    parser.add_argument("--via2-y", type=float, default=-0.25, help="Second waypoint y position in m.")
    parser.add_argument("--end-x", type=float, default=3.0, help="End x position in m.")
    parser.add_argument("--end-y", type=float, default=0.0, help="End y position in m.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Reference yaw in rad.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("trajectory.txt"),
        help="Output txt path.",
    )
    return parser.parse_args()

def solve_linear_system(matrix, vector):
    n = len(vector)
    aug = [list(row) + [value] for row, value in zip(matrix, vector)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            raise ValueError("Polynomial constraint matrix is singular.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1.0e-15:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def poly_row(tau, derivative):
    row = []
    for power in range(8):
        if derivative == 0:
            row.append(tau**power)
        elif derivative == 1:
            row.append(0.0 if power == 0 else power * tau ** (power - 1))
        elif derivative == 2:
            row.append(
                0.0
                if power < 2
                else power * (power - 1) * tau ** (power - 2)
            )
        else:
            raise ValueError("Only derivatives 0, 1, and 2 are supported.")
    return row


def fit_axis(values):
    matrix = []
    rhs = []

    for tau, value in zip(WAYPOINT_TAUS, values):
        matrix.append(poly_row(tau, 0))
        rhs.append(value)

    matrix.append(poly_row(0.0, 1))
    rhs.append(0.0)
    matrix.append(poly_row(1.0, 1))
    rhs.append(0.0)
    matrix.append(poly_row(0.0, 2))
    rhs.append(0.0)
    matrix.append(poly_row(1.0, 2))
    rhs.append(0.0)

    return solve_linear_system(matrix, rhs)


def eval_poly(coeffs, tau, derivative=0):
    value = 0.0
    for power, coeff in enumerate(coeffs):
        if derivative == 0:
            value += coeff * tau**power
        elif derivative == 1 and power >= 1:
            value += coeff * power * tau ** (power - 1)
        elif derivative == 2 and power >= 2:
            value += coeff * power * (power - 1) * tau ** (power - 2)
    return value


def generate_trajectory(
    start_xy,
    via1_xy,
    via2_xy,
    end_xy,
    height,
    max_speed,
    sample_time,
    yaw,
):
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive.")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive.")

    x_coeffs = fit_axis([start_xy[0], via1_xy[0], via2_xy[0], end_xy[0]])
    y_coeffs = fit_axis([start_xy[1], via1_xy[1], via2_xy[1], end_xy[1]])
    z_coeffs = fit_axis([height, height, height, height])

    max_norm_speed = 0.0
    for i in range(20001):
        tau = i / 20000.0
        dx_dtau = eval_poly(x_coeffs, tau, 1)
        dy_dtau = eval_poly(y_coeffs, tau, 1)
        dz_dtau = eval_poly(z_coeffs, tau, 1)
        speed = math.sqrt(dx_dtau * dx_dtau + dy_dtau * dy_dtau + dz_dtau * dz_dtau)
        max_norm_speed = max(max_norm_speed, speed)

    duration_min = max_norm_speed / max_speed
    num_intervals = max(3, int(math.ceil(duration_min / sample_time)))
    remainder = num_intervals % 3
    if remainder:
        num_intervals += 3 - remainder

    duration = num_intervals * sample_time
    yaw_rate = 0.0
    rows = []

    for i in range(num_intervals + 1):
        tau = i / float(num_intervals)
        x = eval_poly(x_coeffs, tau, 0)
        y = eval_poly(y_coeffs, tau, 0)
        z = eval_poly(z_coeffs, tau, 0)
        vx = eval_poly(x_coeffs, tau, 1) / duration
        vy = eval_poly(y_coeffs, tau, 1) / duration
        vz = eval_poly(z_coeffs, tau, 1) / duration
        rows.append((x, y, z, vx, vy, vz, yaw, yaw_rate))

    return duration, rows


def main():
    args = parse_args()
    duration, traj = generate_trajectory(
        start_xy=(args.start_x, args.start_y),
        via1_xy=(args.via1_x, args.via1_y),
        via2_xy=(args.via2_x, args.via2_y),
        end_xy=(args.end_x, args.end_y),
        height=args.height,
        max_speed=args.max_speed,
        sample_time=args.sample_time,
        yaw=args.yaw,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in traj:
            f.write(" ".join(f"{value:.6f}" for value in row) + "\n")

    print(
        f"Saved {args.output}, N = {len(traj)}, duration = {duration:.3f}s, "
        f"max_speed <= {args.max_speed:.3f}m/s"
    )


if __name__ == "__main__":
    main()
