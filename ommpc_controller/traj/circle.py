# --------------------------------------
# Generate smooth circular reference trajectory
# --------------------------------------

from pathlib import Path
import math


# Parameters
sample_time = 0.01  # seconds
duration = 15.0  # seconds for one full circle

r = 0.6
z0 = 1.0

# Circle center. This keeps the old start point at (0, 0, z0).
x0 = 0.0
y0 = -r

clockwise = True
dir_sign = -1.0 if clockwise else 1.0

# Time vector, including the final endpoint.
num_steps = int(round(duration / sample_time))
times = [i * sample_time for i in range(num_steps)]
times.append(duration)

# Start from (0, 0, z0) while keeping (x0, y0, z0) as the circle center.
start_x = 0.0
start_y = 0.0
center_to_start = math.hypot(start_x - x0, start_y - y0)
if not math.isclose(center_to_start, r, rel_tol=1e-6, abs_tol=1e-6):
    raise ValueError(
        f"Start point ({start_x}, {start_y}) is not on the circle centered at "
        f"({x0}, {y0}) with radius {r}. Distance is {center_to_start:.6f}."
    )

theta0 = math.atan2(start_y - y0, start_x - x0)
theta_total = dir_sign * 2.0 * math.pi
traj = []

for t in times:
    tau = t / duration

    # 5th-order time scaling. This gives zero velocity at the beginning and end,
    # avoiding the hard velocity jump from the old constant-speed circle.
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    sdot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration

    theta = theta0 + theta_total * s
    theta_dot = theta_total * sdot

    x = x0 + r * math.cos(theta)
    y = y0 + r * math.sin(theta)
    z = z0

    # Velocity from chain rule.
    vx = -r * math.sin(theta) * theta_dot
    vy = r * math.cos(theta) * theta_dot
    vz = 0.0

    yaw = 0.0
    yaw_rate = 0.0
    traj.append((x, y, z, vx, vy, vz, yaw, yaw_rate))

output_path = Path(__file__).with_name("circle.txt")
with output_path.open("w") as f:
    for row in traj:
        f.write(" ".join(f"{value:.6f}" for value in row) + "\n")

print(f"Saved {output_path}, N = {len(traj)}")
