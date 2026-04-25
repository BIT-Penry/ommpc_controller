#--------------------------------------
#Generate reference trajectory
#--------------------------------------

import numpy as np
import math

# Parameters
sample_time = 0.01             # seconds
duration = 40                  # seconds

r = 0.6
T = 10
v = 2 * r * 3.1415926 / T

# Circle center
x0 = 0
y0 = -r
z0 = 1.0

clockwise = True
factor = -1 if clockwise else 1

# trajectory
traj = np.zeros((int(duration/sample_time+1),8))
t = np.arange(0,duration,sample_time)
t = np.append(t, duration)

# Start from (0, 0, z0) while keeping (x0, y0, z0) as the circle center.
start_x = 0.0
start_y = 0.0
center_to_start = math.hypot(start_x - x0, start_y - y0)
if not math.isclose(center_to_start, r, rel_tol=1e-6, abs_tol=1e-6):
    raise ValueError(
        f"Start point (0, 0) is not on the circle centered at ({x0}, {y0}) "
        f"with radius {r}. Distance to center is {center_to_start:.6f}."
    )

theta0 = math.atan2(start_y - y0, start_x - x0)
omega = factor * v / r
theta = theta0 + omega * t

traj[:,0] = x0 + r * np.cos(theta)
traj[:,1] = y0 + r * np.sin(theta)
traj[:,2] = z0
traj[:,3] = -r * np.sin(theta) * omega
traj[:,4] = r * np.cos(theta) * omega
traj[:,5] =  0
traj[:,6] =  0
traj[:,7] =  0


# write to txt
np.savetxt('circle.txt',traj,fmt='%f')
