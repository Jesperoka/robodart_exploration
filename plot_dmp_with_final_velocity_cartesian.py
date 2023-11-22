"""
=======================
DMP with Final Velocity
=======================

Not all DMPs allow a final velocity > 0. In this case we analyze the effect
of changing final velocities in an appropriate variation of the DMP
formulation that allows to set the final velocity.
"""
print(__doc__)


# import sys
# print(sys.path)
# sys.path.append("/home/chrisova/Desktop/movement_primitives-main")

import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from json import load
import panda_py
from scipy.spatial.transform import Rotation as R


SHOP_FLOOR_IP = "10.0.0.2"


#Y = np.column_stack((np.cos(np.pi * T), -np.cos(np.pi * T), 0.5*T, 0.1*T, 0*T, 0*T, 0.5*T))

with open("trajectory_pos.txt") as f:
    traj = load(f)




####################################################

# Interpolating trajectory from set of poses
# Poses provided
poses = [
    [-0.26684645, 0.72326601, 0.29045006, -0.33319444, -0.63154138, 0.2573719, 0.65107346],
    [-0.27151081, 0.59114054, 0.43162935, -0.2741086, -0.65613221, 0.21893884, 0.66814728],
    [-0.26576335, 0.39897345, 0.58287283, -0.1481909, -0.67848496, 0.13447379, 0.70683408],
    [-0.25124114, 0.21696712, 0.7275276, 0.05008003, -0.69233572, -0.0412439, 0.71865304],
    [-0.24030373, 0.0226624, 0.8485015, 0.36221429, -0.58070067, -0.34858058, 0.6403742 ],
    [-0.26317091, -0.19931589, 0.91034192, 0.61383395, -0.34359184, -0.57310566, 0.42035989]
]

# Variables for np.linspace arguments
num_samples = 1000  # Number of samples for interpolation
axis = 0  # Axis along which to interpolate

# Initialize an empty list to store interpolated arrays
interpolated_poses = []

# Loop through each pair of consecutive poses and interpolate
for i in range(len(poses) - 1):
    start_pose = np.array(poses[i])
    end_pose = np.array(poses[i + 1])
    interpolated = np.linspace(start_pose, end_pose, num_samples, axis=axis)
    interpolated_poses.append(interpolated)

# Concatenate all interpolated arrays
concatenated_interpolated_poses = np.concatenate(interpolated_poses, axis=axis)

###################################################

# Y = traj
Y = concatenated_interpolated_poses
N = max(Y.shape)
execution_time = 5
dt = execution_time / N
T = np.linspace(0, execution_time, N)

dmp = DMPWithFinalVelocity(
        n_dims=7, 
        execution_time=execution_time, dt=dt, 
        n_weights_per_dim=10
        )

dmp.imitate(T, Y)

plt.figure(figsize=(20, 15))
ax1 = plt.subplot(2,7,1)
ax1.set_title("x")
ax1.set_xlabel("Time")
ax1.set_ylabel("Position")

ax2 = plt.subplot(2,7,2)
ax2.set_title("y")
ax2.set_xlabel("Time")
ax2.set_ylabel("Position")

ax3 = plt.subplot(2,7,3)
ax3.set_title("z")
ax3.set_xlabel("Time")
ax3.set_ylabel("Position")

ax4 = plt.subplot(2,7,4)
ax4.set_title("q_x")
ax4.set_xlabel("Time")
ax4.set_ylabel("Position")

ax5 = plt.subplot(2,7,5)
ax5.set_title("q_y")
ax5.set_xlabel("Time")
ax5.set_ylabel("Position")

ax6 = plt.subplot(2,7,6)
ax6.set_title("q_z")
ax6.set_xlabel("Time")
ax6.set_ylabel("Position")

ax7 = plt.subplot(2,7,7)
ax7.set_title("q_w")
ax7.set_xlabel("Time")
ax7.set_ylabel("Position")

ax8 = plt.subplot(2,7,8)
ax8.set_xlabel("Time")
ax8.set_ylabel("Velocity")

ax9 = plt.subplot(2,7,9)
ax9.set_xlabel("Time")
ax9.set_ylabel("Velocity")

ax10 = plt.subplot(2,7,10)
ax10.set_xlabel("Time")
ax10.set_ylabel("Velocity")

ax11 = plt.subplot(2,7,11)
ax11.set_xlabel("Time")
ax11.set_ylabel("Velocity")

ax12 = plt.subplot(2,7,12)
ax12.set_xlabel("Time")
ax12.set_ylabel("Velocity")

ax13 = plt.subplot(2,7,13)
ax13.set_xlabel("Time")
ax13.set_ylabel("Velocity")

ax14 = plt.subplot(2,7,14)
ax14.set_xlabel("Time")
ax14.set_ylabel("Velocity")


ax1.plot(T, Y[:, 0], label="Demo")
ax2.plot(T, Y[:, 1], label="Demo")
ax3.plot(T, Y[:, 2], label="Demo")
ax4.plot(T, Y[:, 3], label="Demo")
ax5.plot(T, Y[:, 4], label="Demo")
ax6.plot(T, Y[:, 5], label="Demo")
ax7.plot(T, Y[:, 6], label="Demo")

ax8.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
ax9.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
ax10.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)
ax11.plot(T, np.gradient(Y[:, 3]) / dmp.dt_)
ax12.plot(T, np.gradient(Y[:, 4]) / dmp.dt_)
ax13.plot(T, np.gradient(Y[:, 5]) / dmp.dt_)
ax14.plot(T, np.gradient(Y[:, 6]) / dmp.dt_)

ax8.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
ax9.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
ax10.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
ax11.scatter([T[-1]], (Y[-1, 3] - Y[-2, 3]) / dmp.dt_)
ax12.scatter([T[-1]], (Y[-1, 4] - Y[-2, 4]) / dmp.dt_)
ax13.scatter([T[-1]], (Y[-1, 5] - Y[-2, 5]) / dmp.dt_)
ax14.scatter([T[-1]], (Y[-1, 6] - Y[-2, 6]) / dmp.dt_)


# cartesian_goal = np.array([[-0.2, 0.1, 1.2],[0.99110923, -0.04505042, 1]])
cartesian_goal = np.array([[-0.26, -0.2, 1.2],[0, -0.14505042, 1]])

for goal_yd in [0.0, 0.1, 0.2]:
    dmp.configure(goal_y=np.array(np.append(cartesian_goal[0],Y[-1,3:7])), goal_yd=np.array(np.append(cartesian_goal[1],[0,0,0,0])))
    T, Y = dmp.open_loop(run_t=execution_time)
    ax1.plot(T, Y[:, 0], label="goal_yd = %g" % goal_yd)
    ax2.plot(T, Y[:, 1], label="goal_yd = %g" % goal_yd)
    ax3.plot(T, Y[:, 2], label="goal_yd = %g" % goal_yd)
    ax4.plot(T, Y[:, 3], label="goal_yd = %g" % goal_yd)
    ax5.plot(T, Y[:, 4], label="goal_yd = %g" % goal_yd)
    ax6.plot(T, Y[:, 5], label="goal_yd = %g" % goal_yd)
    ax7.plot(T, Y[:, 6], label="goal_yd = %g" % goal_yd)

    ax8.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
    ax9.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
    ax10.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)
    ax11.plot(T, np.gradient(Y[:, 3]) / dmp.dt_)
    ax12.plot(T, np.gradient(Y[:, 4]) / dmp.dt_)
    ax13.plot(T, np.gradient(Y[:, 5]) / dmp.dt_)
    ax14.plot(T, np.gradient(Y[:, 6]) / dmp.dt_)

    ax8.scatter([T[-1]], [goal_yd])
    ax9.scatter([T[-1]], [goal_yd])
    ax10.scatter([T[-1]], [goal_yd])
    ax11.scatter([T[-1]], [goal_yd])
    ax12.scatter([T[-1]], [goal_yd])
    ax13.scatter([T[-1]], [goal_yd])
    ax14.scatter([T[-1]], [goal_yd])

ax1.legend()
plt.tight_layout()
plt.savefig("cartesian_test.pdf")
plt.show()

