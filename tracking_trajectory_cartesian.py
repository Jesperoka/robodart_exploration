import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from json import load
from plot_dmp import plot_dmp
from convert_pose_expression import cartesian_pose_to_transformation_matrix, transformation_matrix_to_cartesian_pose
from scipy.spatial.transform import Rotation as R

import panda_py
from panda_py import controllers

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

if __name__ == '__main__':

    panda = panda_py.Panda(SHOP_FLOOR_IP)

    ####################################################

    # Interpolating trajectory from set of poses
    # Cartesian poses provided

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


    demo_y = concatenated_interpolated_poses
    N = max(demo_y.shape)
    execution_time = 15
    dt = execution_time / N
    n_weights_per_dim = 15
    T = np.linspace(0, execution_time, N)   

    dmp = DMPWithFinalVelocity(n_dims=7, execution_time=execution_time, dt=dt, n_weights_per_dim=n_weights_per_dim)
    dmp.imitate(T, demo_y)

    # Example goal position and velocity from target_to_velocity_map.py
    # cartesian_goal = np.array([[-0.26, -0.2, 1.2],[0, -0.14505042, 1]])

    transformation_matrix = np.array([[0.03815925, -0.03711592, -0.99858211, -0.26],
                                    [-0.69274338, 0.71920556, -0.053205, -0.2],
                                    [0.72016055, 0.69379141, 0.00173257, 1.2],
                                    [0, 0, 0, 1]])

    goal_y = transformation_matrix_to_cartesian_pose(transformation_matrix)

    dmp.configure(goal_y=goal_y, goal_yd=np.array(np.append([0, -0.14505042, 1],[0,0,0,0])))

    T, dmp_y = dmp.open_loop(run_t=execution_time)
    dmp_yd = (1.0/dmp.dt_)*np.gradient(dmp_y, axis=0)

    plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd)

    # Generate trajectory in joint space for control purpose
    joint_trajectory = np.zeros_like(dmp_y)
    joint_trajectory[0] = poses[0]

    for index, pose in enumerate(dmp_y):
        if index > 0:
            pose_as_matrix = cartesian_pose_to_transformation_matrix(pose)
            joint_trajectory[index] = panda_py.ik(pose_as_matrix, joint_trajectory[index-1])

    joint_trajectory_yd = (1.0/dmp.dt_)*np.gradient(joint_trajectory, axis=0)

    panda.move_to_joint_position(joint_trajectory[0])
    input("Press any key to start")

    i = 0
    ctrl = controllers.JointPosition()
    panda.start_controller(ctrl)
    with panda.create_context(frequency=500, max_runtime=execution_time) as ctx:
        while ctx.ok():
            ctrl.set_control(joint_trajectory[i], joint_trajectory_yd[i])
            i += 1
