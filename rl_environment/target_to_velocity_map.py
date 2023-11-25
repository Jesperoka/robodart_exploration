from typing import Literal
import matplotlib.pyplot as plt
import numpy as np


# Returns a 4D array arr defining a 3D space where arr[i, j, k] is the 3D point (x,y,z) at index i,j,k along the x, y and z axes respectively.
def create_points(base_pt, len_x, len_y, len_z, volume_res, endpoint=False, idx: Literal['xy', 'ij'] ='ij'):
    return np.stack(np.meshgrid(
        np.linspace(base_pt[0], base_pt[0] + len_x, int(volume_res * len_x), endpoint=endpoint),
        np.linspace(base_pt[1], base_pt[1] + len_y, int(volume_res * len_y), endpoint=endpoint),
        np.linspace(base_pt[2], base_pt[2] + len_z, int(volume_res * len_z), endpoint=endpoint),
        indexing=idx),
        axis=-1)


# Compute the velocity vector that makes point mass projectile launched from start_pt hit end_pt.
def calculate_velocity_vector(start_pt: np.ndarray, end_pt: np.ndarray, vert_vel: float, g=9.81):
    delta = end_pt - start_pt

    time_arr = np.roots([0.5 * g, -vert_vel, delta[2]])
    time_arr = np.real(time_arr) # just converting straight up
    time_hit = max(time_arr)

    if np.allclose(time_hit, 0.0): return np.zeros(3)
    return np.array([delta[0] / time_hit, delta[1] / time_hit, vert_vel])


# Compute the velocity vectors needed to hit point target_pt from points in the 3D space defined by base_point, len_x, len_y, len_z.
def calculate_launch_point_and_velocity_vectors(base_pt, len_x, len_y, len_z, volume_res, target_pt, v_min, v_max, vel_res, endpoint=False, g=9.81):
    launch_point_and_velocities = []
    launch_pts = create_points(base_pt, len_x, len_y, len_z, volume_res, endpoint=endpoint)
    launch_pts = launch_pts.reshape(np.prod(launch_pts.shape[:-1]), launch_pts.shape[-1])

    for launch_pt in launch_pts:
        for vertical_vel in np.linspace(v_min, v_max, int((v_max - v_min) * vel_res), endpoint=endpoint):
            vel_vec = calculate_velocity_vector(launch_pt, target_pt, vertical_vel, g=g)
            launch_point_and_velocities.append((launch_pt, vel_vec))

    return np.array(launch_point_and_velocities)


# Plot trajectory from Newton's equations of motion
def plot_trajectory(target_pt, vel_start_combos, max_t, g=9.81):
    t_pts = np.linspace(0, max_t, 500)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for start_vel in vel_start_combos:
        start_x, start_y, start_z = start_vel[0, :]
        vel_x, vel_y, vel_z = start_vel[1, :]
        ax.scatter(start_x, start_y, start_z, c='g', marker='o')

        x = start_x + vel_x * t_pts
        y = start_y + vel_y * t_pts
        z = start_z + vel_z * t_pts - 0.5 * g * t_pts**2

        ax.plot(x, y, z)

    ax.scatter(*target_pt, c='r', marker='o', label='Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    g = 9.8219  # local gravitational acceleration in Trondheim at 45m according to WolframAlpha 
    base_pt = np.array([-0.5, -0.5, 1.64])
    dim_lengths = (1.0, 1.0, 3.0)
    volume_res = 10
    target_pt = np.array([0.0, -2.46, 1.64])
    vel_limits = (1.0, 2.0)
    vel_res = 10

    launch_point_and_velocities = calculate_launch_point_and_velocity_vectors(base_pt, *dim_lengths, volume_res, target_pt, *vel_limits, vel_res, g=g)

    num_samples = 15 
    max_traj_time = 0.5 
    sample_idxs = np.random.choice(len(launch_point_and_velocities) - 1, num_samples, replace=False)
    sample_launch_data = launch_point_and_velocities[sample_idxs, :, :]

    plot_trajectory(target_pt, sample_launch_data, max_traj_time, g=g)
