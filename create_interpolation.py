import numpy as np

def create_interpolation(poses, num_samples=1000, axis=0):
    # Initialize an empty list to store interpolated arrays
    interpolated_poses = []

    # Loop through each pair of consecutive poses and interpolate
    for i in range(len(poses) - 1):
        start_pose = np.array(poses[i])
        end_pose = np.array(poses[i + 1])
        interpolated = np.linspace(start_pose, end_pose, num_samples, axis=axis)
        interpolated_poses.append(interpolated)

    # Concatenate all interpolated arrays
    return np.concatenate(interpolated_poses, axis=axis)

