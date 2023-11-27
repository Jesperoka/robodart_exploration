import numpy as np
from scipy.spatial.transform import Rotation as R

# Converts a cartesian pose of the form (x, y, z, quaternion) into 4x4 transfomation matrix
def cartesian_pose_to_transformation_matrix(cartesian_pose=np.array(7)):
    
    orientation = R.from_quat(cartesian_pose[3:7]).as_matrix()
    pose_as_matrix = np.eye(4)
    pose_as_matrix[0:3, 0:3] = orientation
    pose_as_matrix[0:3, 3] = cartesian_pose[0:3].T

    return pose_as_matrix

# Converts a 4x4 transfomation matrix into a cartesian pose of the form (x, y, z, quaternion)
def transformation_matrix_to_cartesian_pose(transformation_matrix=np.array((4,4))):
    
    position = transformation_matrix[0:3, 3].T
    orientation = R.from_matrix(transformation_matrix[0:3, 0:3]).as_quat()

    # May have to rearrange quaternion to fit with panda_py quaternions
    # goal_quaternion = np.array([r[1], r[2], r[3], r[0]])

    cartesian_pose = np.append(position, orientation)

    return cartesian_pose

