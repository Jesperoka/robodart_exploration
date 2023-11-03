import numpy as np
from utils.dtypes import NP_DTYPE


# Only reward hitting the target wall
def sparse_reward_function(dart_position, goal_position) -> NP_DTYPE:
    if dart_position[1] <= goal_position[1]:
        return (np.sqrt(2.0**2 + 1.625**2) - np.sqrt((dart_position[0] - goal_position[0])**2 + (dart_position[2] - goal_position[2])**2)).astype(NP_DTYPE)
    return NP_DTYPE(0.0)

def capped_inverse_distance(d_pos, g_pos, minimum_distance=1e-3) -> NP_DTYPE:
    diff = d_pos - g_pos
    euclidean_distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    return np.min(NP_DTYPE(euclidean_distance), initial=NP_DTYPE(1/minimum_distance))
