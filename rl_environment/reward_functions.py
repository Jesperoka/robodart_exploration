import numpy as np
from utils.dtypes import NP_DTYPE

# Euclidean Distance between dart and goal
def distance(d_pos, g_pos) -> NP_DTYPE:
    diff = d_pos - g_pos
    return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]).astype(NP_DTYPE)

# Only reward hitting the target wall
def sparse_reward_function(dart_position, goal_position) -> NP_DTYPE:
    if dart_position[1] <= goal_position[1]:
        return (np.sqrt(2.0**2 + 1.625**2) - np.sqrt((dart_position[0] - goal_position[0])**2 + (dart_position[2] - goal_position[2])**2)).astype(NP_DTYPE)
    return NP_DTYPE(0.0)

def capped_inverse_distance(d_pos, g_pos, minimum_distance=1e-3) -> NP_DTYPE:
    diff = d_pos - g_pos
    euclidean_distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    return np.min(1/euclidean_distance, initial=1.0/minimum_distance).astype(NP_DTYPE)

def scaled_capped_inverse_distance(d_pos, g_pos, minimum_distance=1e-3, scale=0.1) -> NP_DTYPE:
    diff = scale*(d_pos - g_pos)
    euclidean_distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    return np.min(1/euclidean_distance, initial=1.0/minimum_distance).astype(NP_DTYPE)

def modified_capped_inverse_distance(d_pos, g_pos, max_value=20000, scale=0.1, linear_factor=2.5, constant=10.0) -> NP_DTYPE:
    diff = d_pos - g_pos
    euclidean_distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    f_of_x = 1.0/(scale*euclidean_distance) - linear_factor*np.abs(diff[0] + diff[1] + diff[2]) + constant
    return np.min(f_of_x, initial=max_value).astype(NP_DTYPE)

def absolute_difference(input_1, input_2):
    return np.abs(input_1 - input_2).astype(NP_DTYPE)

def capped_inverse_absolute_difference(input_1, input_2, minimum_difference=1e-2):
    return np.min(1.0/np.abs(input_1 - input_2), initial=1.0/minimum_difference).astype(NP_DTYPE)
