import numpy as np
from utils.dtypes import NP_ARRTYPE, NP_DTYPE
from .constants import EnvConsts as _EC

# TODO: jit

def distance(d_pos, g_pos) -> NP_DTYPE:
    return np.linalg.norm(d_pos - g_pos, ord=2)

def close_enough(d_pos, g_pos) -> NP_DTYPE:
    if distance(d_pos, g_pos) <= 0.05: 
        return NP_DTYPE(1)
    return NP_DTYPE(0)

def abs_norm_diff(v1, v2, ord=2):
    return np.abs(np.linalg.norm(v1, ord=2) - np.linalg.norm(v2, ord=2))

def soboleva_tanh(x, a=1.0, b=1.0, c=1.0, d=1.0): 
    return (np.exp(a*x) - np.exp(-b*x)) / (np.exp(c*x) + np.exp(-d*x))

def on_dart_board(d_pos, board_center=_EC.BULLSEYE) -> NP_DTYPE:
    diff = np.array([d_pos[0] - board_center[0], d_pos[2] - board_center[2]], dtype=NP_DTYPE)
    if np.sqrt(diff[0]*diff[0] + diff[1]*diff[1]) <= _EC.BOARD_RADIUS: 
        return NP_DTYPE(1)
    return NP_DTYPE(0)
    
# NOTE: Contrary to what the authors use, I using sum instead of product 
# because cross product is zero when one of the vectors is zero, creating a bad local minima. 
def ts_ss_similarity(v1: NP_ARRTYPE, v2: NP_ARRTYPE) -> NP_DTYPE:
    ts = np.linalg.norm(np.cross(v1, v2), ord=2)
    ss = (np.linalg.norm(v1 - v2, ord=1) + np.linalg.norm(v1 - v2, ord=2))**2
    return ts + ss

def capped_inverse_distance(d_pos, g_pos, minimum_distance=1e-3) -> NP_DTYPE:
    euclidean_distance = distance(d_pos, g_pos)
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
