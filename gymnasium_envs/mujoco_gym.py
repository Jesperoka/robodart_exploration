import mujoco
import os
import numpy as np

from gymnasium.utils import EzPickle
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

# WARNING: Different joints should have different limits
# WARNING: Action space limits need to not be able to break robot limits

A_LOWER = -1.0 
A_HIGHER = 1.0
O_LOWER = -np.inf
O_HIGHER = np.inf 
NUM_JOINTS = 7 
NUM_OBSERVABLE_STATES = 3 # Dart Position
MAX_EPISODE_LENGTH = 69.0
INITIAL_STATE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
GOAL_STATE = np.array([0.0, 0.0, 1.73], dtype=np.float64)


def default_reward_function(dart_position, goal_position) -> float:
    return -np.linalg.norm(dart_position, goal_position)  


class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):

    action_space = Box(low=A_LOWER*np.ones(NUM_JOINTS), high=A_HIGHER*np.ones(NUM_JOINTS))
    observation_space = Box(low=O_LOWER*np.ones(NUM_OBSERVABLE_STATES), high=O_HIGHER*np.ones(NUM_OBSERVABLE_STATES))
    max_episode_length = MAX_EPISODE_LENGTH
    state = INITIAL_STATE
    goal_state = GOAL_STATE

    metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                ],
            "render_fps": 60,
            }

    def __init__(self, 
                 mujoco_model_path, 
                 frame_skip,
                 reward_function=default_reward_function,
                 ):

        self.reward_function = reward_function

        EzPickle.__init__(self)
        MujocoEnv.__init__(self, mujoco_model_path, frame_skip)


    def step(self):

        r = self.compute_reward(self.state, self.goal_state)
        info = {} # placeholder

        return self.state, r, done, info 

    def reset_model(self):
        qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04] 
        qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.set_state(qpos, qvel)
        return self.state 
    
    def compute_reward(self, *args, **kwargs):
        return reward_function(args, kwargs)

    def viewer_setup(self):
        pass # optional





# test



