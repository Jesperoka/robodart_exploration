import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box, MultiBinary, Tuple, flatten_space
from gymnasium.utils import EzPickle
from typing import Dict
from utils.dtypes import NP_DTYPE 

# WARNING: Different joints should have different limits
# WARNING: Action space limits need to not be able to break robot limits

RNG_SEED = 0

A_LOWER = -1.0
A_HIGHER = 1.0
O_LOWER = -NP_DTYPE("inf")
O_HIGHER = NP_DTYPE("inf")
NUM_JOINTS = 7
NUM_OBSERVABLE_STATES = 2*NUM_JOINTS # Dart Position
EPISODE_TIME_LIMIT = 5.0
GOAL = np.array([0.0, -2.46, 1.625], dtype=NP_DTYPE)

# TODO: use this
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def default_reward_function(dart_position, goal_position) -> NP_DTYPE:
    assert(dart_position.dtype == NP_DTYPE and goal_position.dtype == NP_DTYPE), str(dart_position.dtype) + " " + str(goal_position.dtype) 
    return -np.linalg.norm(dart_position - goal_position).astype(NP_DTYPE)


def default_baseline_controller(qpos, qvel) -> np.ndarray:
    Kp = 300.0
    Kd = 10.0

    REFERENCE_POS = np.array([
        1.585363, 0.972815, -0.074798, -1.660640, -1.538792,
        1.733046, 1.126374
    ], dtype=NP_DTYPE)

    e = REFERENCE_POS - qpos
    e_dot = np.zeros(7, dtype=NP_DTYPE) - qvel

    torques = (Kp*e + Kd*e_dot).squeeze()
    
    assert(torques.dtype == NP_DTYPE)
    return torques


class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):

    observation_space = Box(low=O_LOWER * np.ones(NUM_OBSERVABLE_STATES),
                            high=O_HIGHER * np.ones(NUM_OBSERVABLE_STATES), dtype=NP_DTYPE)
    time_limit = EPISODE_TIME_LIMIT 
    goal = GOAL

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        mujoco_model_path,
        frame_skip,
        reward_function=default_reward_function,
        baseline_controller=default_baseline_controller,
        camera_id=None,
        camera_name=None,
    ):

        self.reward_function = reward_function
        self.baseline_controller = baseline_controller

        EzPickle.__init__(self)
        MujocoEnv.__init__(self, mujoco_model_path, frame_skip, observation_space=self.observation_space, camera_name=camera_name, camera_id=camera_id)

        # Override action space
        # WARNING: If step() is only called by me, I don't need to do this
        self.action_space = Tuple([
            Box(low=A_LOWER * np.ones(NUM_JOINTS), high=A_HIGHER * np.ones(NUM_JOINTS), dtype=NP_DTYPE),
            MultiBinary(1) 
        ], seed=RNG_SEED)
        self.flat_action_space = flatten_space(self.action_space)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.render_mode = "human"

    # Sets control based on action simulates frame_skip number of frames and observes state 
    def step(self, action: np.ndarray) -> tuple[np.ndarray, NP_DTYPE, bool, Dict[str, float]]:
        done, reward = self.reward_or_terminate()
        baseline_action = self.baseline_controller(self.data.qpos[0:NUM_JOINTS], self.data.qvel[0:NUM_JOINTS])

        torques = action[0]
        release = action[1]

        torques += baseline_action
        self.model.eq("weld").active = release 

        self.do_simulation(torques, self.frame_skip)
        self.observation = self.noisy_observation() 

        info = {"action": action, "baseline_action": baseline_action} 

        assert(self.observation.dtype == NP_DTYPE and reward.dtype == NP_DTYPE), str(self.observation.dtype) + " " + str(reward.dtype) 
        return self.observation, reward, done, info

    # Checks termination criteria and returns reward
    def reward_or_terminate(self) -> tuple[bool, NP_DTYPE]:
        x_lim = 2.0
        y_lim = self.goal[1] 
        z_lim = 0.0
        dart_pos = self.data.qpos[-7:-4].astype(NP_DTYPE) # dart pos is 7-DOF xyz-quat

        if dart_pos[0] >= x_lim or dart_pos[1] >= -y_lim:
            return (True, -NP_DTYPE("inf"))
        elif dart_pos[1] <= y_lim or dart_pos[2] <= z_lim or self.data.time >= self.time_limit:
            return (True, self.reward_function(dart_pos, self.goal))
        else:
            return (False, self.reward_function(dart_pos, self.goal))

    # TODO: add noise
    def noisy_observation(self) -> np.ndarray:
        pos_noise = 0.0
        vel_noise = 0.0
        observation = np.concatenate([self.data.qpos[0:NUM_JOINTS] + pos_noise, self.data.qvel[0:NUM_JOINTS] + vel_noise], axis=0, dtype=NP_DTYPE)
        assert(observation.shape == (NUM_OBSERVABLE_STATES, ))
        return observation

    # Reset the model in the simulator
    def reset_model(self) -> np.ndarray:
        qpos = np.array([
            1.70923, 0.857012, -0.143804, -1.32138, -1.44791, 1.54729, -0.956189, 0.22234311, 1.62364254,
            0.41342544, 0.00098423, 0.024683026, 0.92662532, -0.37517367
        ], dtype=NP_DTYPE)
        qvel = np.zeros((self.model.nv, ), dtype=NP_DTYPE)

        self.set_state(qpos, qvel)
        self.model.eq("weld").active = 1 
        self.observation = np.concatenate([qpos[0:NUM_JOINTS], qvel[0:NUM_JOINTS]], axis=0) 

        assert(self.observation.shape == (NUM_OBSERVABLE_STATES, ))
        return self.observation

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    # def viewer_setup(self):
        # self.cam
