import os

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box, Discrete, Tuple
from gymnasium.utils import EzPickle
from quaternion import from_rotation_matrix

# WARNING: Different joints should have different limits
# WARNING: Action space limits need to not be able to break robot limits

DTYPE = np.float32
RNG_SEED = 0

A_LOWER = -1.0
A_HIGHER = 1.0
O_LOWER = -DTYPE("inf")
O_HIGHER = DTYPE("inf")
NUM_JOINTS = 7
NUM_OBSERVABLE_STATES = 3  # Dart Position
INITIAL_OBSERVATION = np.array([0.0, 0.0, 0.0], dtype=DTYPE)  # TODO: get proper inital state
GOAL = np.array([0.0, -2.46, 1.625], dtype=DTYPE)
EPISODE_TIME_LIMIT = 69.0


def default_reward_function(dart_position, goal_position) -> DTYPE:
    return -np.linalg.norm(dart_position - goal_position)


def default_baseline_controller(qpos, qvel) -> np.ndarray:
    Kp = 300.0
    Kd = 10.0

    REFERENCE_POS = np.array([
        1.2535917688652172, 0.6972054325100352, 0.38962657131646805, -1.228720040789675, -0.24577336054378085,
        3.436912040630976, 2.358766003196306
    ])

    e = REFERENCE_POS - qpos
    e_dot = np.zeros(7) - qvel

    torques = Kp*e + Kd*e_dot

    return torques.squeeze()


class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):

    observation_space = Box(low=O_LOWER * np.ones(NUM_OBSERVABLE_STATES),
                            high=O_HIGHER * np.ones(NUM_OBSERVABLE_STATES), dtype=DTYPE)
    time_limit = EPISODE_TIME_LIMIT 
    observation = INITIAL_OBSERVATION
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
    ):

        self.reward_function = reward_function
        self.baseline_controller = baseline_controller

        EzPickle.__init__(self)
        MujocoEnv.__init__(self, mujoco_model_path, frame_skip, observation_space=self.observation_space)

        print(self.data.site("release_point").xpos)
        print(self.init_qpos)
        print(self.init_qvel)

        # Override action space
        # WARNING: If step() is only called by me, I don't need to do this
        self.action_space = Tuple([
            Box(low=A_LOWER * np.ones(NUM_JOINTS), high=A_HIGHER * np.ones(NUM_JOINTS), dtype=DTYPE),
            Discrete(n=2, seed=RNG_SEED, start=0)
        ])

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
    def step(self, action: np.ndarray):
        done, reward = self.reward_or_terminate()
        baseline_action = self.baseline_controller(self.data.qpos[0:NUM_JOINTS], self.data.qvel[0:NUM_JOINTS])

        torques = action[0]
        release = action[1]

        print(torques.shape)
        torques += baseline_action
        self.model.eq("weld").active = release 

        self.do_simulation(torques, self.frame_skip)
        self.observation = self.noisy_observation() 

        info = {"action": action, "baseline_action": baseline_action} 

        return self.observation, reward, done, info

    # Checks termination criteria and returns reward
    def reward_or_terminate(self):
        x_lim = 2.0
        y_lim = self.goal[1] 
        z_lim = 0.0
        dart_pos = self.observation[0][-3:]

        if dart_pos[0] >= x_lim or dart_pos[1] >= -y_lim:
            return (True, -DTYPE("inf"))
        elif dart_pos[1] <= y_lim or dart_pos[2] <= z_lim or self.data.time >= self.time_limit:
            return (True, self.reward_function(dart_pos, self.goal))
        else:
            return (False, self.reward_function(dart_pos, self.goal))

    # TODO: add noise
    def noisy_observation(self):
        noise = 0
        return (self.data.qpos[-7:-4].copy() + noise, self.data.qvel[-7:-4].copy() + noise)

    # Reset the model in the simulator
    def reset_model(self):
        qpos = np.array([
            1.70923, 0.857012, -0.143804, -1.32138, -1.44791, 1.54729, -0.956189, 0.22234311, 1.52364254,
            0.41342544, 0.00098423, 0.024683026, 0.92662532, -0.37517367
        ])
        qvel = np.zeros((self.model.nv, ))
        self.set_state(qpos, qvel)
        self.observation = (qpos, qvel) 

        return self.observation

    # def viewer_setup(self):
    #     self.render_mode = "human"
    #     self.camera_id = 0
    #     self.camera_name = "my_camera"


# test
