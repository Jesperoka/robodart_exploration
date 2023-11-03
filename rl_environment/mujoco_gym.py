from typing import Dict

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from mujoco import mju_mulQuat  # type: ignore

from rl_environment.baseline_controllers import zero_controller
from rl_environment.constants import EnvConsts, Poses
from rl_environment.reward_functions import capped_inverse_distance
from utils.dtypes import NP_DTYPE

# Environment Constants
_EC = EnvConsts


# Gymnasium Environment Definition
# ---------------------------------------------------------------------------- #
class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self,
                 mujoco_model_path,
                 frame_skip,
                 reward_function=capped_inverse_distance,
                 baseline_controller=zero_controller,
                 camera_id=None,
                 camera_name=None,
                 render_mode="human"):

        # Init base classes
        EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            mujoco_model_path,
            frame_skip,
            observation_space=Box(low=0, high=0, dtype=NP_DTYPE),  # placeholder
            camera_name=camera_name,
            camera_id=camera_id)

        # Override action and observation spaces
        self.observation_space = Box(low=np.array(_EC.O_MIN), high=np.array(_EC.O_MAX), dtype=NP_DTYPE)  # type: ignore
        self.action_space = Box(low=np.array(_EC.A_MIN), high=np.array(_EC.A_MAX), dtype=NP_DTYPE)  # type: ignore

        # Set final rendering options
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Define learning environment variables
        self.goal = np.array(_EC.GOAL, dtype=NP_DTYPE)  # TODO: add to state
        self.released = False
        self.time_limit = _EC.EPISODE_TIME_LIMIT
        self.reward_function = reward_function
        self.baseline_controller = baseline_controller
        self.failure_penalty = -1

    # Sets control based on action simulates frame_skip number of frames and observes state
    def step(self, action: np.ndarray) -> tuple[np.ndarray, NP_DTYPE, bool, Dict[str, float]]:
        done, reward = self.reward_or_terminate()
        baseline_action = self.baseline_controller(self.data.qpos[0:_EC.NUM_JOINTS],
                                                   self.data.qvel[0:_EC.NUM_JOINTS])
        print("\nba:  ", baseline_action)

        torques = action[0:_EC.NUM_JOINTS]
        release = action[-1]

        if not self.released and release >= 0.0:
            self.model.eq("weld").active = False
            self.released = True

        torques += baseline_action
        print("\ntau: ", torques)

        self.do_simulation(torques, self.frame_skip)
        self.observation = self.noisy_observation()

        info = {"action": action, "baseline_action": baseline_action}

        assert (self.observation.dtype == NP_DTYPE
                and reward.dtype == NP_DTYPE), str(self.observation.dtype) + " " + str(reward.dtype)
        return self.observation, reward, done, info

    # NOTE: IDEA: reward based on (normalized) dot product of dart vel vector and desired launch vector, and dart distance from launch position.
    # After release,

    # Checks termination criteria and returns reward
    def reward_or_terminate(self) -> tuple[bool, NP_DTYPE]:
        dart_pos = np.copy(self.data.qpos[-7:-4]).astype(NP_DTYPE)  # dart pos is 7-DOF xyz-quat
        # Environment limits
        x_lim = 2.0
        y_lim = self.goal[1]
        z_lim = 0.0

        # Main reward
        reward = self.reward_function(dart_pos, self.goal)

        # Intermediary rewards
        # reward += NP_DTYPE(-self.goal[1] -
        #                    dart_pos[1] if dart_pos[1] <= 0 else 0.0)  # can't really measure this

        # Terminal rewards
        if dart_pos[0] >= x_lim: return (True, reward + NP_DTYPE(self.failure_penalty))
        elif dart_pos[1] >= -y_lim: return (True, reward + NP_DTYPE(self.failure_penalty))
        elif dart_pos[2] <= z_lim: return (True, reward + NP_DTYPE(self.failure_penalty))
        elif self.data.time >= self.time_limit: return (True, reward + NP_DTYPE(self.failure_penalty))
        elif dart_pos[1] <= y_lim: return (True, reward)
        else: return (False, reward)

    # TODO: add noise
    def noisy_observation(self) -> np.ndarray:
        pos_noise = 0.0
        vel_noise = 0.0

        observation = np.concatenate([
            self.data.qpos[0:_EC.NUM_JOINTS] + pos_noise, self.data.qvel[0:_EC.NUM_JOINTS] + vel_noise,
            np.array([self.data.time])
        ],
                                     axis=0,
                                     dtype=NP_DTYPE)

        assert (observation.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return observation

    # Reset the model in the simulator
    def reset_model(self) -> np.ndarray:
        self.baseline_controller.reset_controller()

        # Set initial joint angles and compute forward kinematics
        qvel = np.zeros((self.model.nv, ), dtype=NP_DTYPE)
        initial_joint_angles = Poses.q0
        self.set_state(np.array([*initial_joint_angles, *([0.0] * 7)], dtype=NP_DTYPE), qvel)

        # Compute initial dart position
        release_point = self.data.body("temporary_release_mechanism")
        relative_offset = np.array([0.1, 0.0, 0.0], dtype=NP_DTYPE)
        initial_dart_position = (release_point.xpos + release_point.xmat.reshape(
            (3, 3)) @ relative_offset).tolist()

        # Compute initial dart orientation
        minus_90_deg_rotation_y = np.array([0.70710678118, 0, -0.70710678118, 0])
        initial_dart_orientation = self.data.body("temporary_release_mechanism").xquat
        mju_mulQuat(initial_dart_orientation, initial_dart_orientation, minus_90_deg_rotation_y)

        # Configure weld constraint
        self._set_weld_relpose(relative_offset, minus_90_deg_rotation_y, "weld")
        self.model.eq("weld").active = True

        # Set the full initial state and compute forward kinematics
        qpos = np.array([*initial_joint_angles, *initial_dart_position, *initial_dart_orientation.tolist()],
                        dtype=NP_DTYPE)
        self.set_state(qpos, qvel)

        # Reset members
        self.observation = np.concatenate([qpos[0:_EC.NUM_JOINTS], qvel[0:_EC.NUM_JOINTS],
                                           np.array([self.data.time])],
                                          axis=0,
                                          dtype=NP_DTYPE)
        self.released = False

        assert (self.observation.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return self.observation

    # Sets the relative pose used to compute weld constaints, this will probably not be necessary in the future
    def _set_weld_relpose(self, pos: np.ndarray, quat: np.ndarray, name: str):
        pos_idx, quat_idx = 3, 6
        data = self.model.eq(name).data
        data[pos_idx:pos_idx + 3] = np.copy(pos)
        data[quat_idx:quat_idx + 4] = np.copy(quat)

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}


# ---------------------------------------------------------------------------- #
