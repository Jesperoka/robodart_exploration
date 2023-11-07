from typing import Dict

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from mujoco import mju_mulQuat  # type: ignore

from rl_environment import reward_functions
from rl_environment.baseline_controllers import zero_controller
from rl_environment.constants import EnvConsts, Poses
from utils.dtypes import NP_DTYPE

# Environment Constants
_EC = EnvConsts
A_MIN = np.array([-20, -10, -20, -10, -4, -4, -4, -1])
A_MAX = np.array([20, 10, 20, 10, 4, 4, 4, 1])
INITIAL_RELEASE_POTENTIAL = -2.0


# Gymnasium Environment Definition
# ---------------------------------------------------------------------------- #
class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self,
                 mujoco_model_path,
                 frame_skip,
                 reward_function=reward_functions.capped_inverse_distance,
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
        self.observation_space = Box(low=np.array(_EC.O_MIN), high=np.array(_EC.O_MAX),
                                     dtype=NP_DTYPE)  # type: ignore
        self.action_space = Box(low=np.array(A_MIN), high=np.array(A_MAX), dtype=NP_DTYPE)  # type: ignore

        # Set final rendering options
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Define learning environment variables
        self.goal = np.array(_EC.GOAL, dtype=NP_DTYPE)  # TODO: add to state
        self.release_potential = np.array([INITIAL_RELEASE_POTENTIAL], NP_DTYPE)
        self.released = False
        self.time_limit = _EC.EPISODE_TIME_LIMIT
        self.reward_function = reward_function
        self.baseline_controller = baseline_controller

    # Sets control based on action simulates frame_skip number of frames and observes state
    def step(self, action: np.ndarray) -> tuple[np.ndarray, NP_DTYPE, bool, Dict[str, float]]:
        done, reward = self.reward_or_terminate()
        baseline_action = self.baseline_controller(self.data.qpos[0:_EC.NUM_JOINTS],
                                                   self.data.qvel[0:_EC.NUM_JOINTS])
        torques = action[0:_EC.NUM_JOINTS]
        self.release_potential[0] += 1.0065 + action[-1]

        if not self.released and self.release_potential >= 1.0:
            self.model.eq("weld").active = False
            self.released = True
            # reward += NP_DTYPE(10.0) 

        torques += baseline_action
        torques = np.clip(torques, a_min=_EC.A_MIN[0:_EC.NUM_JOINTS],
                          a_max=_EC.A_MAX[0:_EC.NUM_JOINTS])  # WARNING: clipping for now

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

        # Intermediary rewards
        # desired_joint_velocities = np.array([0.0, -0.35, 0.0, 0.0, 0.0, 0.0, 2.6100], dtype=NP_DTYPE)
        # reward = NP_DTYPE(0.05*reward_functions.capped_inverse_distance(self.observation[_EC.NUM_JOINTS:2*_EC.NUM_JOINTS], desired_joint_velocities))
        reward = NP_DTYPE(0.0)

        # Terminal rewards
        if self.terminal(dart_pos):
            if self.released: reward += reward_functions.capped_inverse_distance(dart_pos, self.goal)
            return (True, reward)
        else:
            return (False, reward)

    def terminal(self, dart_pos):
        x_lim, y_lim, z_lim = 2.0, -self.goal[1], 0.0
        return  not (-x_lim <= dart_pos[0] and dart_pos[0] <= x_lim)\
                or not (-y_lim <= dart_pos[1] and dart_pos[1] <= y_lim)\
                or not (z_lim <= dart_pos[2])\
                or not (self.data.time <= self.time_limit)

    # TODO: add noise
    def noisy_observation(self) -> np.ndarray:
        pos_noise = 0.0
        vel_noise = 0.0

        observation = np.concatenate([
            self.data.qpos[0:_EC.NUM_JOINTS] + pos_noise, self.data.qvel[0:_EC.NUM_JOINTS] + vel_noise,
            np.array([self.time_limit - self.data.time]), self.release_potential, self.goal 
        ],
                                     axis=0,
                                     dtype=NP_DTYPE)

        assert (observation.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return observation

    # Reset the model in the simulator
    def reset_model(self) -> np.ndarray:

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
        self.release_potential[0] = INITIAL_RELEASE_POTENTIAL
        self.observation = np.concatenate(
            [qpos[0:_EC.NUM_JOINTS], qvel[0:_EC.NUM_JOINTS],
             np.array([self.time_limit - self.data.time]), self.release_potential, self.goal],
            axis=0,
            dtype=NP_DTYPE)
        self.released = False
        self.baseline_controller.reset_controller()

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
