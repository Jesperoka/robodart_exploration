from typing import Dict

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from mujoco import mju_mulQuat
from numpy.core.multiarray import ndarray  # type: ignore

from rl_environment import reward_functions
from rl_environment.constants import EnvConsts, Poses
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
                 baseline_controller,
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
        self.action_space = Box(low=np.array(_EC.A_MIN), high=np.array(_EC.A_MAX), dtype=NP_DTYPE)  # type: ignore

        # Set final rendering options
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Define learning environment variables
        self.goal = np.array(_EC.GOAL, dtype=NP_DTYPE)
        self.released = False
        self.time_limit = _EC.EPISODE_TIME_LIMIT
        self.baseline_controller = baseline_controller
        self.reward_shrinkage = NP_DTYPE(1.0)


    # Sets control based on action simulates frame_skip number of frames and observes state
    def step(self, action: np.ndarray) -> tuple[np.ndarray, NP_DTYPE, bool, Dict[str, ndarray]]:
        qpos_ref = action[0:-1]
        release = action[-1]

        # Compute Torques 
        torques = self.baseline_controller(self.data.time, self.data.qpos[0:_EC.NUM_JOINTS],
                                                   self.data.qvel[0:_EC.NUM_JOINTS], qpos_ref)
        # Check if dart release
        if not self.released and release >= 0.0:
            self.model.eq("weld").active = False
            self.released = True

        # Simulate and observe environment
        self.do_simulation(torques, self.frame_skip)
        done, reward = self.reward_or_terminate()
        self.observation = self.noisy_observation()

        info = {"action": action}

        assert (self.observation.dtype == NP_DTYPE
                and reward.dtype == NP_DTYPE), str(self.observation.dtype) + " " + str(reward.dtype)
        return self.observation, reward, done, info


    # Checks termination criteria and returns reward
    def reward_or_terminate(self) -> tuple[bool, NP_DTYPE]:
        dart_pos = np.copy(self.data.qpos[-7:-4]).astype(NP_DTYPE)  # dart pos is 7-DOF xyz-quat
        joint_angles = self.data.qpos[0:_EC.NUM_JOINTS]
        joint_angular_velocities = self.data.qvel[0:_EC.NUM_JOINTS]

        # Intermediary rewards
        if (_EC.Q_DOT_MAX < joint_angular_velocities).any() or (joint_angular_velocities < _EC.Q_DOT_MIN).any():
            self.reward_shrinkage = self.reward_shrinkage * NP_DTYPE(0.9)
        if (_EC.Q_MAX <= joint_angles).any() or (joint_angles <= _EC.Q_MIN).any():
            self.reward_shrinkage = self.reward_shrinkage * NP_DTYPE(0.9)
        reward = self.reward_shrinkage*reward_functions.capped_inverse_distance(dart_pos, self.goal) 

        # Terminal rewards
        if self.terminal(dart_pos):
            if self.released: reward += reward_functions.capped_inverse_distance(dart_pos, self.goal)
            return (True, self.reward_shrinkage*reward)
        else:
            return (False, self.reward_shrinkage*reward)

    def terminal(self, dart_pos):
        x_lim, y_lim, z_lim = 2.0, -self.goal[1], 0.0
        return  not (-x_lim <= dart_pos[0] and dart_pos[0] <= x_lim)\
                or not (-y_lim <= dart_pos[1] and dart_pos[1] <= y_lim)\
                or not (z_lim <= dart_pos[2])\
                or not (self.data.time <= self.time_limit)

    # TODO: add noise
    # TODO: normalize observations (not self.released)
    def noisy_observation(self) -> np.ndarray:
        pos_noise = 0.0
        vel_noise = 0.0
        joint_angles = self.data.qpos[0:_EC.NUM_JOINTS]
        joint_angular_velocities = self.data.qvel[0:_EC.NUM_JOINTS]
        remaining_time = np.array([self.time_limit - self.data.time])
        released = np.array([self.released])

        observation = np.concatenate([
            joint_angles + pos_noise, joint_angular_velocities + vel_noise, remaining_time, released,
            self.goal
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
        self.reward_shrinkage = NP_DTYPE(1.0)
        self.released = False
        self.observation = np.concatenate([
            qpos[0:_EC.NUM_JOINTS], qvel[0:_EC.NUM_JOINTS],
            np.array([self.time_limit - self.data.time]), np.array([self.released]), self.goal
        ],
                                          axis=0,
                                          dtype=NP_DTYPE)
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
