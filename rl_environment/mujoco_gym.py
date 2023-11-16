from typing import Dict

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from mujoco import mju_mulQuat, mjv_initGeom, mjv_addGeoms, mjtGeom, mjtObj, mjtCatBit
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

        dart_pos = self.data.qpos[-7:-4]
        info = {"distance": reward_functions.distance(dart_pos, self.goal)}

        assert (self.observation.dtype == NP_DTYPE
                and reward.dtype == NP_DTYPE), str(self.observation.dtype) + " " + str(reward.dtype)
        return self.observation, reward, done, info


    # Checks termination criteria and returns reward
    def reward_or_terminate(self) -> tuple[bool, NP_DTYPE]:
        dart_pos = np.copy(self.data.qpos[-7:-4]).astype(NP_DTYPE)  # dart pos is 7-DOF xyz-quat
        # joint_angles = self.data.qpos[0:_EC.NUM_JOINTS]
        # joint_angular_velocities = self.data.qvel[0:_EC.NUM_JOINTS]

        # reward = reward_functions.close_enough(dart_pos, self.goal)
        reward = -reward_functions.distance(dart_pos, self.goal)

        # Terminal rewards
        if self.terminal2(dart_pos):
            return (True, reward)
        else:
            return (False, reward)

    def terminal(self, dart_pos):
        x_lim, y_lim, z_lim = 2.0, -self.goal[1], 0.0
        return  not (-x_lim <= dart_pos[0] and dart_pos[0] <= x_lim)\
                or not (-y_lim <= dart_pos[1] and dart_pos[1] <= y_lim)\
                or not (z_lim <= dart_pos[2])\
                or not (self.data.time <= self.time_limit)

    def terminal2(self, *args):
        return self.data.time >= self.time_limit

    # TODO: add noise
    # TODO: normalize observations (not self.released)
    def noisy_observation(self) -> np.ndarray:
        pos_noise = 0.0
        vel_noise = 0.0
        dart_pos = self.data.qpos[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3]
        dart_vel = self.data.qvel[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3]
        joint_angles = self.data.qpos[0:_EC.NUM_JOINTS]
        joint_angular_velocities = self.data.qvel[0:_EC.NUM_JOINTS]
        remaining_time = np.array([self.time_limit - self.data.time])
        released = np.array([self.released])

        observation = np.concatenate([
            joint_angles + pos_noise, joint_angular_velocities + vel_noise, remaining_time, released,
            self.goal, dart_pos, dart_vel
        ],
                                     axis=0,
                                     dtype=NP_DTYPE)

        assert (observation.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return observation

    # Reset the model in the simulator
    def reset_model(self) -> np.ndarray:

        # Set initial joint angles and compute forward kinematics
        noise_shape = (len(Poses.q0), )
        qvel = np.zeros((self.model.nv, ), dtype=NP_DTYPE) #+ np.random.uniform(-0.05, 0.05, size=(self.model.nv, ))
        initial_joint_angles = Poses.q0 + np.random.uniform(-0.4, 0.4, size=noise_shape)
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
        self.goal = _EC.GOAL + np.random.uniform(-0.2255, 0.2255, self.goal.shape)
        # self._add_marker(self.goal)
        self.reward_shrinkage = NP_DTYPE(1.0)
        self.released = False
        self.observation = np.concatenate([
            qpos[0:_EC.NUM_JOINTS], qvel[0:_EC.NUM_JOINTS],
            np.array([self.time_limit - self.data.time]), np.array([self.released]), self.goal,
            qpos[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3], qvel[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3]
        ],
                                          axis=0,
                                          dtype=NP_DTYPE)
        self.baseline_controller.reset_controller()

        assert (self.observation.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return self.observation

    def _add_marker(self, pos, color=[1, 0, 0, 1]):
        viewer = self.mujoco_renderer._get_viewer(self.render_mode)
        viewer.add_marker(pos=pos, label="Goal", rgba=color, size=0.05*np.ones(3))
        # print(dir(viewer))
        # input()
        # mjv_initGeom(self.mujoco_renderer._get_viewer(self.render_mode), mjtGeom.mjGEOM_SPHERE, 0.1*np.ones(3), pos, np.identity(3), np.array(color))
        # mjv_addGeoms(self.model, self.data)

    def _render(self):
        self._add_marker(self.goal)
        self.render()
        self.mujoco_renderer._get_viewer(self.render_mode).scn.ngeom -= 1

    # def _add_marker_to_scene(self, marker: dict):
    #     if self.scn.ngeom >= self.scn.maxgeom:
    #         raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

    #     viewer = self.mujoco_renderer._get_viewer(self.render_mode)
    #     g = viewer.scn.geoms[self.scn.ngeom]
    #     # default values.
    #     g.dataid = -1
    #     g.objtype = mjtObj.mjOBJ_UNKNOWN
    #     g.objid = -1
    #     g.category = mjtCatBit.mjCAT_DECOR
    #     g.texid = -1
    #     g.texuniform = 0
    #     g.texrepeat[0] = 1
    #     g.texrepeat[1] = 1
    #     g.emission = 0
    #     g.specular = 0.5
    #     g.shininess = 0.5
    #     g.reflectance = 0
    #     g.type = mjtGeom.mjGEOM_SPHERE 
    #     g.size[:] = np.ones(3) * 0.1
    #     g.mat[:] = np.eye(3)
    #     g.rgba[:] = np.ones(4)

    #     viewer.scn.ngeom += 1

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
