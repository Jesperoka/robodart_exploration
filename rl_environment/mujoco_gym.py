from typing import Dict

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from mujoco import (mjtCatBit, mjtGeom, mjtObj, mju_mulQuat, mjv_addGeoms,
                    mjv_initGeom)
from typeguard import typechecked

from rl_environment import reward_functions
from rl_environment.constants import EnvConsts as _EC
from rl_environment.constants import Poses
from rl_environment.target_to_velocity_map import launch_pairs, trajectory
from utils.dtypes import NP_ARRTYPE, NP_DTYPE


# Gymnasium Environment Definition
# ---------------------------------------------------------------------------- #
class FrankaEmikaDartThrowEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        hybrid: bool,
        reward_function: int,
        mujoco_model_path,
        frame_skip,
        baseline_controller,
        camera_id=None,
        camera_name=None,
        render_mode="human",
        display_traj=False,
        display_corners=False,
    ):

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
        self.observation_space = Box(low=-np.inf * np.ones(_EC.NUM_OBSERVABLE_STATES),
                                     high=np.inf * np.ones(_EC.NUM_OBSERVABLE_STATES),
                                     dtype=NP_DTYPE)  # type: ignore
        
        # Allow for hybrid or continuous action spaces
        self.hybrid = hybrid
        if hybrid:
            self.action_space = Box(low=np.array([*_EC.A_MIN, 0]), high=np.array([*_EC.A_MAX, 1]), dtype=NP_DTYPE)  # type: ignore
        else:
            self.action_space = Box(low=np.array([*_EC.A_MIN, -1]), high=np.array([*_EC.A_MAX, 1]), dtype=NP_DTYPE)  # type: ignore

        # Allow for different reward functions
        self.reward_function = [self.reward, self.sparse_reward][reward_function]

        # Set final rendering options
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        self.display_traj = display_traj
        self.display_corners = display_corners

        # Define learning environment variables
        self.state = np.zeros(_EC.NUM_OBSERVABLE_STATES, dtype=NP_DTYPE)
        self.goal = np.array(_EC.BULLSEYE, dtype=NP_DTYPE)
        self.released = False
        self.releasing = False
        self.time_limit = _EC.EPISODE_TIME_LIMIT
        self.baseline_controller = baseline_controller
        self.launch_pt = np.zeros(3)
        self.launch_vel = np.zeros(3)
        self.launch_traj_res = 20
        self.launch_traj = np.zeros((self.launch_traj_res, 3))
        self.state_indices = np.array([0, 7, 14, 15, 16, 17, 20, 23, 26, 29, 32], dtype=int)
        # q, q_dot, t_r, r1, r2, g, lp, lv, dp, dv

    def step(self, action: np.ndarray,
             disc_action: int) -> tuple[np.ndarray, NP_DTYPE, np.uint8, Dict[str, NP_DTYPE]]:

        qpos_ref = action
        release = bool(disc_action) if self.hybrid else bool(disc_action >= 0)

        self.dart_release(release)
        (joint_angs, joint_ang_vels, *_) = self.decompose_state(self.state)
        torques = self.baseline_controller(self.data.time, joint_angs, joint_ang_vels, qpos_ref)
        self.do_simulation(torques, self.frame_skip)

        self.state = self.observe()
        done, reward = self.reward_function(self.state)

        (*_, dart_pos, _) = self.decompose_state(self.state)
        info = {"distance": reward_functions.distance(dart_pos, self.goal), "reward": reward}

        return self.state, reward, np.uint8(done), info

    # TODO: add noise
    @typechecked
    def observe(self) -> NP_ARRTYPE:
        idxs = self.state_indices
        observation = np.zeros(_EC.NUM_OBSERVABLE_STATES, dtype=NP_DTYPE)

        observation[idxs[0]:idxs[1]] = self.data.qpos[0:_EC.NUM_JOINTS]
        observation[idxs[1]:idxs[2]] = self.data.qvel[0:_EC.NUM_JOINTS]
        observation[idxs[2]:idxs[3]] = self.time_limit - self.data.time
        observation[idxs[3]:idxs[4]] = self.released
        observation[idxs[4]:idxs[5]] = self.releasing
        observation[idxs[5]:idxs[6]] = self.goal
        observation[idxs[6]:idxs[7]] = self.launch_pt
        observation[idxs[7]:idxs[8]] = self.launch_vel
        observation[idxs[8]:idxs[9]] = self.data.qpos[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3]
        observation[idxs[9]:None] = self.data.qvel[_EC.NUM_JOINTS:_EC.NUM_JOINTS + 3]

        return observation

    @typechecked
    def decompose_state(
        self, state: NP_ARRTYPE
    ) -> tuple[NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE,
               NP_ARRTYPE, NP_ARRTYPE]:
        idxs = self.state_indices
        joint_angs = state[idxs[0]:idxs[1]]
        joint_ang_vels = state[idxs[1]:idxs[2]]
        remaining_time = state[idxs[2]:idxs[3]]
        released = state[idxs[3]:idxs[4]]
        releasing = state[idxs[4]:idxs[5]]
        goal = state[idxs[5]:idxs[6]]
        launch_pt = state[idxs[6]:idxs[7]]
        launch_vel = state[idxs[7]:idxs[8]]
        dart_pos = state[idxs[8]:idxs[9]]
        dart_vel = state[idxs[9]:None]
        return (joint_angs, joint_ang_vels, remaining_time, released, releasing, goal, launch_pt, launch_vel,
                dart_pos, dart_vel)

    @typechecked
    def reward(self, state: NP_ARRTYPE) -> tuple[bool, NP_DTYPE]:
        (_, joint_ang_vels, _, released, releasing, goal, launch_pt, launch_vel, dart_pos,
         dart_vel) = self.decompose_state(state)

        reward = 0.0
        bonus = 0.0
        penalty = 0.0

        # TODO: idea, first X seconds, align arm with launch_pt, next, perform throw

        if (np.abs(joint_ang_vels) >= _EC.Q_DOT_MAX).any():
                penalty -= 0.01

        if not released:
            reward += 1.0 / (1.0 + reward_functions.ts_ss_similarity(dart_pos, launch_pt))
            reward += 0.6 / (1.0 + reward_functions.abs_norm_diff(dart_vel, launch_vel))
            reward -= 0.01*np.linalg.norm(joint_ang_vels[np.where(np.abs(joint_ang_vels) > _EC.Q_DOT_MAX)], ord=2)
                            
        if releasing:
            bonus += 10.0 / (1.0 + reward_functions.ts_ss_similarity(dart_pos, launch_pt) + reward_functions.ts_ss_similarity(dart_vel, launch_vel))
            penalty -= 0.1*reward_functions.distance(dart_pos, launch_pt)
            penalty -= 0.1*reward_functions.distance(dart_vel, launch_vel)

        if released:
            reward += 1.0 / (1.0 + np.min(np.linalg.norm(dart_pos - self.launch_traj)))

        reward = np.tanh(0.01 * reward)

        terminal_reward = 0.0
        terminal, fail_penalty = self.terminal(dart_pos)

        if terminal:
            terminal_reward -= reward_functions.distance(dart_pos, goal)
            terminal_reward += reward_functions.capped_inverse_distance(dart_pos, goal)
            terminal_reward += 10.0 * reward_functions.on_dart_board(dart_pos)
            terminal_reward += 100.0 * reward_functions.close_enough(dart_pos, goal)

            terminal_reward = 2.5 * np.tanh(0.01 * terminal_reward)

        return (terminal, NP_DTYPE(reward + bonus + penalty + fail_penalty + terminal_reward))

    @typechecked
    def sparse_reward(self, state: NP_ARRTYPE) -> tuple[bool, NP_DTYPE]:
        (_, _, _, _, releasing, goal, launch_pt, launch_vel, dart_pos,
         dart_vel) = self.decompose_state(state)

        reward = -1.0 

        if releasing:
            reward += reward_functions.close_enough(dart_pos, launch_pt)*reward_functions.close_enough(dart_vel, launch_vel)

        terminal, _ = self.terminal(dart_pos)
        if terminal:
            reward += reward_functions.close_enough(dart_pos, goal)

        return terminal, NP_DTYPE(reward)


    def terminal(self, dart_pos):
        terminal = True
        penalty = -10.0
        if self.out_of_time(): pass
        elif self.too_far_left_or_right(dart_pos[0]): pass
        elif self.too_far_back(dart_pos[1]): pass
        elif self.too_far_up_or_down(dart_pos[2]): pass
        elif self.far_enough_forward(dart_pos[1]):
            penalty = 0.0
        else:
            penalty = 0.0
            terminal = False
        return (terminal, penalty)

    # Reset the model in the simulator # TODO: make cleaner
    def reset_model(self) -> np.ndarray:

        # Set initial joint angles and compute forward kinematics
        noise_shape = (len(Poses.q0), )
        qvel = np.zeros((self.model.nv, ),
                        dtype=NP_DTYPE)  #+ np.random.uniform(-0.05, 0.05, size=(self.model.nv, ))
        initial_joint_angles = Poses.q0 + np.random.uniform(-0.2, 0.2, size=noise_shape)
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

        self._reset_goals()
        self.baseline_controller.reset_controller()
        self.released = False
        self.releasing = False
        self.state = self.observe()

        assert (self.state.shape == (_EC.NUM_OBSERVABLE_STATES, ))
        return self.state

    def dart_release(self, release: bool):
        if not self.released and release:
            self.model.eq("weld").active = False
            self.released = True
            self.releasing = True
        else:
            self.releasing = False

        # Visualization
        gripper_color = self.model.geom("placeholder_gripper").rgba
        if self.released:
            gripper_color[:] = np.array([1, 0.5, 0, 1])  # orange
            if release: gripper_color[:] = np.array([1, 0, 0, 1])  # red
        else: gripper_color[:] = np.array([0, 1, 0, 1])  # green

    # WARNING: MAJOR CLEANUP NEEDED BELOW
    #----------------------------------------------------------#
    #----------------------------------------------------------#
    #----------------------------------------------------------#
    #----------------------------------------------------------#
    #----------------------------------------------------------#
    
    def pre_explore(self, replay_buffer):
        self._reset_simulation()
        self._add_marker(pos=np.array([0, 0, 2.4]), color=[0.5, 0.5, 0.2, 1.0], label="Uniform Pre-Exploration", size=0.01)
        state = self.reset_to_random_state()
        cont_action = self.action_space.sample()[0:-1]
        disc_action = np.random.choice([0, 1])
        full_tanh_action = np.concatenate([np.tanh(cont_action), np.array([disc_action])], axis=0, dtype=NP_DTYPE)
        next_state, reward, done, info = self.step(cont_action, disc_action)
        replay_buffer.store_transition(state, full_tanh_action, reward, next_state, done)

    def _reset_not_released(self, initial_joint_angles):
        self.set_state(np.array([*initial_joint_angles, *([0.0] * 7)], dtype=NP_DTYPE), np.zeros(self.model.nv))

        # Compute initial dart position
        release_point = self.data.body("temporary_release_mechanism")
        relative_offset = np.array([0.1, 0.0, 0.0], dtype=NP_DTYPE)
        initial_dart_position = (release_point.xpos + release_point.xmat.reshape(
            (3, 3)) @ relative_offset).tolist()

        # Compute initial dart orientation
        minus_90_deg_rotation_y = np.array([0.70710678118, 0, -0.70710678118, 0])
        initial_dart_orientation = self.data.body("temporary_release_mechanism").xquat
        mju_mulQuat(initial_dart_orientation, initial_dart_orientation, minus_90_deg_rotation_y)

        self._set_weld_relpose(relative_offset, minus_90_deg_rotation_y, "weld")
        self.model.eq("weld").active = True

        return initial_dart_position, initial_dart_orientation, np.zeros(6)

    def _reset_released(self):
        dart_pos = np.random.uniform([_EC.X_MIN, _EC.Y_MIN, _EC.Z_MIN], [_EC.X_MAX, _EC.Y_MAX, _EC.Z_MAX])
        dart_vel = np.random.uniform([-5, -5, -5, -1, -1, -1], [5, 5, 5, 1, 1, 1])
        dart_ori = np.random.normal(size=4)
        dart_ori /= np.linalg.norm(dart_ori)
        self.model.eq("weld").active = False 

        return dart_pos, dart_ori, dart_vel

    def reset_to_random_state(self) -> np.ndarray:
        qpos_joints = np.array([np.random.uniform(_EC.Q_MIN[i], _EC.Q_MAX[i]) for i in range(_EC.NUM_JOINTS)],
                               dtype=NP_DTYPE)
        qvel_joints = np.array(
            [np.random.uniform(_EC.Q_DOT_MIN[i], _EC.Q_DOT_MAX[i]) for i in range(_EC.NUM_JOINTS)], dtype=NP_DTYPE)
        self.released = np.random.choice([True, False])
        self.releasing = False

        if self.released:
            dart_pos, dart_ori, dart_vel = self._reset_released()
        else:
            dart_pos, dart_ori, dart_vel = self._reset_not_released(qpos_joints)

        qpos_full = np.concatenate([qpos_joints, dart_pos, dart_ori.tolist()], dtype=NP_DTYPE)
        qvel_full = np.concatenate([qvel_joints, dart_vel], dtype=NP_DTYPE)

        self.baseline_controller.reset_controller()

        self.set_state(qpos_full, qvel_full)
        self.do_simulation(np.zeros(7), 1)

        self._reset_goals()
        self.state = self.observe()

        return self.state

    def _reset_goals(self):
        self.goal = _EC.BULLSEYE + np.random.uniform(-_EC.BOARD_RADIUS, _EC.BOARD_RADIUS, self.goal.shape)
        base_pt = np.array([0.2, -0.1, 2.55])
        args = (base_pt, 0.2, 0.4, 0.2, 10, self.goal, 0.0, 1.0, 10)
        launch_combos = launch_pairs(*args)
        min_idx = np.argmin(np.linalg.norm(launch_combos[:, 1, :], axis=1), axis=0)
        self.launch_pt, self.launch_vel = launch_combos[min_idx, 0, :], launch_combos[min_idx, 1, :]
        self.launch_traj = trajectory(self.goal[1], self.launch_pt, self.launch_vel, 0.6, self.launch_traj_res)

    # Custom add_marker() function because MujocoEnv base class is implemented a bit badly
    def _add_marker(self, pos, color=[1, 1, 1, 1], label="Marker", size=0.05):
        viewer = self.mujoco_renderer._get_viewer(self.render_mode)  # type: ignore
        viewer.add_marker(pos=pos, label=label, rgba=color, size=size * np.ones(3))

    # custom render() function to manage scn.ngeom
    def _render(self):
        self._add_marker(self.goal, color=[0, 0, 1, 0.5], label="Goal", size=0.03)
        self._add_marker(self.launch_pt, color=[0.2, 0.7, 0.05, 0.5], label="P*", size=0.03)
        self._add_marker(self.launch_pt + 0.1 * self.launch_vel,
                         color=[0.2, 0.7, 0.05, 0.5],
                         label="V*",
                         size=0.03)

        (*_, dart_pos, dart_vel) = self.decompose_state(self.state)
        self._add_marker(dart_pos, color=[0.2, 0.7, 0.6, 0.35], label="p", size=0.03)
        self._add_marker(dart_pos + 0.1*dart_vel, color=[0.2, 0.3, 0.05, 0.35], label="v", size=0.015)

        if self.display_corners:
            self._display_corners()

        if self.display_traj:
            self._display_traj()

        self.render()

    def _display_corners(self):
        c = [1, 0, 0, 1]
        self._add_marker(np.array([_EC.X_MAX, _EC.Y_MAX, _EC.Z_MAX]), color=c, label="+ + +", size=0.03)
        self._add_marker(np.array([_EC.X_MAX, _EC.Y_MAX, _EC.Z_MIN]), color=c, label="+ + -", size=0.03)
        self._add_marker(np.array([_EC.X_MAX, _EC.Y_MIN, _EC.Z_MAX]), color=c, label="+ - +", size=0.03)
        self._add_marker(np.array([_EC.X_MAX, _EC.Y_MIN, _EC.Z_MIN]), color=c, label="+ - -", size=0.03)
        self._add_marker(np.array([_EC.X_MIN, _EC.Y_MAX, _EC.Z_MAX]), color=c, label="- + +", size=0.03)
        self._add_marker(np.array([_EC.X_MIN, _EC.Y_MAX, _EC.Z_MIN]), color=c, label="- + -", size=0.03)
        self._add_marker(np.array([_EC.X_MIN, _EC.Y_MIN, _EC.Z_MAX]), color=c, label="- - +", size=0.03)
        self._add_marker(np.array([_EC.X_MIN, _EC.Y_MIN, _EC.Z_MIN]), color=c, label="- - -", size=0.03)

    def _display_traj(self):
        c = [0.2, 0.7, 0.6, 0.35]
        for pt in self.launch_traj:
            self._add_marker(pt, color=c, label="", size=0.03)

    # Sets the relative pose used to compute weld constaints, this will probably not be necessary in the future
    def _set_weld_relpose(self, pos: np.ndarray, quat: np.ndarray, name: str):
        pos_idx, quat_idx = 3, 6
        data = self.model.eq(name).data
        data[pos_idx:pos_idx + 3] = np.copy(pos)
        data[quat_idx:quat_idx + 4] = np.copy(quat)

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    # Helpers for termination criteria
    def too_far_left_or_right(self, dart_x):
        return not (_EC.X_MIN <= dart_x and dart_x <= _EC.X_MAX)

    def too_far_back(self, dart_y):
        return not (dart_y <= _EC.Y_MAX)

    def far_enough_forward(self, dart_y):
        return not (self.goal[1] <= dart_y)

    def too_far_up_or_down(self, dart_z):
        return not (_EC.Z_MIN <= dart_z and dart_z <= _EC.Z_MAX)

    def out_of_time(self):
        return self.data.time >= self.time_limit


# ---------------------------------------------------------------------------- #
