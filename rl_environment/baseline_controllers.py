import numpy as np
import matplotlib.pyplot as plt
from utils.dtypes import NP_DTYPE
from utils.common import unpack_dataclass
from rl_environment.constants import EnvConsts, Poses
from copy import copy


# Convenience Logging
# ---------------------------------------------------------------------------- #
class ControlLog():
    do_log = False
    log = {"qpos": [], "qpos_ref": [], "qvel": [], "qvel_ref": [], "e": [], "e_int": [], "e_dot": [], "torques": []}
    def plot_logged(self):
        fig, axes = plt.subplots(4, EnvConsts.NUM_JOINTS, figsize=(15, 12))
        # For each joint
        for i in range(EnvConsts.NUM_JOINTS):
            # Plot qpos and qpos_ref
            axes[0, i].plot(np.array(self.log["qpos"])[:, i], label='qpos')
            axes[0, i].plot(np.array(self.log["qpos_ref"])[:, i], label='qpos_ref', linestyle='--')
            axes[0, i].set_title(f"Joint {i+1} Position")
            # Plot qvel
            axes[1, i].plot(np.array(self.log["qvel"])[:, i], label='qvel')
            axes[1, i].axhline(y=EnvConsts.Q_DOT_MAX[i], color='r', linestyle='--')
            axes[1, i].axhline(y=EnvConsts.Q_DOT_MIN[i], color='r', linestyle='--')
            axes[1, i].set_title(f"Joint {i+1} Velocity")
            # Plot error e
            axes[2, i].plot(np.array(self.log["e"])[:, i], label='e')
            axes[2, i].set_title(f"Joint {i+1} Error (e)")
            # Plot torques
            axes[3, i].plot(np.array(self.log["torques"])[:, i], label='torque')
            axes[3, i].axhline(y=EnvConsts.TAU_MAX[i], color='r', linestyle='--')
            axes[3, i].axhline(y=EnvConsts.TAU_MIN[i], color='r', linestyle='--')
            axes[3, i].set_title(f"Joint {i+1} Torque")
        fig.legend()
        plt.tight_layout()
        plt.show()
    def get_log(self): return self.log
    def clear_log(self): self.log.clear()
    def append_to_log(self, qpos, qpos_ref, qvel, qvel_ref, e, e_int, e_dot, torques):
        self.log["qpos"].append(qpos.copy())
        self.log["qpos_ref"].append(qpos_ref)
        self.log["qvel"].append(qvel.copy())
        self.log["qvel_ref"].append(qvel_ref)
        self.log["e"].append(e)
        self.log["e_int"].append(e_int)
        self.log["e_dot"].append(e_dot)
        self.log["torques"].append(torques)
# ---------------------------------------------------------------------------- #
    

# Pure Pursuit Lookahead Controller based on Interpolated Trajectory
# ---------------------------------------------------------------------------- #
class LookaheadController(ControlLog):
    def __init__(self, dt=None):
        self.e_int = 0.0 
        self.s0 = 0.0
        self.t = 0.0
        self.Kp = 2.0*np.array([100.0, 100.0, 100.0, 95.0, 100.0, 100.0, 150.0])
        self.Ki = 0.5*np.array([20.0, 1.0, 20.0, 1.0, 20.0, 20.0, 1.0])
        self.Kd = np.array([10.0, 40.0, 10.0, 40.0, 10.0, 10.0, 45.0])
        self.dt = 1e-3 if dt == None else dt
        self.lookahead_distance = 0.1  # adjust as needed
        self.poses = np.array(unpack_dataclass(Poses))
        self._setup_interpolation()

    def _setup_interpolation(self, resolution=1000):
        s_values = np.linspace(0, len(self.poses) - 1, len(self.poses))
        self.dense_s = np.linspace(0, len(self.poses) - 1, resolution)  # You can adjust the resolution
        self.lookup_table = np.array([np.interp(self.dense_s, s_values, self.poses[:, i]) for i in range(EnvConsts.NUM_JOINTS)]).T

    def interpolate_pose(self, s):
        index = np.argmin(np.abs(self.dense_s - s)) # TODO: just use indices instead of s since lookup_table
        return self.lookup_table[index]

    def reset_controller(self):
        self.e_int = 0.0
        self.s0 = 0.0
        self.t = 0.0
        self.lookahead_distance = 0.6
    
    def find_lookahead_point(self, current_pose, s0):
        s = s0 
        while s < len(self.poses) - 1: # TODO: just use indices instead of s since lookup_table
            lookahead_pose = self.interpolate_pose(s)
            masked_diff = np.ma.masked_array(current_pose - lookahead_pose, mask=[0, 1, 0, 1, 0, 0, 1])
            if np.linalg.norm(masked_diff) >= self.lookahead_distance:
                return lookahead_pose, s
            s += 0.01  
        return self.poses[-1], s

    def __call__(self, qpos, qvel): # TODO: clamp torques
        self.lookahead_distance += 1.5*self.dt if self.lookahead_distance + self.dt <= 0.9 else 0.0
        qpos_ref, self.s0 = self.find_lookahead_point(qpos, copy(self.s0))
        qvel_ref = np.array([0, -0.35, 0, 0.0, 0, 0, 2.0100])

        e = qpos_ref - qpos     # P
        self.e_int += self.dt*e # I
        e_dot = qvel_ref - qvel # D

        torques = (self.Kp*e + self.Ki*self.e_int + self.Kd * e_dot).squeeze()
        
        if self.do_log:
            self.append_to_log(qpos, qpos_ref, qvel, qvel_ref, e, self.e_int, e_dot, torques)

        return np.clip(torques, a_min=EnvConsts.A_MIN[0:EnvConsts.NUM_JOINTS], a_max=EnvConsts.A_MAX[0:EnvConsts.NUM_JOINTS]).astype(NP_DTYPE)
# ---------------------------------------------------------------------------- #


# PID Position Control of Joints 2, 4 and 7
# ---------------------------------------------------------------------------- #
class SelectedJointsPID(ControlLog):
    def __init__(self):
        # Variables that need resetting
        self.t_prev = 0.0
        self.e_prev = 0.0
        self.e_int = 0.0 

        # Predetermined Control Parameters
        self.Kp = np.array([200.0, 100, 200.0, 100, 100.0, 100.0, 100])
        self.Ki = np.array([20.0,  0, 20.0,  0, 10.0,  10.0,  0])
        self.Kd = np.array([10.0,  0, 10.0,  0, 10.0,  10.0,  0])

        # Predetermined joints
        self.qpos_ref_1 = Poses.q0[0]
        self.qpos_ref_3 = Poses.q0[2]
        self.qpos_ref_5 = Poses.q0[4]
        self.qpos_ref_6 = Poses.q0[5]

        # Control limits
        self.a_min = np.array([*EnvConsts.TAU_MIN]) 
        self.a_max = np.array([*EnvConsts.TAU_MAX]) 

    def reset_controller(self):
        self.t_prev = 0.0
        self.e_prev = 0.0
        self.e_int = 0.0

    def __call__(self, t, qpos, qvel, qpos_ref, qvel_ref=np.zeros(7)):
        dt = t - self.t_prev

        qpos_ref = np.array([self.qpos_ref_1, qpos_ref[0], self.qpos_ref_3, qpos_ref[1], self.qpos_ref_5, self.qpos_ref_6, qpos_ref[2]])

        e = qpos_ref - qpos                                     # P
        self.e_int += dt*(self.e_int + 0.5*(self.e_prev + e))   # I (explicit midpoint rule)
        e_dot = qvel_ref - qvel                                 # D

        torques = (self.Kp*e + self.Ki*self.e_int + self.Kd*e_dot).squeeze()
        
        self.t_prev = t
        self.e_prev = e
        
        if self.do_log:
            self.append_to_log(qpos, qpos_ref, qvel, qvel_ref, e, self.e_int, e_dot, torques)

        return np.clip(torques, a_min=self.a_min, a_max=self.a_max).astype(NP_DTYPE)
# ---------------------------------------------------------------------------- #
