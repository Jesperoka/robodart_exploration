import numpy as np
import matplotlib.pyplot as plt
from utils.dtypes import NP_DTYPE
from utils.common import unpack_dataclass
from rl_environment.constants import EnvConsts, Poses

def zero_controller(dart_pos, goal_pos):
    return np.zeros(7, dtype=NP_DTYPE)

# poses = np.array([pose0, pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8, pose9])
degrees = [1, 3, 1, 2, 1, 1, 2]


class LookaheadController:
    def __init__(self):
        self.e_int = 0.0 
        self.Kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.Ki = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.Kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        self.dt = 1e-3 
        self.lookahead_distance = 0.5  # adjust as needed
        self.poses = np.array(unpack_dataclass(Poses))
        # self.polys = [np.poly1d(np.polyfit(np.arange(len(self.poses)), self.poses[:,i], degrees[i])) for i in range(7)]
        self.do_log = False
        self.log = {
            "qpos": [],
            "qpos_ref": [],
            "qvel": [],
            # "qvel_ref": [],
            "e": [],
            "e_int": [],
            "e_dot": [],
            "torques": [],
        }

    def reset_controller(self):
        self.e_int = 0.0
    
    def interpolate_pose(self, s):
        s_values = np.linspace(0, len(self.poses) - 1, len(self.poses))
        interpolated_pose = []
        for i in range(EnvConsts.NUM_JOINTS):  # for each joint
            joint_values = self.poses[:, i]
            interpolated_pose.append(np.interp(s, s_values, joint_values))
        return np.array(interpolated_pose)

    def find_lookahead_point(self, current_pose):
        s = len(self.poses) - 1
        while s > 0.0:
            lookahead_pose = self.interpolate_pose(s)
            masked_diff = np.ma.masked_array(current_pose - lookahead_pose, mask=[0, 1, 0, 1, 0, 0, 1])
            if np.linalg.norm(masked_diff) <= self.lookahead_distance:
                return lookahead_pose
            s -= 0.005  
        return self.poses[-1]

    def __call__(self, qpos, qvel): # TODO: clamp torques
        qpos_ref = self.find_lookahead_point(qpos)
        # qvel_ref = np.array([0, -2.1750, 0, 0, 0, 0, 2.6100])
        qvel_ref = np.zeros(7)

        e = qpos_ref - qpos     # P
        self.e_int += self.dt*e # I
        e_dot = qvel_ref - qvel # D

        print("e: ", e, "\ne_int:", self.e_int, "\ne_dot:", e_dot)
        if (EnvConsts.Q_DOT_MIN >= qvel).any() or (qvel >= EnvConsts.Q_DOT_MAX).any():
            print("\n!!! joint velocity limits exceeded !!!\n")

        torques = (self.Kp*e + self.Ki*self.e_int + self.Kd * e_dot).squeeze().astype(NP_DTYPE)
        
        if self.do_log:
            self.log["qpos"].append(qpos.copy())
            self.log["qpos_ref"].append(qpos_ref)
            self.log["qvel"].append(qvel.copy())
            # self.log["qvel_ref"].append(qvel_ref)
            self.log["e"].append(e)
            self.log["e_int"].append(self.e_int)
            self.log["e_dot"].append(e_dot)
            self.log["torques"].append(torques)

        return torques

    def get_log(self):
        return self.log

    def clear_log(self):
        self.log.clear()

    def plot_logged(self):
        """Plot the logged values."""
        _, axes = plt.subplots(4, len(self.poses[0]), figsize=(15, 12))

        # Iterate over each joint
        for i in range(len(self.poses[0])):
            # Plot qpos and qpos_ref
            axes[0, i].plot(np.array(self.log["qpos"])[:, i], label='qpos')
            axes[0, i].plot(np.array(self.log["qpos_ref"])[:, i], label='qpos_ref', linestyle='--')
            axes[0, i].set_title(f"Joint {i+1} Position")
            axes[0, i].legend()
            
            # Plot qvel
            axes[1, i].plot(np.array(self.log["qvel"])[:, i], label='qvel')
            axes[1, i].axhline(y=EnvConsts.Q_DOT_MAX[i], color='r', linestyle='--', label='Q_DOT_MAX')
            axes[1, i].axhline(y=EnvConsts.Q_DOT_MIN[i], color='b', linestyle='--', label='Q_DOT_MIN')
            axes[1, i].set_title(f"Joint {i+1} Velocity")
            axes[1, i].legend()

            # Plot error e
            axes[2, i].plot(np.array(self.log["e"])[:, i], label='e')
            axes[2, i].set_title(f"Joint {i+1} Error (e)")
            axes[2, i].legend()
            
            # Plot torques
            axes[3, i].plot(np.array(self.log["torques"])[:, i], label='torque')
            axes[3, i].set_title(f"Joint {i+1} Torque")
            axes[3, i].legend()
        
        plt.tight_layout()
        plt.show()
