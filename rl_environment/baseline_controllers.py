import numpy as np
import matplotlib.pyplot as plt
from utils.dtypes import NP_DTYPE
from rl_environment.constants import EnvConsts, Poses


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
    

# PID Position Control of Selected Joints 
# ---------------------------------------------------------------------------- #
class SelectedJointsPID(ControlLog):
    def __init__(self):
        # Variables that need resetting
        self.t_prev = 0.0
        self.e_prev = 0.0
        self.e_int = 0.0 

        # Predetermined Control Parameters
        self.Kp = np.array([200.0, 100, 200.0, 100, 100.0, 100.0, 100])
        self.Ki = np.array([1.0,  1.0, 1.0,  1.0, 1.0,  1.0,  1.0])
        self.Kd = np.array([20.0,  20.0, 20.0,  20.0, 20.0,  20.0,  20.0])

        # Predetermined joints 
        self.qpos_ref_0 = Poses.q0[0]
        self.qpos_ref_2 = Poses.q0[2]
        self.qpos_ref_4 = Poses.q0[4]
        self.qpos_ref_5 = Poses.q0[5]

        # Control limits
        self.a_min = np.array([*EnvConsts.TAU_MIN]) 
        self.a_max = np.array([*EnvConsts.TAU_MAX]) 

    def reset_controller(self):
        self.t_prev = 0.0
        self.e_prev = 0.0
        self.e_int = 0.0

    def __call__(self, t, qpos, qvel, qpos_ref, qvel_ref=np.zeros(7)):
        dt = t - self.t_prev

        qpos_ref_0 = self.qpos_ref_0 + qpos_ref[0] # joint 0 is only allowed small deviations
        qpos_ref = np.array([qpos_ref_0, qpos_ref[1], self.qpos_ref_2, qpos_ref[2], self.qpos_ref_4, self.qpos_ref_5, qpos_ref[3]])

        e = qpos_ref - qpos                                     # P
        self.e_int += dt*(self.e_int + 0.5*(self.e_prev + e))   # I (explicit midpoint rule)
        e_dot = qvel_ref - qvel                                 # D # TODO: lowpass filter

        torques = (self.Kp*e + self.Ki*self.e_int + self.Kd*e_dot).squeeze()
        
        self.t_prev = t
        self.e_prev = e
        
        if self.do_log:
            self.append_to_log(qpos, qpos_ref, qvel, qvel_ref, e, self.e_int, e_dot, torques)

        return np.clip(torques, a_min=self.a_min, a_max=self.a_max).astype(NP_DTYPE)
# ---------------------------------------------------------------------------- #
