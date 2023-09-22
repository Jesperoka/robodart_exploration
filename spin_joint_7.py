import numpy as np
import panda_py as pp
import panda_py.libfranka as lf


TORQUE = 4.7 

REFERENCE_POS =  np.array([1.2535917688652172, 0.6972054325100352, 0.38962657131646805, 
                           -1.228720040789675, -0.24577336054378085, 3.436912040630976, 
                           2.358766003196306])

def spin_test_callback(robot_state, duration):

    Kp = 300.0
    Kd = 10.0

    e = REFERENCE_POS - np.array([robot_state.q])
    e_dot = np.zeros(7) - np.array([robot_state.dq])

    torques = Kp * e + Kd * e_dot 
    torques[0, -1] = -TORQUE # joint 7
    torques[0, -3] = -TORQUE # joint 5
    torques[0, 2] = -3.0*TORQUE # joint 3
    torques[0, 0] = -3.0*TORQUE # joint 1

    return lf.Torques(torques.squeeze())