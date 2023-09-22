import numpy as np

from panda_py import controllers

def spin_test(panda, torque):
    pos1 =  np.array([ 0.  , 0.5,  0.  , -2,  0.  , 1.57,  -2.89]) #Change to good test-pos
    panda.move_to_joint_position(pos1, speed_factor=0.2)
    ctrl = controllers.AppliedTorque(filter_coeff=1.0)
    ctrl.set_control([0, 0, 0, 0, 0, 0, torque])
    panda.start_controller(ctrl)



