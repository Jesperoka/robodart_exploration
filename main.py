import threading
import time
import numpy as np

import panda_py as pp
import panda_py.libfranka as lf
import panda_py.controllers as contr

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

THROW_MAX_RUNTIME = 100.0
SPEED_FACTOR = 0.80

# Hardcoded trajectory
START_POSE = np.array([0.05088351893545949, -1.2975023430774084, 0.3135964171493919, 
                       -0.07528110453107005, 1.2732888533539242, 1.6346095751921337, 
                       -0.50029037349919476])

WAYPOINT_POSE = np.array([0.07555643242092827, -0.6085925320160931, 0.3017500475582323,
                          -0.14378755696143414, 1.273197977375101, 1.6340747307936347,
                          0.4962575651532825])

LAUNCH_POSE = np.array([0.12380197155073044, -0.06677692938461456, 0.1462209859605421, 
                        -0.10090838005985799, 1.263665055033812, 1.6197725406487782, 
                        0.7696788805077474])

# Don't use follow though if using catapult-like gripper extension
FOLLOW_THROUGH_POSE = np.array([-0.1303975102995331, 0.8493790016434748, 0.08240450900957766, 
                                -1.4597037342138457, 1.4373264386124078, 1.6275337981383007, 
                                0.8712892756213744])

# Arm commands 
def setup(panda, gripper) -> contr.TorqueController:
    gripper.move(0.08, 0.2)
    panda.move_to_start()

def throw(panda, max_runtime):
    panda.move_to_joint_position(START_POSE, speed_factor=0.4)

    print("\n\nCaution: arm in throwing position.\n\n")
    input("\n\n\nPLACE GRIPPER EXTENSION IN GRIPPER AND PRESS ENTER\n\n\n")

    gripper.grasp(0.045, 0.1, 70, 0.01, 0.01)
    input("\n\n\nPRESS ENTER TO VERIFY GRASPING\n\n\n")
    print("\n\n\nis_grasped:", gripper.read_once().is_grasped, "\n\n\n")

    print("\n\n\nPLACE PROJECTILE IN GRIPPER\n\n\n") 
    input("\n\n\nPRESS ENTER TO THROW\n\n\n")

    with panda.create_context(frequency=1e3, max_runtime=max_runtime) as ctx:
        while ctx.ok():
            panda.move_to_joint_position([START_POSE, LAUNCH_POSE], speed_factor=SPEED_FACTOR)
            break

if __name__ == "__main__":

    panda = pp.Panda(SHOP_FLOOR_IP)
    gripper = lf.Gripper(SHOP_FLOOR_IP)

    setup(panda, gripper)
    throw(panda, THROW_MAX_RUNTIME)