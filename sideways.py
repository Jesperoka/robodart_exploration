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
SPEED_FACTOR = 0.9

# Hardcoded trajectory
START_POSE = np.array([1.6368521356974775, 0.8021087540158054, 0.08699638480466734, -1.674536410967509, -1.5845267440271966, 1.4980160632928183, 0.9149081668605407])

WAYPOINT_POSE = np.array([1.7336436875222019, -1.4129802368615583, 0.05788243003424845, -2.2661295427355848, -1.5845225499206117, 1.4766486670887473, -0.6997704345633585])

LAUNCH_POSE = np.array([1.710020262523701, -1.0660343691230312, 0.009964245462757454, -0.4824696887312765, -1.65880014176184, 1.6023682567090807, -0.3327083705516739])

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

    desk = pp.Desk(SHOP_FLOOR_IP, username, password)
    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated
    panda = pp.Panda(SHOP_FLOOR_IP)
    gripper = lf.Gripper(SHOP_FLOOR_IP)


    setup(panda, gripper)
    throw(panda, THROW_MAX_RUNTIME)