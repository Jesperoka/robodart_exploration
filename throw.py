import threading
import time

import numpy as np

import panda_py
from panda_py import controllers, libfranka

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

pos0 = np.array([ 0   , -1.5,  0.  , -0.3,  0.  , 2.3,  0.79])
pos1 = np.array([ 0.  , 0.5,  0.  , -2,  0.  , 1.57,  0.79])

if __name__ == "__main__":

    panda = panda_py.Panda(SHOP_FLOOR_IP)
    gripper = libfranka.Gripper(SHOP_FLOOR_IP)
    #gripper.homing()
    panda.move_to_start()

    def move(panda, runtime):

        panda.move_to_start()       

        with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
            while ctx.ok():
                time.sleep(1)
                panda.move_to_joint_position(pos1, speed_factor=0.3)
                panda.move_to_joint_position(pos0, speed_factor = 0.50)
                break

    def grasp(gripper, runtime):
        gripper.grasp(0, 0.2, 20)
        time.sleep(5)
        gripper.move(0.08, 0.2)

    runtime = 10

    threading.Thread(target=move, args=(panda, runtime)).start()
    threading.Thread(target=grasp, args=(gripper, runtime)).start()

    


    # Move to the arm one joint at a time passing only a single 7x1 np.ndarray
    #gripper.grasp(0.02, 0.2, 10, 0.04, 0.04)
    #panda.move_to_joint_position(pos0)
    #panda.move_to_joint_position(pos1)



    # Move back to the neutral position

