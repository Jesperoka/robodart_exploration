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

pos0 = np.array([-2.4561541090417456, -0.4201926019669851, -1.4144669946047714, -1.8604822223228317, 1.3630775381988949, 2.4335151671568553, 0.9596177520143658]
)
#pos1 = np.array([-0.30387781099478406, 0.6064244038315066, -0.05066064980098308, -0.9506315781359087, 1.4807728046046362, 3.0872235901295926, 0.9687159340925927]
#)

pos1 = np.array([0.4274162864245866, 0.6441666006666874, 0.16723876531291426, -0.7891901614791414, 1.4471768309572592, 2.1269190367497903, 0.9480698071764831])

speed = np.array([1, 1, 1, 1, 1, 1, 1])

if __name__ == "__main__":

    panda = panda_py.Panda(SHOP_FLOOR_IP)
    gripper = libfranka.Gripper(SHOP_FLOOR_IP)
    #gripper.homing()
    panda.move_to_start()

    def move(panda, runtime):
        ctrl = controllers.JointPosition(filter_coeff=1.0)
        x0 = panda.get_position()
        q0 = panda.get_orientation()
        panda.start_controller(ctrl)

        panda.move_to_start()

        

        with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
            while ctx.ok():
                #ctrl.set_control(pos0, speed)
                time.sleep(1)
                panda.move_to_joint_position(pos0, speed_factor=0.3)
                panda.move_to_joint_position(pos1, speed_factor = 0.65)
                break
        #         x_d = x0.copy()
        #         x_d[1] += 0.2 * np.sin(ctrl.get_time())
        #         ctrl.set_control(x_d, q0)

    def grasp(gripper, runtime):
        gripper.grasp(0, 0.2, 20)
        time.sleep(5.85)
        gripper.move(0.08, 0.2)

    runtime = 10

    threading.Thread(target=move, args=(panda, runtime)).start()
    threading.Thread(target=grasp, args=(gripper, runtime)).start()

    


    # Move to the arm one joint at a time passing only a single 7x1 np.ndarray
    #gripper.grasp(0.02, 0.2, 10, 0.04, 0.04)
    #panda.move_to_joint_position(pos0)
    #panda.move_to_joint_position(pos1)



    # Move back to the neutral position

