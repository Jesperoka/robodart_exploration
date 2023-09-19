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

pos0 = np.array([-0.05826081972320875, 1.518139931452203, 0.015952493496804526, -1.9738094989538244, -0.00664670701821645, 2.1474642659134724, 0.8557886689255634])
pos1 = np.array([-0.05363916656321649, -0.19706883226558733, 0.06755293216203388, -0.08941960363011611, -0.1320708698137767, 3.644003728793108, 0.7782894581576624])

speed = np.array([1, 1, 1, 1, 1, 1, 1])

if __name__ == "__main__":

    panda = panda_py.Panda(SHOP_FLOOR_IP)
    gripper = libfranka.Gripper(SHOP_FLOOR_IP)
    gripper.homing()
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
                panda.move_to_joint_position(pos1, speed_factor = 0.7)
                break
        #         x_d = x0.copy()
        #         x_d[1] += 0.2 * np.sin(ctrl.get_time())
        #         ctrl.set_control(x_d, q0)

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

