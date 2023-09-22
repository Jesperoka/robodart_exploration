import numpy as np
import panda_py as pp
import panda_py.libfranka as lf
from spin_joint_7 import spin_test_callback, REFERENCE_POS

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

FREQUENCY = 1e3
MAX_RUNTIME = 10.0

EXTERNAL_FORCE_THRESHOLD = [50.0]*6
MINUS_EXTERNAL_FORCE_THRESHOLD = [-50.0]*6

EXTERNAL_TORQUE_THRESHOLD = [50.0]*7
MINUS_EXTERNAL_TORQUE_THRESHOLD = [-50.0]*7

print()

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":

    # Connect to the robot
    desk = pp.Desk(SHOP_FLOOR_IP, username, password)

    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated

    pp.Panda(SHOP_FLOOR_IP).move_to_joint_position(REFERENCE_POS, speed_factor=0.1)
    ctx = pp.Panda(SHOP_FLOOR_IP).create_context(frequency=FREQUENCY, max_runtime=MAX_RUNTIME)
    panda = lf.Robot(SHOP_FLOOR_IP)

    input("ENTER TO START CONTROL LOOP")
    
    with ctx:
        while ctx.ok():
            panda.set_collision_behavior(
                lower_force_thresholds=MINUS_EXTERNAL_FORCE_THRESHOLD,
                upper_force_thresholds=EXTERNAL_FORCE_THRESHOLD,
                lower_torque_thresholds=MINUS_EXTERNAL_TORQUE_THRESHOLD,
                upper_torque_thresholds=EXTERNAL_TORQUE_THRESHOLD,
                                         )
            panda.control(spin_test_callback)

    # torque1 = 0.09175
    # torque2 = 0.1835
    # torque3 = 0.3670
    # torque4 = 0.7340
    # torque5 = 1.468
    # torque6 = 2.936
    # torque7 = 5.872
    # torque8 = 11.744
    # torque9 = 12
    # torque10 = 12.1

    # input("Run test1, torque = "+ str(torque1))
    # spin_test(panda, torque1)
    # input("Run test2, torque = "+ str(torque2))
    # spin_test(panda, torque2)
    # input("Run test3, torque = "+ str(torque3))
    # spin_test(panda, torque3)
    # input("Run test4, torque = "+ str(torque4))
    # spin_test(panda, torque4)
    # input("Run test5, torque = "+ str(torque5))
    # spin_test(panda, torque5)
    # input("Run test6, torque = "+ str(torque6))
    # spin_test(panda, torque6)
    # input("Run test7, torque = "+ str(torque7))
    # spin_test(panda, torque7)
    # input("Run test8, torque = "+ str(torque8))
    # spin_test(panda, torque8)
    # input("Run test9, torque = "+ str(torque9))
    # spin_test(panda, torque9)
    # input("Run test10, torque = "+ str(torque10))
    # spin_test(panda, torque10)


    #torque_joint_5
    #spin_test_move(panda, torque1, torque_joint_5)

