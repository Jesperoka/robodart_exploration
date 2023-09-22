import panda_py
from spin_joint_7 import spin_test

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":

    print("Connecting to Desk")

    # Connect to the robot
    desk = panda_py.Desk(SHOP_FLOOR_IP, username, password)

    input("Press any key to unlock FCI")

    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated

    input("Press any key to connect to Robot")

    panda = panda_py.Panda(SHOP_FLOOR_IP)

    torque1 = 0.09175
    torque2 = 0.1835
    torque3 = 0.3670
    torque4 = 0.7340
    torque5 = 1.468
    torque6 = 2.936
    torque7 = 5.872
    torque8 = 11.744
    torque9 = 12
    torque10 = 12.1

    spin_test(panda, torque1)

    #torque_joint_5
    #spin_test_move(panda, torque1, torque_joint_5)

