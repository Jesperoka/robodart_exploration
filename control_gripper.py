import panda_py
import numpy as np
from time import sleep

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

def main():
    with open('sens.txt', 'r') as file:
        username = file.readline().strip()
        password = file.readline().strip()

    print("Connecting to Desk")

    # Connect to the robot
    desk = panda_py.Desk(SHOP_FLOOR_IP, username, password)

    #input("Press any key to unlock FCI")

    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated

    #input("Press any key to connect to Robot")

    gripper = panda_py.libfranka.Gripper(SHOP_FLOOR_IP)

    #input("Press any key to move arm")

    gripper.grasp(0.02, 0.2, 10, 0.04, 0.04)
    #sleep(2)
    gripper.move(0.09, 5)

if __name__ == "__main__":
    main()