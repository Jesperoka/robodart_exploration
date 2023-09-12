import panda_py
import numpy as np

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP


def main():
    # with open('sens.txt', 'r') as file:
    #     username = file.readline().strip()
    #     password = file.readline().strip()

    print("Connecting to Desk")

    # Connect to the robot
    desk = panda_py.Desk(SHOP_FLOOR_IP, "admin", "utsDoWnsfORvielemEra")

    #input("Press any key to unlock FCI")
    print("AAAAAAAAAAAAA")

    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated

    #input("Press any key to connect to Robot")

    panda = panda_py.Panda(SHOP_FLOOR_IP)

    #input("Press any key to move arm")

    # Move to panda_py.constants.JOINT_POSITION_START
    panda.move_to_start()

    # Define some joint positions pretty close to panda_py.constants.JOINT_POSITION_START
    pos0 = np.array([-0.1 , -0.79,  0.  , -2.36,  0.  , 1.57,  0.79])
    pos1 = np.array([ 0.  , -0.79,  0.  , -2.36,  0.  , 1.57,  0.79])
    pos2 = np.array([ 0.1 , -0.79,  0.  , -2.36,  0.  , 1.57,  0.79])
    pos3 = np.array([ 0.1 , -0.69,  0.  , -2.36,  0.  , 1.57,  0.79])
    pos4 = np.array([ 0.1 , -0.69,  0.1 , -2.36,  0.  , 1.57,  0.79])
    pos5 = np.array([ 0.1 , -0.69,  0.1 , -2.26,  0.  , 1.57,  0.79])
    pos6 = np.array([ 0.1 , -0.69,  0.1 , -2.26,  0.1 , 1.57,  0.79])
    pos7 = np.array([ 0.1 , -0.69,  0.1 , -2.26,  0.1 , 1.67,  0.79])

    # Move to the arm one joint at a time passing only a single 7x1 np.ndarray
    panda.move_to_joint_position(pos0)
    panda.move_to_joint_position(pos1)
    panda.move_to_joint_position(pos2)
    panda.move_to_joint_position(pos3)
    panda.move_to_joint_position(pos4)
    panda.move_to_joint_position(pos5)
    panda.move_to_joint_position(pos6)
    panda.move_to_joint_position(pos7)

    # Or pass a list of 7x1 np.ndarrays as waypoints for a motion
    waypoints = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
    panda.move_to_joint_position(waypoints)

    # Move back to the neutral position
    panda.move_to_start()

if __name__ == "__main__":
    main()