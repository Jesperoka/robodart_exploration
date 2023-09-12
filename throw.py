from panda_py import libfranka, Desk, Panda
import numpy as np

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":

    print("Connecting to Desk")

    # Connect to the robot
    desk = Desk(SHOP_FLOOR_IP, username, password)

    input("Press any key to unlock FCI")

    desk.unlock()       # don't need if already unlocked
    desk.activate_fci() # don't need if already activated

    input("Press any key to connect to Robot")

    panda = Panda(SHOP_FLOOR_IP)
    gripper = libfranka.Gripper(SHOP_FLOOR_IP)

    #panda.get_robot().set_cartesian_impedance([100, 100, 100, 100, 100, 100])
    input("Press any key to move arm")

    # Move to panda_py.constants.JOINT_POSITION_START
    #panda.move_to_start()
    #state = panda.get_state()
    #duration = libfranka.Duration(2000)
    #panda.get_robot().control((state, duration))
    

    # Define some joint positions pretty close to panda_py.constants.JOINT_POSITION_START
    pos0 = np.array([ 0   , -1.5,  0.  , 0,  0.  , 2.5,  0.79])
    pos1 = np.array([ 0.  , 0.5,  0.  , -2,  0.  , 1.57,  0.79])

    # Move to the arm one joint at a time passing only a single 7x1 np.ndarray
    #gripper.grasp(0.02, 0.2, 10, 0.04, 0.04)
    panda.move_to_joint_position(pos0)
    panda.move_to_joint_position(pos1)

    #gripper.move(0.09, 5)

    # Or pass a list of 7x1 np.ndarrays as waypoints for a motion
    #waypoints = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
    #waypoints = [pos0, pos1]
    #panda.move_to_joint_position(waypoints)


    # Move back to the neutral position
    panda.move_to_start()
