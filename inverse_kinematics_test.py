import numpy as np
import panda_py
from panda_py import controllers, libfranka


SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP


################################################
robot = libfranka.Robot(SHOP_FLOOR_IP)
state = libfranka.Robot.read_once(robot)
model = libfranka.Robot.load_model(robot)
EE_frame = libfranka.Frame.kEndEffector
pose_of_EE_in_base = np.reshape(model.pose(EE_frame, state), (4, 4)).T


print(state.q)
print(pose_of_EE_in_base)

cartesian_pose = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [-0.26, -0.2, 1.2, 1]])


# cartesian_pose = np.array([[0.03815925, -0.03711592, -0.99858211, -0.20445222],
#                             [-0.69274338, 0.71920556, -0.053205, -0.05161481],
#                             [0.72016055, 0.69379141, 0.00173257, 0.90406091],
#                             [0, 0, 0, 1]])

cartesian_pose = np.array([[0.03815925, -0.03711592, -0.99858211, -0.26],
                            [-0.69274338, 0.71920556, -0.053205, -0.2],
                            [0.72016055, 0.69379141, 0.00173257, 1.2],
                            [0, 0, 0, 1]])

joint_pose = panda_py.ik(cartesian_pose, state.q, -0.69036709)
print(joint_pose)
#################################################
