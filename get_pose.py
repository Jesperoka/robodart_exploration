import threading
import time

import numpy as np

import panda_py
from panda_py import controllers, libfranka

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

if __name__ == "__main__":

    panda = panda_py.Panda(SHOP_FLOOR_IP)
    pose = panda.get_state()
    print(pose.q)


