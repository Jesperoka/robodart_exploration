from panda_py import Panda, Desk
import numpy as np

SHOP_FLOOR_IP = "10.0.0.2"

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

if __name__ == "__main__":
    #Desk(SHOP_FLOOR_IP, username, password).activate_fci()
    panda = Panda(SHOP_FLOOR_IP)
    p = panda.get_position()
    o = panda.get_orientation()
    pose = np.append(p,o)
    print(pose)