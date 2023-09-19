import panda_py

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk

if __name__ == "__main__":

    panda = panda_py.Panda(SHOP_FLOOR_IP)
    pose = panda.get_state()
    print(pose.q)


