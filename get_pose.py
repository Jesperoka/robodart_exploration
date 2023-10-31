from panda_py import Panda
if __name__ == "__main__":
#
    panda = Panda(SHOP_FLOOR_IP)
    pose = panda.get_state()
    print(pose.q)
