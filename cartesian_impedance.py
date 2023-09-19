import sys
import threading
import time

import numpy as np

import panda_py
from panda_py import controllers, libfranka

SHOP_FLOOR_IP = "10.0.0.2"

if __name__ == '__main__':

  panda = panda_py.Panda(SHOP_FLOOR_IP)
  gripper = libfranka.Gripper(SHOP_FLOOR_IP)
  gripper.homing()
  panda.move_to_start()

  def move(panda, runtime):
    ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
    x0 = panda.get_position()
    q0 = panda.get_orientation()
    panda.start_controller(ctrl)

    with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
      while ctx.ok():
        x_d = x0.copy()
        x_d[1] += 0.2 * np.sin(ctrl.get_time())
        ctrl.set_control(x_d, q0)

  def grasp(gripper, runtime):
    start = time.time()
    while time.time() - start < runtime:
      gripper.grasp(0, 0.2, 20)
      gripper.move(0.08, 0.2)

  runtime = np.pi * 4.0

  threading.Thread(target=move, args=(panda, runtime)).start()
  threading.Thread(target=grasp, args=(gripper, runtime)).start()

