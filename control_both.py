from threading import Thread
import control_gripper
import control_robot


robot = Thread(target=control_robot.main, daemon=True)
gripper = Thread(target=control_gripper.main, daemon=True)

gripper.start()
robot.start()
