from threading import Thread
import control_gripper
import control_robot
import test1
import test2


# robot = Thread(target=control_robot.main, daemon=True)
# gripper = Thread(target=control_gripper.main, daemon=True)

# gripper.start()
# robot.start()



robot = Thread(target=test1.main, daemon=True)
gripper = Thread(target=test2.main, daemon=True)


gripper.start()
robot.start()
