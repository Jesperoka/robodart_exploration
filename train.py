import faulthandler

faulthandler.enable()
from os import environ

import numpy as np

from rl_algorithms import sac
from rl_environment import baseline_controllers, mujoco_gym, reward_functions

MODEL_PATH = "./mujoco_models/scene.xml"
MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 10 

DURATION = 3.8  # (seconds)
FPS = 60  # (Hz)


def zero_controller(dart_pos, goal_pos):
    return np.zeros(7, dtype=np.float32)


if __name__ == "__main__":
    environ["MUJOCO_GL"] = "glfw"

    lookahead_controller = baseline_controllers.LookaheadController()
    lookahead_controller.do_log = False 

    environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH,
                                                     FRAME_SKIP,
                                                     reward_function=reward_functions.capped_inverse_distance,
                                                     baseline_controller=lookahead_controller,
                                                     camera_name="dart_cam",
                                                     render_mode="human")

    sac.basic_training_loop(environment, 1000000)
    # lookahead_controller.plot_logged()
