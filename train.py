import faulthandler

faulthandler.enable()

from os import environ

import matplotlib.pyplot as plt
import numpy as np
from mujoco import viewer

from gymnasium_envs import mujoco_gym
from rl_algorithms import sac

MODEL_PATH = "./mujoco_models/scene.xml"
MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 1

DURATION = 3.8  # (seconds)
FPS = 60  # (Hz)


def zero_controller(dart_pos, goal_pos):
    return np.zeros(7, dtype=np.float32)


if __name__ == "__main__":
    environ["MUJOCO_GL"] = "glfw"

    environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH,
                                                     FRAME_SKIP,
                                                     baseline_controller=zero_controller,
                                                     camera_name="dart_cam")

    sac.basic_training_loop(environment, 100)
