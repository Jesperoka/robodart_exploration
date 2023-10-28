import faulthandler

faulthandler.enable()

import matplotlib.pyplot as plt
from mujoco import viewer
from quaternion import from_rotation_matrix

import gymnasium_envs
from gymnasium_envs import mujoco_gym
from utils import common

from rl_algorithms import sac

MODEL_PATH = "./mujoco_models/scene.xml"
MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 1

DURATION = 3.8  # (seconds)
FPS = 60  # (Hz)

import numpy as np


def zero_controller(dart_pos, goal_pos):
    return np.zeros(7)


if __name__ == "__main__":
    environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH, FRAME_SKIP, baseline_controller=zero_controller)
    environment.reset()
    environment.render()

    # score = 0
    # for i in range(500):
    #     if i % 50 == 0: environment.reset()
    #     action = environment.action_space.sample()
    #     state, reward, done, info = environment.step(action)
    #     score += reward
    #     environment.render()

    # print("Score: {}".format(score))

    sac.basic_training_loop(environment, 100)
