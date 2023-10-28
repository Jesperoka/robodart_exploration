import faulthandler
import gc

from quaternion import from_rotation_matrix 

faulthandler.enable()

import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer

import gymnasium_envs
from gymnasium_envs import mujoco_gym
from utils import common

MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 1

DURATION = 3.8  # (seconds)
FPS = 60  # (Hz)

end = False

if __name__ == "__main__":
    environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH, FRAME_SKIP)
    environment.reset()

    score = 0
    for i in range(500):
        action = environment.action_space.sample()
        print(action)
        state, reward, done, info = environment.step(action)
        score += reward
        environment.render()

    print("Score: {}".format(score))
