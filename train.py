import mujoco
from mujoco import viewer

import mediapy
import matplotlib.pyplot as plt

from utils import common 

from gymnasium_envs import mujoco_gym

MODEL_PATH = "mujoco_models/scene.xml"

DURATION = 3.8  # (seconds)
FPS = 60        # (Hz)

if __name__ == "__main__":

    # ctx = mujoco.GLContext(800, 800)
    # ctx.make_current()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    viewer.launch(model, data)

    # frames = []

    # while data.time < DURATION: 

    #     mujoco.mj_step(model, data)

    #     if len(frames) < data.time * FPS:

    #         renderer.update_scene(data)
    #         image = renderer.render()
    #         frames.append(image)

    # common.save_video(frames, filename="vid.mp4", show=True) 
