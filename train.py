import faulthandler
import gc

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

    end = False

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)


    def key_callback(keycode):
        global end
        if chr(keycode) == " ":
            end = not end
            mujoco.mj_resetDataKeyframe(model, data, model.keyframe("throw_start_tau_ctrl").id)
            print("\nxpos: ", data.site("release_point").xpos)
            print("\nxmat: ", data.site("release_point").xmat)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    while viewer.is_running():
        if not end:
            viewer.sync()
            mujoco.mj_step(model, data)
        else:
            viewer.sync()
            print("\nxpos: ", data.site("release_point").xpos)
            print("\nxmat: ", data.site("release_point").xmat)
