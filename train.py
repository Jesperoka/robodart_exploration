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

    # ctx = mujoco.GLContext(800, 800)
    # ctx.make_current()

    # environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH, FRAME_SKIP)
    # print(environment.data.site("release_point").xpos)
    # input("wait")
    # environment.reset()

    end = False

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    # print(dir(data))
    # exit()

    def key_callback(keycode):
        global end
        if chr(keycode) == " ":
            end = not end
            if end:
                mujoco.mj_resetDataKeyframe(model, data, model.keyframe("throw_start_tau_ctrl").id)
            else:
                model.eq("weld").active = 0
            # print("\nxpos: ", data.site("release_point").xpos)
            # print("\nxmat: ", data.site("release_point").xmat)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    while viewer.is_running():
        if not end:
            viewer.sync()
            mujoco.mj_step(model, data)
        else:
            viewer.sync()
            print("\nxpos: ", data.site("release_point").xpos)
            print("\nxquat: ", from_rotation_matrix(data.site("release_point").xmat.reshape((3,3))))
            # viewer.close()
            # gc.collect()

    # score = 0
    # for i in range(500):
    #     action = environment.action_space.sample()
    #     state, reward, done, info = environment.step(action)
    #     score += reward
    #     environment.render()

    # print("Score: {}".format(score))

    # frames = []

    # while data.time < DURATION:

    #     mujoco.mj_step(model, data)

    #     if len(frames) < data.time * FPS:

    #         renderer.update_scene(data)
    #         image = renderer.render()
    #         frames.append(image)

    # common.save_video(frames, filename="vid.mp4", show=True)
