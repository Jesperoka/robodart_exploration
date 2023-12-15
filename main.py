import faulthandler

faulthandler.enable()
from os import environ

import numpy as np

from rl_environment import baseline_controllers 
import experiments

MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 10

DURATION = 3.8  # (seconds)

if __name__ == "__main__":
    environ["MUJOCO_GL"] = "glfw"

    controller = baseline_controllers.SelectedJointsPID()
    controller.do_log = False 

    simulator_config = {
        "baseline_controller": controller,
        "mujoco_model_path": MODEL_PATH,
        "frame_skip": 10,
        "camera_id": None,
        "camera_name": None,
        "render_mode": "depth_array",
        "display_traj": False,
        "display_corners": False,
    }

    experiments.run(experiments.configs[-1:], simulator_config)
    
    # TODO: create simulator config
    # TODO: organize reward functions to test

    # TODO: First
    # - make sure plotting data works fine
    # - make a structured test of the learned policy
    # - validate algorithms on simpler environments like moonlander

    # TODO: Second
    # - L2 Regularization
    # - Hybrid vs Non-hybrid action space
    # - LaBER SAC vs non-LaBER
    # - HER with sparse reward
    # - HER with dense reward
    # - Spectral normalization
    # - Larger neural network 
    # - Many different reward functions
    # - Manual Curriculum Learning (i.e. fine-tune with new reward)
