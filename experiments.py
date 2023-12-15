from pprint import pprint
from datetime import datetime 

import numpy as np
from tqdm import tqdm

from rl_algorithms.hybrid_sac import HybridSAC
from rl_algorithms.sac import SAC
from rl_environment.mujoco_gym import FrankaEmikaDartThrowEnv
from utils import data
from utils.dtypes import NP_DTYPE

# Experiment Configurations
# ---------------------------------------------------------------------------- #
hyperparams = {
    "lr_actor": 0.0003,
    "lr_critic_1": 0.0003,
    "lr_critic_2": 0.0003,
    "weight_decay": 0.001,
    "alpha": 1.0,
    "gamma": 1.0,
    "batch_size": 512,
}

configs = [
    {
        "name": "baseline_r0",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "hybrid_r0",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "munch_laber_large_snorm_r0",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": True,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": True,
            "spectral_norm": True,
        },
        "reward_function": 0,
    },
    {
        "name": "hybrid_munch_laber_large_snorm_r0",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": True,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": True,
            "spectral_norm": True,
        },
        "reward_function": 0,
    },
    {
        "name": "her_r1",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": False,
            "HER": True,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 1,
    },
    {
        "name": "hybrid_her_r1",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": False,
            "HER": True,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 1,
    },
    {
        "name": "laber_r0",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "hybrid_laber_r0",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "munch_r0",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": True,
            "HER": False,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "hybrid_munch_r0",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": True,
            "HER": False,
            "LaBER": False,
            "tune_alpha": True,
            "larger_nn": False,
            "spectral_norm": False,
        },
        "reward_function": 0,
    },
    {
        "name": "large_laber_snorm_r0",
        "hyperparams": hyperparams,
        "hybrid": False,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": True,
            "spectral_norm": True,
        },
        "reward_function": 0,
    },
    {
        "name": "hybrid_large_laber_snorm_r0",
        "hyperparams": hyperparams,
        "hybrid": True,
        "extensions": {
            "munchausen": False,
            "HER": False,
            "LaBER": True,
            "tune_alpha": True,
            "larger_nn": True,
            "spectral_norm": True,
        },
        "reward_function": 0,
    },
]
# ---------------------------------------------------------------------------- #


def run(configs: list[dict], simulator_config: dict):
    num_episodes = 9000 
    num_runs_per_config = 3 

    for config in configs:
        min_mean_final_distance = np.inf
        print("\nRunning With Config:\n"); pprint(config)
        for run in range(num_runs_per_config):
            print("\nRun "+str(run+1)+"/"+str(num_runs_per_config)+":\n")
            args = (num_episodes, run, config, simulator_config)
            training_loop(args, min_mean_final_distance)


def agent_creator(env, config: dict):
    hyperparams = config["hyperparams"]
    extensions = config["extensions"]

    a_min = env.action_space.low
    a_max = env.action_space.high
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.shape[0]

    if config["hybrid"]:
        disc_a_size = 2
        return HybridSAC(a_min[0:-1], a_max[0:-1], s_size, a_size-1, disc_a_size, config["name"], **extensions, **hyperparams)
    else:
        return SAC(a_min, a_max, s_size, a_size, config["name"], **extensions, **hyperparams)


def training_loop(args: tuple[int, int, dict, dict], min_mean_final_distance):
    num_episodes, run, config, simulator_config = args

    env = FrankaEmikaDartThrowEnv(config["hybrid"], config["reward_function"], **simulator_config)
    agent = agent_creator(env, config)

    # filename = config["name"] + "_run" + str(run) + "_" + np.datetime_as_string(np.datetime64("now"))
    filename = f"{config['name']}_run{run}_{datetime.now().strftime('D%y%m%d_T%H%M')}"
    filepath = 'logs/' + filename + str(".h5py")

    chunk_size = 200
    final_distance_data = np.empty(chunk_size, dtype=NP_DTYPE)
    undiscounted_return_data = np.empty(chunk_size, dtype=NP_DTYPE)
    labels = ["final distance", "undiscounted return"]

    mean_final_distance = np.inf
    mean_window = 30

    render = True if simulator_config["render_mode"] == "human" else False

    print("\nTraining...\n")
    for episode in tqdm(range(num_episodes)):
        idx = episode % chunk_size
        state, info = env.reset()
        done = False

        # Episode Rollout
        step = 0
        undiscounted_return = 0
        while not done:
            cont_action, disc_action, full_tanh_action = agent.choose_action(state)
            next_state, reward, done, info = env.step(cont_action, disc_action)
            agent.remember(state, full_tanh_action, reward, next_state, done)

            state = next_state
            step += 1
            if render: env._render()
            undiscounted_return += reward

        if agent.HER: agent.replay_buffer.hindsight_experience_replay(step, env.reward_function, env.decompose_state, k=4) 

        for i in range(step):
            agent.learn(episode + i)

        if episode % chunk_size == 0 and episode != 0:
            data.save(filepath, [final_distance_data, undiscounted_return_data], labels)

        final_distance_data[idx] = info["distance"]
        undiscounted_return_data[idx] = undiscounted_return

        if episode >= chunk_size:
            window = np.arange(idx, idx - mean_window, -1) % chunk_size
            mean_final_distance = final_distance_data[window].mean()

            if mean_final_distance < min_mean_final_distance:
                min_mean_final_distance = mean_final_distance
                agent.save_models()

    print("\nMinimum mean final distance:", min_mean_final_distance)
    env.close()

