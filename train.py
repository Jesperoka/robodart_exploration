import faulthandler

faulthandler.enable()
from os import environ

import numpy as np

from rl_algorithms import sac
from rl_environment import baseline_controllers, mujoco_gym 
from utils import data
from utils.dtypes import NP_DTYPE

MODEL_PATH = "./mujoco_models/scene.xml"
FRAME_SKIP = 10

DURATION = 3.8  # (seconds)

def basic_training_loop(env, num_episodes):

    agent = sac.SAC(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0]-1,
                  discrete_action_size=2,
                  max_action=env.action_space.high[0:-1],
                  min_action=env.action_space.low[0:-1],
                  update_actor_every=1,
                  update_targets_every=1,
                  num_gradient_steps_per_episode=1,
                  )

    filename = 'training_data_' + np.datetime_as_string(np.datetime64("now"))
    filepath = 'logs/' + filename + str(".h5py")

    chunk_size = 200
    final_distance_data = np.empty(chunk_size, dtype=NP_DTYPE) 
    undiscounted_return_data = np.empty(chunk_size, dtype=NP_DTYPE) 
    labels = ["final distance", "undiscounted return"]

    min_mean_final_distance = np.inf 
    mean_final_distance = np.inf 
    mean_undiscounted_return = -np.inf

    load_checkpoint = False
    load_first = False
    test_every = 50 

    if load_first:
        agent.load_models()

    # num_pre_explorations = 4*agent.batch_size 
    # print("Gathering uniform state-action transition samples...")
    # for _ in tqdm(range(num_pre_explorations)):
    #     env.pre_explore(agent.replay_buffer)
    #     env.render()

    # print("Learning from uniform state-action transition samples...")
    # for i in tqdm(range(num_pre_explorations//10)):
    #     agent.learn(i)

    print("\nTraining...\n")
    for episode in range(num_episodes):
        idx = episode % chunk_size
        state, info = env.reset()
        done = False
        test = (episode % test_every) == 0

        # Episode Rollout
        step = 0
        undiscounted_return = 0
        while not done: 
            if test:
                cont_action, disc_action = agent.choose_deterministic_action(state)
                next_state, reward, done, info = env.step(cont_action, disc_action)
            else:
                cont_action, disc_action, full_tanh_action = agent.choose_action(state)
                next_state, reward, done, info = env.step(cont_action, disc_action)
                agent.remember(state, full_tanh_action, reward, next_state, done)

            state = next_state
            step += 1
            env._render()
            undiscounted_return += reward
            
        if not load_checkpoint: # one-to-one environment and gradient steps
            for i in range(step//10):
                agent.learn(episode+i)

        # Hindsight Experience Replay 
        # agent.replay_buffer.hindsight_experience_replay(step, k=4)

        if episode % chunk_size == 0 and episode != 0:
            data.save(filepath, [final_distance_data, undiscounted_return_data], labels)

        final_distance_data[idx] = info["distance"]
        undiscounted_return_data[idx] = undiscounted_return 
        
        if episode >= chunk_size:
            window = np.arange(idx, idx - 100, - 1) % chunk_size
            mean_final_distance = final_distance_data[window].mean()
            mean_undiscounted_return = undiscounted_return_data[window].mean()

            if mean_final_distance < min_mean_final_distance:
                min_mean_final_distance = mean_final_distance

                if not load_checkpoint: 
                    agent.save_models()

        if not test:
            print('episode', episode, 'distance %.8f' % info["distance"], 'mean distance %.8f' % mean_final_distance, 'return %.8f' % undiscounted_return, 'mean return %.8f' % mean_undiscounted_return)
        else:
            print('\n\nDETERMINISTIC:\nepisode', episode, 'distance %.8f' % info["distance"], 'return %.8f' % undiscounted_return,"\n\n")



if __name__ == "__main__":
    environ["MUJOCO_GL"] = "glfw"

    controller = baseline_controllers.SelectedJointsPID()
    controller.do_log = False 

    environment = mujoco_gym.FrankaEmikaDartThrowEnv(MODEL_PATH,
                                                     FRAME_SKIP,
                                                     baseline_controller=controller,
                                                     camera_name="dart_cam",
                                                     render_mode="human")

    basic_training_loop(environment, 1000000)
    # lookahead_controller.plot_logged()


    # TODO: First
    # - make sure plotting data works fine
    # - make configuration dict to allow validation
    # - multiple runs per configuration
    # - give each config a name, and store data with that name for later plotting
    # - make a structured test of the learned policy
    # - validate algorithms on simpler environments like moonlander

    # TODO: Second
    # - L2 Regularization
    # - Hybrid vs Non-hybrid action space
    # - LaBER SAC vs non-LaBER
    # - Alternating step and optimization vs complete trajectories
    # - HER with sparse reward
    # - HER with dense reward
    # - Manual Curriculum Learning (i.e. fine-tune with new reward)
    # - Spectral normalization
    # - Deeper neural networks
    # - Wider neural networks
    # - Many different reward functions
