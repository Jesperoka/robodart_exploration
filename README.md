# Deep Reinforcement Learning on the Franka Emika Robot

I decided to go ahead with MuJoCo, since GPU availability is not guaranteed, but we can definitely access calcfarm servers that have a many CPU cores and lots of RAM, so performance will not be reliant on my personal desktop computer.

I want to implement (as far as possible) a framework which allows for different RL algorithms, RL environment specifications (how much of the control is directly from RL) and which supports sim-to-real transfer, both zero-shot and fine-tuned.

## TODO:

1. Implement SAC bare bones skeleton (not able to fully implement until baseline controller is done and RL environment(s) is/are specified)

    - [x] Replay Buffer
    - [x] Value, Action-Value, Actor and Critic Neural Networks (depending on which SAC implementation I go with first)
    - [x] Main algorithm skeleton

2. Add extension arm model into MuJoCo
    - [ ] Add programmatic release of dart
    - [ ] Use extendo_gripper mesh instead of basic geometric shapes

3. Implement temporary baseline controller, including figuring out how to implement torque control in MuJoCo with the Panda model

    - [ ] Open loop optimal control? trajectory tracking (kinematic)? just
    - [ ] Optionally skip baseline controller and go ahead with RL-only control

4. Implement define a Gymnasium (formerly OpenAI Gym) environment based on the temporary controller

    - [ ] Define rewards, state-action space and start and terminal states.

5. Complete the SAC implementation

6. Train an agent to any capacity

    - [ ] Will probably include some degree of tuning

7. Start work on Sim-To-Real

    - [ ] Rewrite baseline controller to be identical to Christian's real robot controller
    - [ ] Implement the real-world camera reward functionality
    - [ ] Collaborate and add RL policy to real robot controller (this should be done continuously)
    - [ ] Implement a fine-tuning 

8. Evaluate if multiprocessing/vectorization is necessary

    - [ ] It might be more fruitful to implement another algorithm (TD3 or simpler things like DDPG and PPO)
    - [ ] Or it might be more fruitful to tune SAC further
    - [ ] Or it might be more fruitful to implement other variations of SAC

