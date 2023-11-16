# Deep Reinforcement Learning on the Franka Emika Robot

I decided to go ahead with MuJoCo, since GPU availability is not guaranteed, but we can definitely access calcfarm servers that have a many CPU cores and lots of RAM, so performance will not be reliant on my personal desktop computer.

I want to implement (as far as possible) a framework which allows for different RL algorithms, RL environment specifications (how much of the control is directly from RL) and which supports sim-to-real transfer, both zero-shot and fine-tuned.

## TODO:

1. Implement SAC bare bones skeleton (not able to fully implement until baseline controller is done and RL environment(s) is/are specified)

    - [x] Replay Buffer.
    - [x] Value, Action-Value, Actor and Critic Neural Networks (depending on which SAC implementation I go with first).
    - [x] Main algorithm skeleton.

2. Add extension arm model into MuJoCo
    - [X] Add programmatic release of dart.
    - [ ] Use extendo_gripper mesh instead of basic geometric shapes.

3. Implement temporary baseline controller, including figuring out how to implement torque control in MuJoCo with the Panda model

    - [X] Open loop optimal control? trajectory tracking (kinematic)? 
    - [X] Optionally skip baseline controller and go ahead with RL-only control.

4. Implement define a Gymnasium (formerly OpenAI Gym) environment based on the temporary controller

    - [X] Define rewards, state-action space and start and terminal states.
    - [ ] Reformulate as in https://arxiv.org/pdf/2210.00609.pdf
    - [ ] After reformulating, can add dart pos as state (it practically observable while robot is holding it).
    - [ ] Add goal pos randomization.
    - [ ] Add initial pose randomization.

5. Complete the SAC implementation

    - [X] First impl. following youtube tutorial.
    - [X] Refactor and update to newest SAC version.

6. Train an agent to any capacity

    - [X] Will probably include some degree of tuning.
    - [X] Learn some kind of bad throw.
    - [ ] Learn to hit target.
    - [ ] Learn to hit close to goal pos sometimes.
    - [ ] Learn to hit close to goal pos most of the time.
    - [ ] Learn to throw dart accurately.

7. Start work on Sim-To-Real

    - [ ] Rewrite baseline controller to be identical to Christian's real robot controller.
    - [ ] Implement the real-world camera reward functionality.
    - [ ] Collaborate and add RL policy to real robot controller (this should be done continuously).
    - [ ] Implement a fine-tuning process.

8. Evaluate if multiprocessing/vectorization is necessary

9. Implement/add implementations of other algorithms
