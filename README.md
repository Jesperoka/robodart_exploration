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

## Waypoints

[1.6107915528870635, 1.5525780056084473, -0.11313920660380519, -0.7339375209208336, -1.4657986472182802, 1.6814151651859284, -2.8062478421779558]

[1.6315304899178595, 0.9646478766487201, -0.08685739709786426, -1.4703425200779734, -1.47846811832322, 1.6141401752428557, -2.3161907760775944]

[1.6300490850852782, 0.32923692989918013, -0.1102309887495663, -2.221418001609936, -1.4777691815429264, 1.6151759734409383, -1.9463963213215272]

[1.6060873788074408, -0.4067821982450511, -0.16251752290181945, -2.533035615285238, -1.5513210105786672, 1.6089720550378162, -1.6761394923928497]

[1.5794342359304425, -0.9236706516615699, -0.09293284878319552, -2.5182514419890287, -1.6603285350432753, 1.6273696882724762, -1.7784775661081074]

[1.5763291470607121, -1.4995582987969382, -0.07416483234313496, -2.295306028566862, -1.6199807633682126, 1.614242450316747, -1.4609177215734646]

[1.5951008298962612, -1.3519723510317008, -0.07068710005444366, -1.2809748270637609, -1.57656607315754, 1.649874658505803, -1.1752330182643813]

[1.5451935434733566, -0.9572197887019106, -0.024129691323335725, -0.44686389571742, -1.587876229777748, 1.5885597769162976, 0.6952317190152573]

[1.4947413081690775, -1.2398676652741014, -0.057350458682106246, -0.08582277291010126, -1.515310267421934, 1.5979226810932186, 1.2949260392136528]

[1.4752757614222929, -1.6534933501461095, -0.06813756467934505, -0.07739810377888262, -1.5177574449910056, 1.5965783865451812, 1.297909321126349]
