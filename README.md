
<h1 align="center"> Meeting 20.09.2023 </h1>

<h2 align="center"> -- Done since last time -- </h2>

I've read about reinforcement learning to learn MPC horizon and MPC recalculation timing, separately and jointly.
Relevant because I need to frame the RL problem with reference to a controller or
in general the arm motion. Also just helps with learning about how RL can be a framework
for optimization of controller parameters.

That pointed me towards reading about some specific RL methods, in particular Soft Actor Critic (SAC). The reason it's interesting is that is has high sample efficiency and can converge to more robust policies. This also ties in to the paper on dart release timing, and how we might want to learn a release timing adjustment policy that maximizes the time spent in the good-release-zone of the throwing trajectory.

<h3 align="left"> Soft Actor Critic </h3>

This approach considers not just the expected return but also the entropy of the policy. The objective is to maximize the expected reward and the entropy of the policy. This leads to more exploratory policies and can prevent premature convergence to suboptimal strategies. 

Furthermore SAC is said to be less sensitive to hyperparameters, and it can be modified to automatically tune the main 'temperature' hyperparameter which is a factor in the objective function that determines the weighting of entropy versus reward, essentially exploration vs exploitation.

We definitely note that there are many other candidate algorithms (PPO, TD3PG), and testing multiple algorithms is one potential goal. 
Either way we need simulation frameworks that we can employ learning in.

<h3 align="left"> Simulation Environments </h3>

<h4 align="left"> MuJoCo </h4>

CPU-based, now interfaces with OpenAI Gym in what looks like a quite user-friendly manner. Did some tutorial notebooks, need to investigate more.
To begin with I think it is probably a good idea to try MuJoco. MuJoCo also has the Franka Emika model translated from the ROS one, but this is probably not unique to MuJoCo, since the model is in a general modeling language.

<h3 align="center"> Panda MuJoCo Model </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/PandaModel.gif?raw=true" width=350>
</p>

<h4 align="left"> Brax </h4>

JAX-based, so supports fully differentiable simulation on GPUs. Probably very good, but while I have some experience with JAX, it can get tricky at times. That being said, while being a physics engine it has some focus on reinforcement learning, and indeed has implementations of SAC, PPO and APG.
Thus, if we end up needing to train on GPUs, this might be the way to go.

<h3 align="left"> RL implementation tricks </h3>

Random Exploration at the Start: For a fixed number of steps at the beginning (set with the start_steps keyword argument), the agent takes actions which are sampled from a uniform random distribution over valid actions. After that, it returns to normal exploration.

Experience Replay: This involves storing recent experiences (state, action, reward, next state) in a buffer and sampling from this buffer to train the model. This breaks the temporal correlations and allows the algorithm to learn from a diverse set of experiences.

Prioritized Experience Replay: Instead of sampling uniformly from the replay buffer, prioritize experiences where the agent has the most to learn (e.g., high temporal difference error).

Target Networks: Especially in Q-learning based methods, using a separate, slowly-updated target network to calculate target Q-values can stabilize training. 

Multi-step Learning: Instead of learning from single step transitions, use multi-step returns for a more informed update.

Polyak Averaging: Instead of copying weights to the target network, a weighted average (controlled by a factor) of the main and target network weights can be used for smoother updates.

Gradient Clipping: To prevent large updates that can destabilize learning, the gradients can be clipped to a maximum value.

Normalization: Normalizing inputs (states) to neural networks can make training more stable. Similarly, reward normalization or advantage normalization can also be beneficial.

Noise Injection: Adding noise to the policy (e.g., Ornstein-Uhlenbeck noise) can help with exploration.

<h3 align="left"> Franka Emika Panda Max Velocity Issue </h3>

We've come to realize that the Panda has some pretty low max linear velocities, and that if linear motion limits are exceeded the motion is stopped. This means we can't just use the angular velocity limits of the individual joints to produce higher linear end-effector velocity. Unless we want to hack and override these limits on a 100 000 NOK robot, we need to think about solving this some other way.

<h3 align="center"> Panda Diagrams and Limits </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/panda_diagram.png?raw=true" width=350>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/panda_diagram_2.png?raw=true" width=350>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/panda_limits.png?raw=true" width=350>
</p>

One way is to change the parameters of the task, i.e. move the dart board closer, or lay it on the ground. This feels kind of lame, so personally I think we should make some kind of gripper extension to get a larger lever arm, and then use the combined maximum linear velocity of the robot with the gripper joint maximum angular velocity to achieve a sufficient launch velocity. To get this going as quickly as possible, here are some simple ideas:

<h3 align="center"> Two Simple Gripper Extension Ideas </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/gripper_extension.jpg?raw=true" width=500>
</p>

The first idea outlined is just to have the gripper hold some sort of lever arm in which we slot the dart (if we can keep the fin on, great) and the have the motion stop at the release point.
The second idea is to keep more of a gripper style end-effector that releases the dart like we were planning to do with the standard gripper anyway. If we opt for this we might have the option to essentially 'gear-shift' up the speed of the gripper, solving an earlier issue of the gripper hitting the dart-fins, which would mean we could have the arm motion stop at the release points and not have to continue with a follow-through as has been our current thought.

In relation to getting something going quickly so that we can start implementing *everything else*, we can create the first option with some cardboard and tape.
For the gripper extension approach we could try one of these bad boys:

<h3 align="center"> Illustrative Images </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/claw.png?raw=true" width=350>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/cardboard.jpg?raw=true" width=350>
</p>

<h3 align="left"> Automatic Recording of Throw Results </h3>

For the automatic recording of the dart throw results, it would be nice to use something already implemented, or atleast only change and already existing implementation a bit to fit out goals. Of course there needs to be some interfacing code that sends the result to the RL framework if we are learning on the real robot. In simulation it doesn't matter, but we need to make sure the dart board dimensions are the same so that rewards are the same scale (or adjust for it).

> OpenCV, a Raspberry Pi 3 Model B and two webcams: [github.com/hanneshoettinger/opencv-steel-darts](https://github.com/hanneshoettinger/opencv-steel-darts) <br>
> OpenCV and Single Calibrated (april tags) Camera: [github.com/LarsG21/Darts_Project](https://github.com/LarsG21/Darts_Project) <br>
> Neural network approach: [github.com/wmcnally/deep-darts](https://github.com/wmcnally/deep-darts)

<h2 align="center"> -- Going forward -- </h2>


Get a basic gripper extension going, and think about if we should/can develop a better solution.

I want to familiarize myself with OpenAI Gym and MuJoCo, and set up a simulated robot throw. If I can hardcode this, then I can try to use a reinforcement learning algorithm like SAC to learn the gripper release point. From there I can do more research on how to effectively implement a learning framework.

Another thing on the agenda is to work towards the real robot, and in that regard we need a functioning camera system that can record the result of a throw. This is shouldn't be too hard in terms of the code, but since we are using he Linux RT kernel patch, we don't have working nVidia drivers on the lab computer, so it might prove annoying to get any kind of camera system working. 

Then there is the literature search, which I haven't had enough time to do properly yet, but so far I've found some papers that have promising titles and abstracts.

Sim-to-Real Transfer of Robotic Control with Dynamics Randomization: https://arxiv.org/pdf/1710.06537.pdf

Learning Motor Primitives for Robotics: https://www.researchgate.net/publication/224557356_Learning_Motor_Primitives_for_Robotics

Reinforcement Learning to adjust Robot Movements to New Situations: https://www.researchgate.net/publication/50402188_Reinforcement_Learning_to_adjust_Robot_Movements_to_New_Situations

Reinforcement Learning to Adjust Parametrized Motor Primitives to New Situations: https://www.researchgate.net/publication/245030976_Reinforcement_Learning_to_Adjust_Parametrized_Motor_Primitives_to_New_Situations

Learning Perceptual Coupling for Motor Primitives: Learning Perceptual Coupling for Motor Primitives

Learning Table Tennis with a Mixture of Motor Primitives: https://www.researchgate.net/publication/50402372_Learning_Table_Tennis_with_a_Mixture_of_Motor_Primitives

High-speed Throwing Motion Based on Kinetic Chain Approach: https://ishikawa-vision.org/members/senoo/paper/senoo_iros08.pdf 

Learning to Throw with a Handful of Samples Using Decision Transformers: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9984828

A Solution to Adaptive Mobile Manipulator Throwing: https://arxiv.org/pdf/2207.10629.pdf 

Dynamic Compensation in Throwing Motion with High-Speed Robot Hand-Arm: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9560866

A stochastic dynamic motion planning algorithm for object-throwing: https://ieeexplore.ieee.org/document/7139530

Learning of a basketball free throw with a flexible link robot: https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings/IDETC-CIE2021/85468/V009T09A033/1128185

Robust trajectory design for object throwing based on sensitivity for model uncertainties: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7139623

Off-Policy Deep Reinforcement Learning without Exploration: https://arxiv.org/pdf/1812.02900.pdf (useful if we end up with high-dimensional continuous state-action space)
