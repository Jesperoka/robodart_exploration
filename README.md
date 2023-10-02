<h1 align="left"> Table of contents </h1>

---

1. [Meeting 21.09.2023](#m210923)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1 [Done Since Last Time](#m210923-dslt)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.1 [First Control](#m210923-fpc)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.2 [Soft Actor Critic](#m210923-sac)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.3 [Simulation Evironments](#m210923-sim)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.4 [RL Implementation Tricks](#m210923-rltricks)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.5 [Franka Emika Panda Max Velocity Issue](#m210923-velissue)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.6 [Automatic Recording of Throw Results](#m210923-cam)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2  [Going Forward](#m210923-gf)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.1 [Literature Search](#m210923-ls)<br><br>
2. [Meeting 26.09.2023 (Extra Meeting)](#m260923)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 [Intro](#m210923-intro)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 [Calculations](#m210923-calculations)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3 [In Practice](#m210923-in-practice)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.4 [Initial Conclusions](#m210923-initial-conclusions)<br>
---

<a name="m210923"></a><h1 align="center"> Meeting 21.09.2023 </h1>
<a name="m210923-dslt"></a><h2 align="center"> -- Done since last time -- </h2>

Got the robot working with libfranka and the panda-python wrapper library, and set up and showed the 3D printing group how to do the same.

<a name="m210923-fpc"></a><h3 align="center"> Our First Programmatic Control of the Panda </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/first_movement.gif?raw=true" width=350>
</p>

I've read about reinforcement learning to learn MPC horizon and MPC recalculation timing, separately and jointly.
Relevant because I need to frame the RL problem with reference to a controller or
in general the arm motion. Also just helps with learning about how RL can be a framework
for optimization of controller parameters.

That pointed me towards reading about some specific RL methods, in particular Soft Actor Critic (SAC). The reason it's interesting is that is has high sample efficiency and can converge to more robust policies. This also ties in to the paper on dart release timing, and how we might want to learn a release timing adjustment policy that maximizes the time spent in the good-release-zone of the throwing trajectory.

<a name="m210923-sac"></a><h3 align="left"> Soft Actor Critic </h3>

This approach considers not just the expected return but also the entropy of the policy. The objective is to maximize the expected reward and the entropy of the policy. This leads to more exploratory policies and can prevent premature convergence to suboptimal strategies. 

Furthermore SAC is said to be less sensitive to hyperparameters, and it can be modified to automatically tune the main 'temperature' hyperparameter which is a factor in the objective function that determines the weighting of entropy versus reward, essentially exploration vs exploitation.

We definitely note that there are many other candidate algorithms (PPO, TD3PG), and testing multiple algorithms is one potential goal. 
Either way we need simulation frameworks that we can employ learning in.

<a name="m210923-sim"></a><h3 align="left"> Simulation Environments </h3>

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

<a name="m210923-rltricks"></a><h3 align="left"> RL implementation tricks </h3>

Random Exploration at the Start: For a fixed number of steps at the beginning (set with the start_steps keyword argument), the agent takes actions which are sampled from a uniform random distribution over valid actions. After that, it returns to normal exploration.

Experience Replay: This involves storing recent experiences (state, action, reward, next state) in a buffer and sampling from this buffer to train the model. This breaks the temporal correlations and allows the algorithm to learn from a diverse set of experiences.

Prioritized Experience Replay: Instead of sampling uniformly from the replay buffer, prioritize experiences where the agent has the most to learn (e.g., high temporal difference error).

Target Networks: Especially in Q-learning based methods, using a separate, slowly-updated target network to calculate target Q-values can stabilize training. 

Multi-step Learning: Instead of learning from single step transitions, use multi-step returns for a more informed update.

Polyak Averaging: Instead of copying weights to the target network, a weighted average (controlled by a factor) of the main and target network weights can be used for smoother updates.

Gradient Clipping: To prevent large updates that can destabilize learning, the gradients can be clipped to a maximum value.

Normalization: Normalizing inputs (states) to neural networks can make training more stable. Similarly, reward normalization or advantage normalization can also be beneficial.

Noise Injection: Adding noise to the policy (e.g., Ornstein-Uhlenbeck noise) can help with exploration.

<a name="m210923-velissue"></a><h3 align="left"> Franka Emika Panda Max Velocity Issue </h3>

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

<a name="m210923-cam"></a><h3 align="left"> Automatic Recording of Throw Results </h3>

For the automatic recording of the dart throw results, it would be nice to use something already implemented, or atleast only change and already existing implementation a bit to fit out goals. Of course there needs to be some interfacing code that sends the result to the RL framework if we are learning on the real robot. In simulation it doesn't matter, but we need to make sure the dart board dimensions are the same so that rewards are the same scale (or adjust for it).

> OpenCV, a Raspberry Pi 3 Model B and two webcams: [github.com/hanneshoettinger/opencv-steel-darts](https://github.com/hanneshoettinger/opencv-steel-darts) <br>
> OpenCV and Single Calibrated (april tags) Camera: [github.com/LarsG21/Darts_Project](https://github.com/LarsG21/Darts_Project) <br>
> Neural network approach: [github.com/wmcnally/deep-darts](https://github.com/wmcnally/deep-darts)

<a name="m210923-gf"></a><h2 align="center"> -- Going forward -- </h2>


Get a basic gripper extension going, and think about if we should/can develop a better solution.

I want to familiarize myself with OpenAI Gym and MuJoCo, and set up a simulated robot throw. If I can hardcode this, then I can try to use a reinforcement learning algorithm like SAC to learn the gripper release point. From there I can do more research on how to effectively implement a learning framework.

Another thing on the agenda is to work towards the real robot, and in that regard we need a functioning camera system that can record the result of a throw. This is shouldn't be too hard in terms of the code, but since we are using he Linux RT kernel patch, we don't have working nVidia drivers on the lab computer, so it might prove annoying to get any kind of camera system working. 

<a name="m210923-ls"></a><h3 align="left"> Literature Search </h3>

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

---

<a name="m260923"></a><h1 align="center"> Meeting 26.09.2023 </h1>

The focus of this meeting is to address the velocity issues and consequently how to solve the now needed gripper extension design or overriding the speed limits of the Franka Emika robot.

<a name="m260923-intro"></a><h2 align="left"> Intro </h2>

In order to get some basic specifications on the gripper extension we have done some basic calculations on lever arm length and the corresponding limits on the mass or rotational inertia of the extension. Additionally we did some basic tests on the robot in order to see how those calculations play out in practice, to get better idea of what kind of weight the extension can actually have, and to get some qualitative understanding of what the limiting factors are going to be for the control problem.

We've determined that in terms of the possible torque output of the robot alone, we will not be facing issues related to the mass of the extension. Issues from mass come primarily from oscillations and dynamics of the attachment, as too much external torque make the robot reflex (stop) because it reaches limits meant to signify a collison. Indeed the external torque and force limits are based on estimates, and can be wrong. These limits can be adjusted within the libfranka api, which mitigates the issue and is necessary. It should be noted that doing this means the robot is less safe to be around when operating, since it essentially means we are expecting to have collisions.

<a name="m260923-calculations"></a><h2 align="left"> Calculations </h2>

To be able to reach the dart board we need a lever arm of at least $1.0\text{ m}$, and preferably slightly longer.

For a uniform extension rod of $1.0\text{ m}$ the max mass of the extension is ~ $1.813\text{ kg}$ accounting for gravity but without accounting for friction. The corresponding inertia one third of that, i.e. ~ $0.604\text{ kg m}^{2}$. 

For a uniform extension rod of $1.5\text{ m}$ the max mass of the extension is ~ $1.11\text{ kg}$ accounting for gravity but without accounting for friction. The corresponding inertia one third of that, i.e. ~ $0.37\text{ kg m}^{2}$.

These calculations assume no friction, which makes them less conservative, but require the arm to be able to hold the lever arm horizontally against gravity while applying the maximum possible torque to achieve the minimum acceleration needed to reach max joint angular velocity. This outweights the lack of friction, making them in fact quite conservative based on our physical tests.

<a name="m260923-in-practice"></a><h2 align="left"> In Practice </h2>

In practice friction plays an important role, and makes it so higher torque is required to accelerate the extension mass. On the other hand, the assumption of having to be able to counteract gravity is a very conservative one, since we are able to start the lever arm in a position where gravity helps us accelerate, thus allowing us to break the joint velocity violation while using substantially lower torques than the max torque.

We were not able to trigger the total power limit violation while testing constant torque of 3 joints simultaneously with a lever arm attached. The tests had lever arm rotational inertias of very approximately

|1|2|3|4|5|
|----|--|--|--|--|
| $0.125\text{ kg m}^{2}$ | $0.4\text{ kg m}^{2}$ | $0.5\text{ kg m}^{2}$ |  $0.69\text{ kg m}^{2}$ | $1.25\text{ kg m}^{2}$
 
These test configurations can be seen in the figure below with inertias in order of lowest-to-highest, left-to-right.

<h3 align="center"> Test Configurations </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/test_configurations.PNG?raw=true" width=400>
</p>
<p align="center"> Lever arms with different rotational inertias </p>

The approximate figures comes from treating the bottles as point masses and including the carboard rod as a uniform rod as well.

For configurations 1, 2 and 3, we were able to reach maximum joint velocity from a standstill starting position, confirmed by triggering the `joint_velocity_violation` error, and while using only constant torque on joint 7. Tests can be seen in the figure below.

<h3 align="center"> Reaching Maximum Joint Velocity (From Standstill) </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_1_joint_velocity.gif?raw=true" width=200>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_2_joint_velocity.gif?raw=true" width=200>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_3_joint_velocity.gif?raw=true" width=200>
</p>
<p align="center"> From a standstill, the maximum inertia that allows<br> us to reach maximum joint velocity of the end effector <br>is somewhere in the range [0.5, 0.69] kg m^2 </p>

However, using gravity to assist us in the acceleration allows for higher inertias as illustrated in the figure below.

<h3 align="center"> Reaching Maximum Joint Velocity (Gravity Assist) </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_4_joint_velocity_gravity_assist.gif?raw=true" width=200>
</p>
<p align="center"> Gravity assist allows for higher inertias<br> while still achieving max joint velocity </p>

Therefore, the main issue that comes with having a higher mass gripper extension actually comes from the dynamics and potential oscillations it can introduce into the system. If external torques and forces are ever estimated to be too high, the robot has detected a collison and motion is aborted. The figure below shows this happening for configurations 4 and 5.

<h3 align="center"> Reflex Errors </h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_4_reflex.gif?raw=true" width=200>
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_5_reflex.gif?raw=true" width=200>
</p>
<p align="center"> Higher mass means higher chance of <br>exceeding external force and torque limits </p>

Note that, as mentioned, its possible to adjust the external torque and force limits, but we have not found any documentation on the limits of this or mapped out how well this works in practice.

Lastly, we were not able to trigger the `power_limit_violation` error using the simple torque controller we used during the tests, which is at least promising for the control task.


<h3 align="center"> Multiple Joints Simultaneously</h3>
<p align="center">
<img src="https://github.com/Jesperoka/robodart_exploration/blob/jesper_meeting_notes/imgs/config_2_three_joints_no_error.gif?raw=true" width=200>
</p>
<p align="center"> Constant torque applied to 3 joints simultaneously while <br> the other 4 joints are kept at a constant position using a simple <br> PD controller did not exceed the maximum power limits </p>

<a name="m260923-initial-conclusion"></a><h2 align="left"> Initial Conclusion </h2>

We probably want an arm length of about $1.25 \text{ m}$, and while the weight can probably be quite high in practice, **the lighter the better**, and we feel like it should be possible to make a grip-and-release mechanism that has a rotational intertia of ~ $0.4\text{ kg m}^{2}$ or less, i.e. have a mass $m \in [1.11, 1.6] \text{ kg}$ depending on distribution. 

There is also a point to be made that the lever arm needs to be rigid, and if possible the forces on the robot at the points of attachment should be spread out as evenly as possible. This is to potentially have more stable external torque and force estimates.

---

<a name="m111023"></a><h1 align="center"> Meeting 11.10.2023 </h1>

<a name="temp"></a><h3 align="center"> Temp </h3>

https://impact-aware-robotics-database.tue.nl/

https://www.youtube.com/watch?v=07tucCGxVj8 ModBus

https://download.franka.de/documents/221010_Operating%20Instructions%20ECBPi%20CobotPump_11.18_EN.pdf Vacuum gripper (useful for specs when designing own)
https://pimmedia.schmalz.com/MAM_Library/Dokumente/Datenblatt_Produktfamilie/0_/054/05463/Datasheet_Vacuum%20Generators%20ECBPi_en-EN.pdf

https://amslaurea.unibo.it/20126/1/Tesi%20Magistrale%20_%20Marco%20Speziali.pdf
https://github.com/felixduvallet/allegro-hand-ros

ModBus TCP/IP is something, I don't know if that's what the gripper uses, I think it probably is, but it is non-realtime.

https://www.youtube.com/watch?v=LU2rSG2Orwg Some open source 3D-printable grippers.
https://www.youtube.com/watch?v=yI8E50Orkng 10 gripper mechanisms
https://www.youtube.com/watch?v=IalkWUN6wvE another gripper
https://www.youtube.com/watch?v=p9QoT1iwVY4 another gripper
https://www.youtube.com/watch?v=oVVvKC19Y7I another gripper

https://blog.robotiq.com/how-to-choose-a-parallel-gripper-for-my-robot

https://jrl-umi3218.github.io/mc_rtc/doxygen.html
https://github.com/jrl-umi3218/mc_franka
https://github.com/jrl-umi3218/mc_rtc
https://github.com/jrl-umi3218/mc_panda
https://github.com/rohanpsingh/mc_mujoco
