
<h1 align="center"> Meeting 20.09.2023 </h1>

<h2 align="left"> Done since last time </h2>

Read about reinforcement learning to learn MPC horizon and MPC recalculation timing, separately and jointly.
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

<h2 align="left"> Going forward </h2>

I want to familiarize myself with OpenAI Gym and MuJoCo, and set up a simulated robot throw. If I can hardcode this, then I can try to use a reinforcement learning algorithm like SAC to learn the gripper release point. From there I can do more research on how to effectively implement a learning framework.

Another thing on the agenda is to work towards the real robot, and in that regard we need a functioning camera system that can record the result of a throw. This is shouldn't be too hard in terms of the code, but since we are using he Linux RT kernel patch, we don't have working nVidia drivers on the lab computer, so it might prove annoying to get any kind of camera system working. 

Then there is the literature search, which I haven't had enough time to do properly yet, but so far I've found some papers that have promising titles and abstracts.

High-speed Throwing Motion Based on Kinetic Chain Approach: https://ishikawa-vision.org/members/senoo/paper/senoo_iros08.pdf 

Learning to Throw with a Handful of Samples Using Decision Transformers: https://www.scopus.com/record/display.uri?origin=citedby&eid=2-s2.0-85144752233&noHighlight=false&sort=plf-f&src=s&sid=177bde1f685c94a4941acef603ebdf68&sot=b&sdt=b&sl=45&s=TITLE-ABS-KEY%28Robot+Arm+Manipulator+Throwing%29&relpos=1 

A Solution to Adaptive Mobile Manipulator Throwing: https://arxiv.org/pdf/2207.10629.pdf 

A stochastic dynamic motion planning algorithm for object-throwing: https://www.scopus.com/record/display.uri?eid=2-s2.0-84938264955&origin=resultslist&sort=r-f&src=s&mltEid=2-s2.0-85146317944&mltType=ref&mltAll=t&imp=t&sid=dcbfc07bca1dc208a81672f85efcf864&sot=mlt&sdt=mlt&sl=579&s=REFEID%28%28%222-s2.0-0027835913%22%29+OR+%28%222-s2.0-85124369853%22%29+OR+%28%222-s2.0-85068442743%22%29+OR+%28%222-s2.0-85089524155%22%29+OR+%28%222-s2.0-84864047294%22%29+OR+%28%222-s2.0-0032759587%22%29+OR+%28%222-s2.0-69549117333%22%29+OR+%28%222-s2.0-70350389386%22%29+OR+%28%222-s2.0-84872338783%22%29+OR+%28%222-s2.0-84999015488%22%29+OR+%28%222-s2.0-84893782296%22%29+OR+%28%222-s2.0-85049685503%22%29+OR+%28%222-s2.0-78651496635%22%29+OR+%28%222-s2.0-67650302094%22%29+OR+%28%222-s2.0-84938264955%22%29+OR+%28%222-s2.0-84864421794%22%29+OR+%28%222-s2.0-0030212126%22%29+OR+%28%222-s2.0-0033726628%22%29+OR+%28%222-s2.0-78650803456%22%29+OR+%28%222-s2.0-85053318433%22%29+OR+%28%222-s2.0-85111222398%22%29%29+AND+NOT+EID+%282-s2.0-85146317944%29&relpos=5&citeCnt=12&searchTerm=

Learning of a basketball free throw with a flexible link robot: https://www.scopus.com/record/display.uri?eid=2-s2.0-85120442147&origin=resultslist&sort=r-f&src=s&mltEid=2-s2.0-85146317944&mltType=ref&mltAll=t&imp=t&sid=dcbfc07bca1dc208a81672f85efcf864&sot=mlt&sdt=mlt&sl=579&s=REFEID%28%28%222-s2.0-0027835913%22%29+OR+%28%222-s2.0-85124369853%22%29+OR+%28%222-s2.0-85068442743%22%29+OR+%28%222-s2.0-85089524155%22%29+OR+%28%222-s2.0-84864047294%22%29+OR+%28%222-s2.0-0032759587%22%29+OR+%28%222-s2.0-69549117333%22%29+OR+%28%222-s2.0-70350389386%22%29+OR+%28%222-s2.0-84872338783%22%29+OR+%28%222-s2.0-84999015488%22%29+OR+%28%222-s2.0-84893782296%22%29+OR+%28%222-s2.0-85049685503%22%29+OR+%28%222-s2.0-78651496635%22%29+OR+%28%222-s2.0-67650302094%22%29+OR+%28%222-s2.0-84938264955%22%29+OR+%28%222-s2.0-84864421794%22%29+OR+%28%222-s2.0-0030212126%22%29+OR+%28%222-s2.0-0033726628%22%29+OR+%28%222-s2.0-78650803456%22%29+OR+%28%222-s2.0-85053318433%22%29+OR+%28%222-s2.0-85111222398%22%29%29+AND+NOT+EID+%282-s2.0-85146317944%29&relpos=17&citeCnt=0&searchTerm=

Robust trajectory design for object throwing based on sensitivity for model uncertainties: https://www.scopus.com/record/display.uri?eid=2-s2.0-84938269497&origin=resultslist&sort=r-f&src=s&nlo=&nlr=&nls=&mltEid=2-s2.0-85146317944&mltType=ref&mltAll=t&imp=t&sid=dcbfc07bca1dc208a81672f85efcf864&sot=mlt&sdt=mlt&sl=579&s=REFEID%28%28%222-s2.0-0027835913%22%29+OR+%28%222-s2.0-85124369853%22%29+OR+%28%222-s2.0-85068442743%22%29+OR+%28%222-s2.0-85089524155%22%29+OR+%28%222-s2.0-84864047294%22%29+OR+%28%222-s2.0-0032759587%22%29+OR+%28%222-s2.0-69549117333%22%29+OR+%28%222-s2.0-70350389386%22%29+OR+%28%222-s2.0-84872338783%22%29+OR+%28%222-s2.0-84999015488%22%29+OR+%28%222-s2.0-84893782296%22%29+OR+%28%222-s2.0-85049685503%22%29+OR+%28%222-s2.0-78651496635%22%29+OR+%28%222-s2.0-67650302094%22%29+OR+%28%222-s2.0-84938264955%22%29+OR+%28%222-s2.0-84864421794%22%29+OR+%28%222-s2.0-0030212126%22%29+OR+%28%222-s2.0-0033726628%22%29+OR+%28%222-s2.0-78650803456%22%29+OR+%28%222-s2.0-85053318433%22%29+OR+%28%222-s2.0-85111222398%22%29%29+AND+NOT+EID+%282-s2.0-85146317944%29&relpos=21&citeCnt=6&searchTerm=

