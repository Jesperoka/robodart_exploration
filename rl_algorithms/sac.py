import numpy as np
import torch as T
import torch.nn.functional as F

from rl_function_approximators.neural_networks import (ActorNetwork,
                                                       CriticNetwork,
                                                       ValueNetwork)
from utils.common import plot_learning_curve
from utils.dtypes import NP_DTYPE, T_DTYPE


class ReplayBuffer():

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=NP_DTYPE)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=NP_DTYPE)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=NP_DTYPE)
        self.reward_memory = np.zeros(self.mem_size, dtype=NP_DTYPE)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)  # TODO: investigate 

    def store_transition(self, state, action, reward, state_, done):
        assert(state.dtype == NP_DTYPE and action.dtype == NP_DTYPE and reward.dtype == NP_DTYPE and state_.dtype == NP_DTYPE), state_.dtype
        assert(np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(reward).all() and np.isfinite(state_).all() and np.isfinite(done).all())

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        done = self.terminal_memory[batch]

        assert(np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(rewards).all() and np.isfinite(states_).all() and np.isfinite(done).all())
        return states, actions, rewards, states_, done


class Agent():
    # TODO: set better dafaults
    def __init__(self,
                 alpha=0.00003, # 0.0003
                 beta=0.0001, # 0.0003
                 input_dims=[8],
                 env=None,
                 gamma=0.999,
                 n_actions=2,
                 max_size=100000,
                 tau=0.005,
                 layer1_size=256,
                 layer2_size=256,
                 batch_size=512,
                 reward_scale=0.01,
                 max_action=None,
                 epsilon=0.001):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.epsilon = epsilon

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, name='actor', max_action=max_action)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, flat_observation):
        # Action from uniform distribution
        if np.random.default_rng().uniform() < self.epsilon:
            print("uniform")
            return self.uniform_random_action()

        # Action from current policy
        state = T.from_numpy(flat_observation).unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def uniform_random_action(self):
        return np.random.default_rng().uniform(low=-self.actor.max_action, high=self.actor.max_action).astype(NP_DTYPE)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            layer_weights = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
            value_state_dict[name] = layer_weights 

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # TODO: use from_numpy() instead
        reward = T.tensor(reward, dtype=T_DTYPE).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T_DTYPE).to(self.actor.device)
        state = T.tensor(state, dtype=T_DTYPE).to(self.actor.device)
        action = T.tensor(action, dtype=T_DTYPE).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0 # TODO: investigate

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # What is this
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


# QUESTIONS:
# Is batch size large enough? Do I need batch size to be large enough to capture a full rollout?
# Different scales of action values cause problems
# Need to include baseline controller as part of state?
# Make baseline_controller only depend on time?
# Use baseline_controller to only stabilize lateral movement?
# Use agent to only determine lateral movement and release?

def basic_training_loop(env, n_games):
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(env.action_space.high)
    agent = Agent(env=env,
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.shape[0],
                  max_action=env.action_space.high)

    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'franka_dart_throw.png'

    figure_file = 'logs/' + filename

    best_score = -np.inf
    score_history = []
    load_checkpoint = False 
    load_first = False 

    if load_first:
        agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        done = False

        score = 0
        while not done:
            action = agent.choose_action(observation)

            # Teach to hold dart for longer
            if i <= 25:
                action[-1] = -0.965 + 0.0002*i
            # Uniform exploration
            elif i <= 50:
                action = agent.uniform_random_action()

            if i % 100 == 0: print("action", action[-1])
            observation_, reward, done, info = env.step(action)

            score += reward
            agent.remember(observation, action, reward, observation_, done)

            observation = observation_
            
            # Vizualization
            env.render()

        if not load_checkpoint and ((i-1) % 10 == 0):
            num_gradient_steps = 10 
            for k in range(num_gradient_steps):
                agent.learn() # TODO: can move learning outside of while loop

        if i % 200 == 0: 
        #     env.render()
            if not load_checkpoint:
                x = np.arange(0, i, 1, dtype=np.int32)
                plot_learning_curve(x, score_history, figure_file)

        # print("samples so far:", agent.memory.mem_cntr)
            
        # if env.baseline_controller.do_log:
        #     env.baseline_controller.do_log = False

        # if i % 1000 == 0:
        #     env.baseline_controller.do_log = True

        # if (i-1) % 1000 == 0:
        #     env.baseline_controller.plot_logged()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = score

            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.8f' % score, 'avg_score %.8f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
