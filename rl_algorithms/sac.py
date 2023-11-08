import numpy as np
import torch as T
import torch.nn.functional as F

from rl_function_approximators.neural_networks import (ActorNetwork,
                                                       CriticNetwork)
from utils.common import plot_learning_curve
from utils.dtypes import NP_DTYPE


class ReplayBuffer():

    def __init__(self, replay_buffer_size, state_size, action_size):
        self.mem_size = replay_buffer_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_size), dtype=NP_DTYPE)
        self.new_state_memory = np.zeros((self.mem_size, state_size), dtype=NP_DTYPE)
        self.action_memory = np.zeros((self.mem_size, action_size), dtype=NP_DTYPE)
        self.reward_memory = np.zeros(self.mem_size, dtype=NP_DTYPE)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)  # TODO: investigate

    def store_transition(self, state, action, reward, state_, done):
        assert (state.dtype == NP_DTYPE and action.dtype == NP_DTYPE and reward.dtype == NP_DTYPE
                and state_.dtype == NP_DTYPE), state_.dtype
        assert (np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(reward).all()
                and np.isfinite(state_).all() and np.isfinite(done).all())

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

        assert (np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(rewards).all()
                and np.isfinite(states_).all() and np.isfinite(done).all())
        return states, actions, rewards, states_, done


class Agent():
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __init__(
        self,
        lr_actor=0.0003,
        lr_critic_1=0.0003,
        lr_critic_2=0.0003,
        weight_decay=0.0001,
        state_size=0,
        action_size=0,
        max_action=0,
        min_action=0,
        alpha=1.0,
        gamma=0.99,
        tau=0.005,
        update_actor_every=1,
        update_targets_every=1,
        batch_size=512,
        replay_buffer_size=1000000,
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    ):
        self.device = device
        self.max_action = T.tensor(max_action, device=device)
        self.min_action = T.tensor(min_action, device=device)

        self.target_alpha = T.tensor(-action_size, device=device)
        self.log_alpha = T.tensor([0.0], requires_grad=True, device=device)
        self.alpha = T.tensor(alpha, device=device)
        self.alpha_optimizer = T.optim.Adam(params=[self.log_alpha], lr=lr_actor)
        # self.action_prior = # TODO: investigate

        self.gamma = T.tensor(gamma, device=device)
        self.tau = tau

        self.update_actor_every = update_actor_every
        self.update_targets_every = update_targets_every

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_size, action_size)

        self.actor = ActorNetwork(lr_actor, weight_decay, state_size, action_size, device=device, name='actor')

        self.critic_1 = CriticNetwork(lr_critic_1, weight_decay, state_size, action_size, device=device, name='critic_1')
        self.critic_2 = CriticNetwork(lr_critic_2, weight_decay, state_size, action_size, device=device, name='critic_2')

        self.critic_1_target = CriticNetwork(lr_critic_1, weight_decay, state_size, action_size, device=device, name='critic_1')
        self.critic_2_target = CriticNetwork(lr_critic_2, weight_decay, state_size, action_size, device=device, name='critic_2')
        self.update_target_networks(1.0)

    def choose_action(self, state):
        state = T.tensor(state, device=self.device).unsqueeze(0)
        tanh_actions, _ = self.actor.sample(state, reparameterize=False)
        action = 0.5*(self.max_action - self.min_action)*tanh_actions + 0.5*(self.max_action + self.min_action)
        return action.numpy(force=True).squeeze(), tanh_actions.numpy(force=True).squeeze()

    def uniform_random_action(self, min_action, max_action):
        return np.random.default_rng().uniform(low=min_action, high=max_action).astype(NP_DTYPE)

    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)

    def update_target_networks(self, tau):
        # Get critic and critic target weights
        c1_weights = self.critic_1.parameters()
        c2_weights = self.critic_2.parameters()
        c1_trgt_weights = self.critic_1_target.parameters()
        c2_trgt_weights = self.critic_2_target.parameters()

        # Soft critic weights to targets
        for c1tw, c2tw, c1w, c2w in zip(c1_trgt_weights, c2_trgt_weights, c1_weights, c2_weights):
            c1tw.data.copy_(tau * c1w.data + (1.0-tau) * c1tw.data)
            c2tw.data.copy_(tau * c2w.data + (1.0-tau) * c2tw.data)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_1_target.save_checkpoint()
        self.critic_2_target.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_1_target.load_checkpoint()
        self.critic_2_target.load_checkpoint()

    def numpy_to_tensor_on_device(self, *args):
        return (T.from_numpy(arg).to(self.device) for arg in args)

    def learn(self, step_index):
        if self.replay_buffer.mem_cntr < self.batch_size: return
        # Sample replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)
        state, action, reward, next_state, done = self.numpy_to_tensor_on_device(state, action, reward, next_state, done)

        self.critic_gradient_step(state, action, reward, next_state, done)

        action, log_probs = self.actor.sample(state, reparameterize=True)

        if step_index % self.update_actor_every == 0:
            self.alpha_gradient_step(log_probs)
            self.actor_gradient_step(state, action, log_probs)

        if step_index % self.update_targets_every == 0:
            self.update_target_networks(self.tau)

    def critic_gradient_step(self, state, action, reward, next_state, done):
        # Compute target q values
        with T.no_grad():
            next_action, next_log_probs = self.actor.sample(next_state, reparameterize=False)
            next_q1_target = self.critic_1_target(next_state, next_action)
            next_q2_target = self.critic_2_target(next_state, next_action)
            next_q_target = T.min(next_q1_target, next_q2_target)
            q_target = (reward + self.gamma * (1 - done) * (next_q_target - self.alpha * next_log_probs)).detach()

        # Compute critic loss
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        q1_loss = 0.5 * F.mse_loss(q1, q_target)
        q2_loss = 0.5 * F.mse_loss(q2, q_target)

        # Backpropagate critic networks
        self.critic_1.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.critic_2.optimizer.step()

    def alpha_gradient_step(self, log_probs):
        _alpha = T.exp(self.log_alpha)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_alpha)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = _alpha

    def actor_gradient_step(self, state, action, log_probs):
        q1 = self.critic_1(state, action)
        actor_loss = (self.alpha * log_probs - q1).mean()  # - policy_prior_log_probs
        self.actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor.optimizer.step()


# QUESTIONS:
# Is batch size large enough? Do I need batch size to be large enough to capture a full rollout?
# Different scales of action values cause problems
# Need to include baseline controller as part of state?
# Make baseline_controller only depend on time?
# Use baseline_controller to only stabilize lateral movement?
# Use agent to only determine lateral movement and release?


def basic_training_loop(env, n_games):
    agent = Agent(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0],
                  max_action=env.action_space.high,
                  min_action=env.action_space.low,
                  update_actor_every=1,
                  update_targets_every=1)

    filename = 'franka_dart_throw.png'
    figure_file = 'logs/' + filename

    best_score = -np.inf
    score_history = []
    load_checkpoint = False
    load_first = False

    if load_first:
        agent.load_models()

    # Episode
    for i in range(n_games):
        state, info = env.reset()
        done = False
        score = 0

        # Rollout
        while not done:
            action, tanh_action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, tanh_action, reward, next_state, done)
            score += reward

            state = next_state
            env.render()

        # Optimization
        if not load_checkpoint:
            agent.learn(i)

        if i % 200 == 0:
            x = np.arange(0, i, 1, dtype=np.int32)
            plot_learning_curve(x, score_history, figure_file)

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
