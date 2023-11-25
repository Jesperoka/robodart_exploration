from gymnasium import Env
from typing import Unpack
import numpy as np
from numpy._typing import NDArray
from typeguard import typechecked 
import torch as T
import torch.nn.functional as F


from rl_function_approximators.neural_networks import (HybridActorNetwork,
                                                       HybridCriticNetwork)
from utils.common import all_finite_in, all_finite_out, all_np_dtype_in, all_t_dtype_in, all_t_dtype_out, plot_learning_curve 
from utils.dtypes import NP_DTYPE, NP_ARRTYPE, T_ARRTYPE 

rng = np.random.default_rng()

class ReplayBuffer():
    
    def __init__(self, replay_buffer_size: int, state_size: int, action_size: int):
        self.mem_size = replay_buffer_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_size), dtype=NP_DTYPE)
        self.next_state_memory = np.zeros((self.mem_size, state_size), dtype=NP_DTYPE)
        self.action_memory = np.zeros((self.mem_size, action_size), dtype=NP_DTYPE)
        self.reward_memory = np.zeros(self.mem_size, dtype=NP_DTYPE)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)  


    @typechecked
    @all_finite_in(method=True)
    def store_transition(self, state: NP_ARRTYPE, action: NP_ARRTYPE, reward: NP_DTYPE, next_state: NP_ARRTYPE, done: np.uint8):

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state 
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    @all_finite_out
    def sample_buffer(self, batch_size: int):
        
        max_mem = min(self.mem_cntr, self.mem_size) # WARNING: not correct when mem_cntr loops
        # min_mem = max_mem - 

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, new_states, done 


    # WARNING: Currently broken 
    def hindsight_experience_replay(self, num_episode_steps: int, env: Env, k:int=4):
        episode_start = self.get_episode_start_index(num_episode_steps)
        episode_end = episode_start+num_episode_steps
        
        # Calculate indices as if infinite memory
        start_indices = np.arange(episode_start+1, episode_end) 
        end_indices = episode_end*np.ones_like(start_indices)
        new_goal_indices = rng.integers(start_indices, end_indices, size=(k, max(start_indices.shape)), endpoint=True)

        # Correct any indices past memory size
        new_goal_indices = np.mod(new_goal_indices.flatten(), self.mem_size)

        # Perform hindsight replay 
        for goal_mem_idx in new_goal_indices:
            if self.reward_memory[goal_mem_idx-1] == NP_DTYPE(0): continue # Filtered HER
            
            state = self.state_memory[goal_mem_idx].copy() 
            # env.reward_or_terminate()

            goal_idx = 16 # WARNING: hardcoded for now
            dart_idx = 19

            # TODO: 
            # if dart is released, skip?
            # if dart is not released, 
            #   compute reward as if launch point was the reached point
            #   compute reward as if launch velocity was the reached dart velocity vector  

            

            state[goal_idx:goal_idx+3] = state[dart_idx:dart_idx+3]              
            action = self.action_memory[goal_mem_idx].copy()
            next_state = self.next_state_memory[goal_mem_idx].copy()
            next_state[goal_idx:goal_idx+3] = next_state[dart_idx:dart_idx+3]   
            done = self.terminal_memory[goal_mem_idx].copy() 


            self.store_transition(state, action, NP_DTYPE(1), next_state, done)


    def get_episode_start_index(self, num_episode_steps:int):
        return (self.mem_cntr - (num_episode_steps - 1) ) % self.mem_size


class Agent():
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __init__(
        self,
        lr_actor=0.0003,
        lr_critic_1=0.0003,
        lr_critic_2=0.0003,
        weight_decay=0.0001,
        state_size=0,
        action_size=0, # TODO: rename to continuous and add toggles
        discrete_action_size=0,
        max_action=0,
        min_action=0,
        alpha=1.0,
        alpha_d=1.0,
        gamma=0.99,
        tau=0.005,
        m_scale=0.9,
        l_zero=-1.0,
        update_actor_every=1,
        update_targets_every=1,
        num_gradient_steps_per_episode=1,
        batch_size=512,
        replay_buffer_size=1000000,
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    ):
        self.device = device
        self.max_action = T.tensor(max_action, device=device)
        self.min_action = T.tensor(min_action, device=device)

        self.target_alpha = T.tensor(-action_size, device=device)
        self.target_alpha_d = T.tensor(-discrete_action_size, device=device)

        self.log_alpha = T.tensor([0.0], requires_grad=True, device=device)
        self.log_alpha_d = T.tensor([0.0], requires_grad=True, device=device)

        self.alpha = T.tensor(alpha, device=device)
        self.alpha_d = T.tensor(alpha_d, device=device)

        self.alpha_optimizer = T.optim.Adam(params=[self.log_alpha], lr=lr_actor)
        self.alpha_d_optimizer = T.optim.Adam(params=[self.log_alpha_d], lr=lr_actor)

        # self.action_prior = # TODO: investigate
        self.action_split_index = action_size

        self.gamma = T.tensor(gamma, device=device)
        self.tau = tau
        self.m_scale = m_scale
        self.l_zero = l_zero

        self.update_actor_every = update_actor_every
        self.update_targets_every = update_targets_every
        self.num_gradient_steps_per_episode = num_gradient_steps_per_episode

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_size, action_size + 1)

        self.actor = HybridActorNetwork(lr_actor, weight_decay, state_size, action_size, discrete_action_size, device=device, name='actor')

        self.critic_1 = HybridCriticNetwork(lr_critic_1, weight_decay, state_size, action_size, discrete_action_size, device=device, name='critic_1')
        self.critic_2 = HybridCriticNetwork(lr_critic_2, weight_decay, state_size, action_size, discrete_action_size, device=device, name='critic_2')

        self.critic_1_target = HybridCriticNetwork(lr_critic_1, weight_decay, state_size, action_size, discrete_action_size, device=device, name='critic_1')
        self.critic_2_target = HybridCriticNetwork(lr_critic_2, weight_decay, state_size, action_size, discrete_action_size, device=device, name='critic_2')
        self.update_target_networks(1.0)


    @all_np_dtype_in(method=True)
    def choose_action(self, state): # TODO: make cleaner
        state = T.tensor(state, device=self.device).unsqueeze(0)
        tanh_action, _, disc_action, _, _ = self.actor.sample(state, reparameterize=False)
        tanh_and_disc_action = T.cat((tanh_action, disc_action.unsqueeze(0)))
        action = 0.5*(self.max_action - self.min_action)*tanh_action + 0.5*(self.max_action + self.min_action)
        return action.numpy(force=True).squeeze(), disc_action.numpy(force=True).squeeze(), tanh_and_disc_action.numpy(force=True).squeeze()


    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)


    def update_target_networks(self, tau):
        # Get critic and critic target weights
        c1_weights = self.critic_1.parameters()
        c2_weights = self.critic_2.parameters()
        c1_trgt_weights = self.critic_1_target.parameters()
        c2_trgt_weights = self.critic_2_target.parameters()

        # Soft copy critic weights to targets
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


    @typechecked
    def numpy_to_tensor_on_device(self, *args: Unpack[tuple[NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NDArray[np.uint8]]]):
        return (T.from_numpy(arg).to(self.device) for arg in args)


    def separate_continuous_discrete(self, action: T_ARRTYPE):
        return action[:, 0:self.action_split_index], action[:, self.action_split_index:]


    def learn(self, step_index: int):
        if self.replay_buffer.mem_cntr < self.batch_size: return

        # Perform stochastic batch gradient based optimization num_gradient_steps_per_update times
        for _ in range(self.num_gradient_steps_per_episode):
            # Sample replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)
            state, action, reward, next_state, done = self.numpy_to_tensor_on_device(state, action, reward, next_state, done)

            cont_action, disc_action = self.separate_continuous_discrete(action)

            self.critic_gradient_step(state, cont_action, disc_action, reward, next_state, done)
            cont_action, cont_log_probs, disc_action, disc_log_probs, disc_probs = self.actor.sample(next_state, reparameterize=True)                                                                                                                           

            if step_index % self.update_actor_every == 0:
                self.actor_gradient_step(state, cont_action, cont_log_probs, disc_log_probs, disc_probs)
                self.alpha_gradient_step(cont_log_probs, disc_log_probs, disc_probs)

            if step_index % self.update_targets_every == 0:
                self.update_target_networks(self.tau)


    @typechecked 
    def critic_gradient_step(self, state: T_ARRTYPE, cont_action: T_ARRTYPE, disc_action: T_ARRTYPE, reward: T_ARRTYPE, next_state: T_ARRTYPE, done: T_ARRTYPE):
        # Compute target q values
        with T.no_grad():
            next_cont_action, next_cont_log_probs, next_disc_action, next_disc_log_probs, next_disc_probs = self.actor.sample(next_state, reparameterize=False)                                                                                                                           
            next_q1_target = self.critic_1_target(next_state, next_cont_action)
            next_q2_target = self.critic_2_target(next_state, next_cont_action)
            next_q_target = T.min(next_q1_target, next_q2_target)

            # WARNING: this assumes continuous actions are dependent on discrete ones I think
            expected_next_q_target = T.sum(next_disc_probs * (next_q_target - self.alpha * next_cont_log_probs - self.alpha_d * next_disc_log_probs), dim=1).view(-1)

            # TODO: not using for now while implementing hybrid actions
            # Munchausen Reinforcement Learning
            # _, log_probs = self.actor.sample(state, reparameterize=False)
            scaled_log_policy = 0 #self.m_scale * self.alpha * T.clamp(log_probs, min=self.l_zero, max=0)

            # Temporal Difference
            q_target = (reward + scaled_log_policy + self.gamma * (1 - done) * (expected_next_q_target)).detach()
            # q_target = (reward + scaled_log_policy + self.gamma * (1 - done) * (next_q_target - self.alpha * next_cont_log_prob)).detach()

        # Compute critic loss
        q1 = self.critic_1(state, cont_action).gather(1, disc_action.long().view(-1, 1)).squeeze()
        q2 = self.critic_2(state, cont_action).gather(1, disc_action.long().view(-1, 1)).squeeze()
        q1_loss = 0.5 * F.mse_loss(q1, q_target)
        q2_loss = 0.5 * F.mse_loss(q2, q_target)
        # TODO: prioritized experience replay

        # Backpropagate critic networks
        self.critic_1.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.critic_2.optimizer.step()


    @all_t_dtype_in(method=True)
    def actor_gradient_step(self, state, cont_action, cont_log_probs, disc_log_probs, disc_probs):
        q = T.min(self.critic_1(state, cont_action), self.critic_2(state, cont_action))

        actor_loss_c = T.sum(disc_probs * (self.alpha * cont_log_probs - q), dim=1).mean()  # - policy_prior_log_probs
        actor_loss_d = T.sum(disc_probs * (self.alpha_d * disc_log_probs - q), dim=1).mean()

        self.actor.zero_grad(set_to_none=True)
        (actor_loss_c + actor_loss_d).backward()
        self.actor.optimizer.step()


    @all_t_dtype_in(method=True)
    def alpha_gradient_step(self, cont_log_probs, disc_log_probs, disc_probs):
        alpha_c_loss = -T.sum(self.log_alpha * (disc_probs * (cont_log_probs + self.target_alpha)).detach(), dim=1).mean()
        alpha_d_loss = -T.sum(self.log_alpha_d * (disc_probs * (disc_log_probs + self.target_alpha_d)).detach(), dim=1).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_c_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = T.exp(self.log_alpha)

        self.alpha_d_optimizer.zero_grad(set_to_none=True)
        alpha_d_loss.backward()
        self.alpha_d_optimizer.step()
        self.alpha_d = T.exp(self.log_alpha_d)




def basic_training_loop(env, n_games):
    agent = Agent(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0]-1,
                  discrete_action_size=2,
                  max_action=env.action_space.high[0:-1],
                  min_action=env.action_space.low[0:-1],
                  update_actor_every=1,
                  update_targets_every=1,
                  num_gradient_steps_per_episode=1,
                  )

    filename = 'franka_dart_throw.png'
    figure_file = 'logs/' + filename

    best_score = np.inf # high is bad 
    score_history = []
    load_checkpoint = False
    load_first = False

    if load_first:
        agent.load_models()

    # Episode
    for i in range(n_games):
        state, info = env.reset()
        done = False

        # Episode Rollout
        step = 0
        sum_rewards = 0
        while not done:
            cont_action, disc_action, full_tanh_action = agent.choose_action(state)
            next_state, reward, done, info = env.step(cont_action, disc_action)
            agent.remember(state, full_tanh_action, reward, next_state, done)
            state = next_state
            env._render()
            step += 1
            sum_rewards += reward

        score = info["distance"] # record final distance of dart 

        # Hindsight Experience Replay 
        # agent.replay_buffer.hindsight_experience_replay(step, k=4)

        # Optimization (standard experience replay)
        if not load_checkpoint:
            agent.learn(i)

        if i % 200 == 0:
            x = np.arange(0, i, 1, dtype=np.int32)
            plot_learning_curve(x, score_history, figure_file)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score < best_score:
            best_score = score

            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.8f' % score, 'avg_score %.8f' % avg_score, 'return %.8f' % sum_rewards)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
