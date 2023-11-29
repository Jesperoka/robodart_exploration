from gymnasium import Env
from typing import Unpack
import numpy as np
from numpy._typing import NDArray
from typeguard import typechecked 
from tqdm import tqdm
import torch as T
import torch.nn.functional as F


from rl_function_approximators.neural_networks import (HybridActorNetwork,
                                                       HybridCriticNetwork)
from utils.common import all_finite_in, all_finite_out, all_np_dtype_in, all_t_dtype_in, all_t_dtype_out, plot_learning_curve 
from utils.dtypes import NP_DTYPE, NP_ARRTYPE, T_ARRTYPE 
from utils import data

rng = np.random.default_rng()

# NOTE: considering switching to Reverb in future
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
        lr_critic_1=0.0001,
        lr_critic_2=0.0001,
        weight_decay=0.0000, # turned off for now
        state_size=0,
        action_size=0, # TODO: rename to continuous and add toggles
        discrete_action_size=0,
        max_action=0,
        min_action=0,
        alpha=1.0,
        alpha_d=1.0,
        gamma=1.0,
        tau=0.005,
        m_scale=0.9,
        l_zero=-1.0,
        update_actor_every=2,
        update_targets_every=2,
        num_gradient_steps_per_episode=1,
        batch_size=512,
        replay_buffer_size=1000000,
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    ):
        self.device = device
        self.max_action = T.tensor(max_action, device=device)
        self.min_action = T.tensor(min_action, device=device)

        self.target_alpha = T.tensor(-action_size, device=device).detach()
        self.target_alpha_d = T.tensor(-discrete_action_size, device=device).detach()

        self.log_alpha = T.tensor([0.0], requires_grad=True, device=device)
        self.log_alpha_d = T.tensor([0.0], requires_grad=True, device=device)

        self.alpha = self.log_alpha.exp().detach()
        self.alpha_d = self.log_alpha_d.exp().detach()

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

            if step_index % self.update_actor_every == 0:
                self.actor_gradient_step(state)
                self.alpha_gradient_step(state)

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

            # TODO: check performance with the other formulation
            expected_next_q_target = T.sum(next_disc_probs * (next_q_target - self.alpha * next_cont_log_probs - self.alpha_d * next_disc_log_probs), dim=1)

            # TODO: not using for now while implementing hybrid actions
            # Munchausen Reinforcement Learning
            # _, log_probs = self.actor.sample(state, reparameterize=False)
            scaled_log_policy = 0 #self.m_scale * self.alpha * T.clamp(log_probs, min=self.l_zero, max=0)

            # Temporal Difference
            q_target = (reward + scaled_log_policy + self.gamma * (1 - done) * (expected_next_q_target)).view(-1)
            # q_target = (reward + scaled_log_policy + self.gamma * (1 - done) * (next_q_target - self.alpha * next_cont_log_prob)).detach()

        # Compute critic loss
        q1 = self.critic_1(state, cont_action).gather(1, disc_action.long().view(-1, 1)).view(-1)
        q2 = self.critic_2(state, cont_action).gather(1, disc_action.long().view(-1, 1)).view(-1)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss
        # TODO: prioritized experience replay

        # Backpropagate critic networks
        self.critic_1.zero_grad(set_to_none=True)
        self.critic_2.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_1.optimizer.step()


    @all_t_dtype_in(method=True)
    def actor_gradient_step(self, state):
        cont_action, cont_log_probs, disc_action, disc_log_probs, disc_probs = self.actor.sample(state, reparameterize=True)  
        q = T.min(self.critic_1(state, cont_action), self.critic_2(state, cont_action))

        actor_loss_c = T.sum(disc_probs * (self.log_alpha.exp() * cont_log_probs - q), dim=1).mean()  # - policy_prior_log_probs
        actor_loss_d = T.sum(disc_probs * (self.log_alpha_d.exp() * disc_log_probs - q), dim=1).mean()

        self.actor.zero_grad(set_to_none=True)
        (actor_loss_c + actor_loss_d).backward()
        self.actor.optimizer.step()


    @all_t_dtype_in(method=True)
    def alpha_gradient_step(self, state):
        with T.no_grad():
            _, cont_log_probs, _, disc_log_probs, disc_probs = self.actor.sample(state)
        alpha_c_loss = -T.sum(self.log_alpha * (disc_probs * (cont_log_probs + self.target_alpha)).detach(), dim=1).mean()
        alpha_d_loss = -T.sum(self.log_alpha_d * (disc_probs * (disc_log_probs + self.target_alpha_d)).detach(), dim=1).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_c_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        self.alpha_d_optimizer.zero_grad(set_to_none=True)
        alpha_d_loss.backward()
        self.alpha_d_optimizer.step()
        self.alpha_d = self.log_alpha_d.exp().detach()


def basic_training_loop(env, num_episodes):

    agent = Agent(state_size=env.observation_space.shape[0],
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

    load_checkpoint = False
    load_first = False

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

        # Episode Rollout
        step = 0
        undiscounted_return = 0
        while not done: # TODO: use fixed for loop instead (it's the same)
            cont_action, disc_action, full_tanh_action = agent.choose_action(state)
            next_state, reward, done, info = env.step(cont_action, disc_action)
            agent.remember(state, full_tanh_action, reward, next_state, done)
            state = next_state
            env._render()
            step += 1
            undiscounted_return += reward

            if not load_checkpoint:
                agent.learn(episode)

        # Hindsight Experience Replay 
        # agent.replay_buffer.hindsight_experience_replay(step, k=4)


        if episode % chunk_size == 0 and episode != 0:
            data.save(filepath, [final_distance_data, undiscounted_return_data], labels)

        final_distance_data[idx] = info["distance"]
        undiscounted_return_data[idx] = undiscounted_return 
        
        if episode >= chunk_size:
            window = np.arange(idx, idx - 100, - 1) % chunk_size
            mean_final_distance = final_distance_data[window].mean()

            if mean_final_distance < min_mean_final_distance:
                min_mean_final_distance = mean_final_distance

                if not load_checkpoint: 
                    agent.save_models()

        print('episode', episode, 'distance %.8f' % info["distance"], 'mean distance %.8f' % mean_final_distance, 'return %.8f' % undiscounted_return)
