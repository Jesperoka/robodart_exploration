from typing import Unpack
import numpy as np
from numpy._typing import NDArray
from typeguard import typechecked 
import torch as T
import torch.nn.functional as F


from rl_function_approximators.neural_networks import (ActorNetwork,
                                                       CriticNetwork)
from rl_algorithms.replay_buffer import ReplayBuffer

from utils.common import all_np_dtype_in, all_t_dtype_in 
from utils.dtypes import NP_DTYPE, NP_ARRTYPE, T_ARRTYPE 
from utils import data

# Object oriented programming, yuck!
class CanSaveModels():
    actor: T.nn.Module
    critic_1: T.nn.Module
    critic_2: T.nn.Module
    critic_1_target: T.nn.Module
    critic_2_target: T.nn.Module

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


class SAC(CanSaveModels):
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __init__(
        self,
        min_action,
        max_action,
        state_size,
        action_size,
        config_name,
        munchausen=False,
        HER=False,              # not yet toggleable
        LaBER=False,            # not yet toggleable
        tune_alpha=True,        # not yet toggleable
        larger_nn=False,        # not yet toggleable
        spectral_norm=False,    # not yet toggleable
        lr_actor=0.0003,
        lr_critic_1=0.0001,
        lr_critic_2=0.0001,
        weight_decay=0.0000, # turned off for now
        alpha=1.0,
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

        self.munchausen = munchausen

        self.target_alpha = T.tensor(-action_size, device=device).detach()
        self.log_alpha = T.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = T.optim.Adam(params=[self.log_alpha], lr=lr_actor)

        self.gamma = T.tensor(gamma, device=device)
        self.tau = tau
        self.m_scale = m_scale
        self.l_zero = l_zero

        self.update_actor_every = update_actor_every
        self.update_targets_every = update_targets_every
        self.num_gradient_steps_per_episode = num_gradient_steps_per_episode

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_size, action_size)

        self.actor = ActorNetwork(lr_actor, weight_decay, state_size, action_size, device=device, name='actor', config_name=config_name)

        self.critic_1 = CriticNetwork(lr_critic_1, weight_decay, state_size, action_size, device=device, name='critic_1', config_name=config_name)
        self.critic_2 = CriticNetwork(lr_critic_2, weight_decay, state_size, action_size, device=device, name='critic_2', config_name=config_name)

        self.critic_1_target = CriticNetwork(lr_critic_1, weight_decay, state_size, action_size, device=device, name='critic_1', config_name=config_name)
        self.critic_2_target = CriticNetwork(lr_critic_2, weight_decay, state_size, action_size, device=device, name='critic_2', config_name=config_name)
        self.update_target_networks(1.0)


    @all_np_dtype_in(method=True)
    def choose_action(self, state): # TODO: make cleaner
        state = T.tensor(state, device=self.device).unsqueeze(0)
        tanh_action, _ = self.actor.sample(state, reparameterize=False)
        disc_action = tanh_action.squeeze()[-1] 
        action = 0.5*(self.max_action - self.min_action)*tanh_action + 0.5*(self.max_action + self.min_action)
        return action.numpy(force=True).squeeze()[:-1], disc_action.numpy(force=True).squeeze(), tanh_action.numpy(force=True).squeeze()

    @all_np_dtype_in(method=True)
    def choose_deterministic_action(self, state):
        state = T.tensor(state, device=self.device).unsqueeze(0)
        tanh_action = self.actor.sample_deterministic(state)
        disc_action = tanh_action.squeeze()[-1] 
        action = 0.5*(self.max_action - self.min_action)*tanh_action + 0.5*(self.max_action + self.min_action)
        return action.numpy(force=True).squeeze()[:-1], disc_action.numpy(force=True).squeeze()

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


    @typechecked
    def numpy_to_tensor_on_device(self, *args: Unpack[tuple[NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NP_ARRTYPE, NDArray[np.uint8]]]):
        return (T.from_numpy(arg).to(self.device) for arg in args)

    def learn(self, step_index: int):
        if self.replay_buffer.mem_cntr < self.batch_size: return

        for _ in range(self.num_gradient_steps_per_episode): # TODO: rename
            state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)
            state, action, reward, next_state, done = self.numpy_to_tensor_on_device(state, action, reward, next_state, done)

            # TODO: impl. LaBER 

            self.critic_gradient_step(state, action, reward, next_state, done)

            if step_index % self.update_actor_every == 0:
                self.actor_gradient_step(state)
                self.alpha_gradient_step(state)

            if step_index % self.update_targets_every == 0:
                self.update_target_networks(self.tau)


    @typechecked 
    def critic_gradient_step(self, state: T_ARRTYPE, action: T_ARRTYPE, reward: T_ARRTYPE, next_state: T_ARRTYPE, done: T_ARRTYPE):
        # Compute target q values
        with T.no_grad():
            next_action, next_log_probs = self.actor.sample(next_state, reparameterize=False)                                                                                                                           
            next_q1_target = self.critic_1_target(next_state, next_action)
            next_q2_target = self.critic_2_target(next_state, next_action)
            next_q_target = T.min(next_q1_target, next_q2_target)

            # Munchausen Reinforcement Learning
            scaled_log_policy = 0.0
            if self.munchausen: 
                _, log_probs = self.actor.sample(state, reparameterize=False)
                scaled_log_policy = self.m_scale * self.alpha * T.clamp(log_probs, min=self.l_zero, max=0)

            q_target = (reward + scaled_log_policy + self.gamma * (1 - done) * (next_q_target - self.alpha * next_log_probs)).view(-1)

        # Compute critic loss
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = 0.5*(q1_loss + q2_loss)

        # Backpropagate critic networks
        self.critic_1.zero_grad(set_to_none=True)
        self.critic_2.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_1.optimizer.step()


    @all_t_dtype_in(method=True)
    def actor_gradient_step(self, state):
        action, log_probs = self.actor.sample(state, reparameterize=True)  
        q = T.min(self.critic_1(state, action), self.critic_2(state, action))

        actor_loss_c = (self.log_alpha.exp() * log_probs - q).mean()

        self.actor.zero_grad(set_to_none=True)
        actor_loss_c.backward()
        self.actor.optimizer.step()


    @all_t_dtype_in(method=True)
    def alpha_gradient_step(self, state):
        with T.no_grad():
            _, log_probs = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_alpha).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

