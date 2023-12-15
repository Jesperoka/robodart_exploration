import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical 
from utils.common import all_t_dtype_in_out, all_t_dtype_out

from utils.dtypes import T_DTYPE

CHECKPOINT_DIR = "./nn_models/sac"

# TODO: change to explicit use of reparameterization trick: a = mu(phi) + sigma(phi)*epsilon
# TODO: try out hybrid action space formulation 2 

# Object oriented programming, yuck!
class CanSaveWeights(nn.Module):
    checkpoint_file: str
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# Continuous Actions Critic Q-function approximator network
# ---------------------------------------------------------------------------- #
class CriticNetwork(CanSaveWeights):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 action_size,
                 fc1_size=256,
                 fc2_size=256,
                 device=T.device("cpu"),
                 spec_norm=False,
                 name='critic',
                 config_name="NONE",
                 chkpt_dir=CHECKPOINT_DIR):

        super(CriticNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_" + config_name) 

        self.fc1 = nn.Linear(state_size + action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        if spec_norm:
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.q_hat = nn.Linear(fc2_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)
        self.register_grad_clip_hooks()

    @all_t_dtype_in_out(method=True)
    def forward(self, state, action):
        state_action = T.cat([state, action], dim=1)
        q = F.relu(self.fc1(state_action), inplace=True)
        q = F.relu(self.fc2(q), inplace=True)
        q = self.q_hat(q).squeeze()

        return q

    def register_grad_clip_hooks(self):
        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -1.0, 1.0))
# ---------------------------------------------------------------------------- #


# Continuous Actions Actor Policy Network
# ---------------------------------------------------------------------------- #
class ActorNetwork(CanSaveWeights):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 action_size,
                 fc1_size=256,
                 fc2_size=256,
                 device=T.device("cpu"),
                 spec_norm=False,
                 name='actor',
                 config_name="NONE",
                 chkpt_dir=CHECKPOINT_DIR):

        super(ActorNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_" + config_name) 

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        if spec_norm:
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.mu = nn.Linear(fc2_size, action_size)
        self.sigma = nn.Linear(fc2_size, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)
        self.register_grad_clip_hooks()


    @all_t_dtype_in_out(method=True)
    def forward(self, state, eps=1e-6):
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        action_mean = self.mu(x)
        action_std = self.sigma(x)
        action_std = T.clamp(action_std, min=eps, max=1.0)

        return action_mean, action_std

    @all_t_dtype_out
    def sample(self, state, reparameterize=True, eps=1e-6):
        mu, sigma = self.forward(state)
        gaussian = Normal(mu, sigma)

        if reparameterize: actions = gaussian.rsample()
        else: actions = gaussian.sample()

        tanh_actions = T.tanh(actions)

        log_probs = gaussian.log_prob(actions) - T.sum(T.log(1.0 - T.pow(tanh_actions, 2) + eps), dim=1, keepdim=True)
        log_prob = T.sum(log_probs, dim=1, keepdim=True) 

        return tanh_actions.squeeze(), log_prob.squeeze()

    def sample_deterministic(self, state):
        mu, _ = self.forward(state) 
        tanh_action = T.tanh(mu)
        return tanh_action 

    def register_grad_clip_hooks(self):
        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -1.0, 1.0))
# ---------------------------------------------------------------------------- #


# Hybrid Actions Actor Policy Network
# ---------------------------------------------------------------------------- #
class HybridActorNetwork(CanSaveWeights):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 continuous_action_size,
                 discrete_action_size,
                 fc1_size=256,
                 fc2_size=256,
                 spec_norm=False,
                 device=T.device("cpu"),
                 name='actor',
                 config_name="NONE",
                 chkpt_dir=CHECKPOINT_DIR):

        super(HybridActorNetwork, self).__init__()
        self.device = device
        self.cont_action_size = continuous_action_size 
        self.disc_action_size = discrete_action_size

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_" + config_name) 

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        if spec_norm:
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.mu = nn.Linear(fc2_size, discrete_action_size*continuous_action_size)
        self.log_sigma = nn.Linear(fc2_size, discrete_action_size*continuous_action_size)
        self.log_pi_d = nn.Linear(fc2_size, discrete_action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)
        self.register_grad_clip_hooks()


    @all_t_dtype_in_out(method=True)
    def forward(self, state, eps=1e-6):
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        action_mean = self.mu(x)
        action_std = T.clamp(self.log_sigma(x).exp(), min=eps, max=1.0)
        disc_action_distr = self.log_pi_d(x)

        return action_mean, action_std, disc_action_distr 

    # @all_t_dtype_out
    def sample(self, state, eps=1e-6):
        mu, sigma, pi_d = self.forward(state)
        
        gaussian = Normal(mu, sigma)
        discrete = Categorical(logits=pi_d)

        cont_action = gaussian.rsample() 
        disc_action = discrete.sample()

        cont_tanh_action = T.tanh(cont_action)

        # Continuous action log probability
        unparameterized_cont_log_probs = gaussian.log_prob(cont_action) - T.sum(T.log(1.0 - T.pow(cont_tanh_action, 2) + eps), dim=1, keepdim=True)
        cont_log_probs = unparameterized_cont_log_probs.view(-1, self.disc_action_size, self.cont_action_size).sum(dim=2, keepdim=True)

        # Discrete action log probabilities
        disc_probs = discrete.probs
        disc_log_probs = T.log(disc_probs + eps) # type: ignore

        # Select continuous action set based on discrete action
        cont_tanh_action = cont_tanh_action.view(-1, self.disc_action_size, self.cont_action_size)
        cont_tanh_action = cont_tanh_action[T.arange(cont_tanh_action.size(0)), disc_action]

        return cont_tanh_action.squeeze(), cont_log_probs.squeeze(), disc_action.squeeze(), disc_log_probs, disc_probs

    # @all_t_dtype_out
    def sample_deterministic(self, state):
        mu, _, pi_d = self.forward(state) 

        cont_tanh_action = T.tanh(mu)
        disc_action = T.argmax(Categorical(logits=pi_d).probs)

        cont_tanh_action = cont_tanh_action.view(-1, self.disc_action_size, self.cont_action_size)
        cont_tanh_action = cont_tanh_action[T.arange(cont_tanh_action.size(0)), disc_action]

        return cont_tanh_action.squeeze(), disc_action.squeeze()

    def register_grad_clip_hooks(self):
        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -1.0, 1.0))
# ---------------------------------------------------------------------------- #


# Hybrid Actions Critic Q-function approximator network
# ---------------------------------------------------------------------------- #
class HybridCriticNetwork(CanSaveWeights):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 cont_action_size,
                 disc_action_size,
                 fc1_size=256,
                 fc2_size=256,
                 spec_norm=False,
                 device=T.device("cpu"),
                 name='critic',
                 config_name="NONE",
                 chkpt_dir=CHECKPOINT_DIR):

        super(HybridCriticNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_" + config_name) 

        self.fc1 = nn.Linear(state_size + cont_action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        if spec_norm:
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.q_hat = nn.Linear(fc2_size, disc_action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)
        self.register_grad_clip_hooks()

    @all_t_dtype_in_out(method=True)
    def forward(self, state, cont_action):
        state_action = T.cat([state, cont_action], dim=1)
        q = F.relu(self.fc1(state_action), inplace=True)
        q = F.relu(self.fc2(q), inplace=True)
        q = self.q_hat(q).squeeze()

        return q

    def register_grad_clip_hooks(self):
        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -1.0, 1.0))
# ---------------------------------------------------------------------------- #
