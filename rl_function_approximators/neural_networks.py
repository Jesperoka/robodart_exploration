import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from utils.dtypes import T_DTYPE

CHECKPOINT_DIR = "./nn_models/sac"


# TODO: modify to output discrete action-values as well
# Critic Q-function approximator network
# ---------------------------------------------------------------------------- #
class CriticNetwork(nn.Module):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 action_size,
                 fc1_size=256,
                 fc2_size=256,
                 device=T.device("cpu"),
                 name='critic',
                 chkpt_dir=CHECKPOINT_DIR):

        super(CriticNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(state_size + action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.q_hat = nn.Linear(fc2_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, state, action):
        state_action = T.cat([state, action], dim=1)
        q = F.relu(self.fc1(state_action), inplace=True)
        q = F.relu(self.fc2(q), inplace=True)
        q = self.q_hat(q).squeeze()

        assert (q.dtype == T_DTYPE)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
# ---------------------------------------------------------------------------- #


# TODO: modify to output discrete actions as well
# Actor Policy Network
# ---------------------------------------------------------------------------- #
class ActorNetwork(nn.Module):

    def __init__(self,
                 lr,
                 weight_decay,
                 state_size,
                 action_size,
                 fc1_size=256,
                 fc2_size=256,
                 device=T.device("cpu"),
                 name='actor',
                 chkpt_dir=CHECKPOINT_DIR):

        super(ActorNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.mu = nn.Linear(fc2_size, action_size)
        self.sigma = nn.Linear(fc2_size, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)

    def forward(self, state, eps=1e-6):
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        action_mean = self.mu(x)
        action_std = self.sigma(x)
        action_std = T.clamp(action_std, min=eps, max=1.0)

        assert (action_mean.dtype == T_DTYPE and action_std.dtype == T_DTYPE)
        return action_mean, action_std

    def sample(self, state, reparameterize=True, eps=1e-6):
        mu, sigma = self.forward(state)
        gaussian = Normal(mu, sigma)

        if reparameterize: actions = gaussian.rsample()
        else: actions = gaussian.sample()

        # Squash distribution
        tanh_actions = T.tanh(actions)

        # Log probabilities
        log_probs = gaussian.log_prob(actions) - T.sum(T.log(1.0 - T.pow(tanh_actions, 2) + eps), dim=1, keepdim=True)
        log_prob = T.sum(log_probs, dim=1, keepdim=True)  # TODO: find where is this done in the paper

        assert (tanh_actions.dtype == T_DTYPE and log_prob.dtype == T_DTYPE)
        return tanh_actions.squeeze(), log_prob.squeeze()

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
# ---------------------------------------------------------------------------- #
