import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils.dtypes import T_DTYPE

CHECKPOINT_DIR = "./nn_models/sac"

# TODO: add regularization to networks

# TODO: modify to output discrete action-values as well
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir=CHECKPOINT_DIR):

        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        
        assert(q.dtype == T_DTYPE)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# TODO: remove
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir=CHECKPOINT_DIR):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        assert(v.dtype == T_DTYPE)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# TODO: modify to output discrete actions as well
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, min_action, fc1_dims=256, fc2_dims=256, n_actions=None,
                 name='actor', chkpt_dir=CHECKPOINT_DIR):
        super(ActorNetwork, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = T.tensor(max_action).to(self.device)
        self.min_action = T.tensor(min_action).to(self.device)
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1.0)

        assert(mu.dtype == T_DTYPE and sigma.dtype == T_DTYPE)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        gaussian = Normal(mu, sigma)

        if reparameterize: actions = gaussian.rsample()
        else: actions = gaussian.sample()

        # Squash distribution
        tanh_actions = T.tanh(actions)

        # TODO: do scaling and shifting on environment side
        # Scale and Shift actions to environment range
        action = 0.5*(self.max_action - self.min_action)*tanh_actions + 0.5*(self.max_action + self.max_action)
        
        # Log probabilities
        log_probs = gaussian.log_prob(actions) - T.sum(1.0 - T.pow(tanh_actions, 2), dim=1, keepdim=True)
        log_prob = T.sum(log_probs, dim=1, keepdim=True) # TODO: where is this done in paper 

        assert(action.dtype == T_DTYPE and log_prob.dtype == T_DTYPE), str(action.dtype) + " " + str(log_prob.dtype)
        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

