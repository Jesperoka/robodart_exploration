import numpy as np
from gymnasium import Env
from typeguard import typechecked

from utils.common import all_finite_in, all_finite_out
from utils.dtypes import NP_ARRTYPE, NP_DTYPE

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
    def store_transition(self, state: NP_ARRTYPE, action: NP_ARRTYPE, reward: NP_DTYPE, next_state: NP_ARRTYPE,
                         done: np.uint8):

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    @all_finite_out
    def sample_buffer(self, batch_size: int):

        max_mem = min(self.mem_cntr,
                      self.mem_size)  # WARNING: with this impl., when mem_cntr loops we kind of reset sampling
        # min_mem = max_mem -

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, new_states, done

    # WARNING: Currently broken
    def hindsight_experience_replay(self, num_episode_steps: int, env: Env, k: int = 4):
        assert (False), "HER not fixed yet"
        episode_start = self.get_episode_start_index(num_episode_steps)
        episode_end = episode_start + num_episode_steps

        # Calculate indices as if infinite memory
        start_indices = np.arange(episode_start + 1, episode_end)
        end_indices = episode_end * np.ones_like(start_indices)
        new_goal_indices = rng.integers(start_indices,
                                        end_indices,
                                        size=(k, max(start_indices.shape)),
                                        endpoint=True)

        # Correct any indices past memory size
        new_goal_indices = np.mod(new_goal_indices.flatten(), self.mem_size)

        # Perform hindsight replay
        for goal_mem_idx in new_goal_indices:
            if self.reward_memory[goal_mem_idx - 1] == NP_DTYPE(0): continue  # Filtered HER

            state = self.state_memory[goal_mem_idx].copy()
            # env.reward_or_terminate()

            goal_idx = 16  # WARNING: hardcoded for now
            dart_idx = 19

            # TODO:
            # if dart is released, skip?
            # if dart is not released,
            #   compute reward as if launch point was the reached point
            #   compute reward as if launch velocity was the reached dart velocity vector

            state[goal_idx:goal_idx + 3] = state[dart_idx:dart_idx + 3]
            action = self.action_memory[goal_mem_idx].copy()
            next_state = self.next_state_memory[goal_mem_idx].copy()
            next_state[goal_idx:goal_idx + 3] = next_state[dart_idx:dart_idx + 3]
            done = self.terminal_memory[goal_mem_idx].copy()

            self.store_transition(state, action, NP_DTYPE(1), next_state, done)

    def get_episode_start_index(self, num_episode_steps: int):
        return (self.mem_cntr - (num_episode_steps-1)) % self.mem_size

