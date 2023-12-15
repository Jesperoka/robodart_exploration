from typing import Callable
import numpy as np
from typeguard import typechecked
from rl_environment.mujoco_gym import FrankaEmikaDartThrowEnv

from utils.common import all_finite_in, all_finite_out
from utils.dtypes import NP_ARRTYPE, NP_DTYPE

rng = np.random.default_rng()


# NOTE: considering switching to Reverb in future
class ReplayBuffer():

    def __init__(self, replay_buffer_size: int, state_size: int, action_size: int):
        self.mem_size = replay_buffer_size
        self.mem_cntr = 0
        self.filled_once = False
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
        self.filled_once = False if (not self.filled_once and self.mem_cntr != self.mem_size) else True 

    @all_finite_out
    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size) if not self.filled_once else self.mem_size 

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, new_states, done

    def hindsight_experience_replay(self, num_episode_steps: int, reward_function: Callable, decompose_state: Callable, k: int = 4):
        episode_start = self.get_episode_start_index(num_episode_steps)
        episode_end = episode_start + num_episode_steps

        # Calculate indices as if infinite memory
        start_indices = np.arange(episode_start + 1, episode_end)
        end_indices = episode_end * np.ones_like(start_indices)
        new_goal_indices = rng.integers(start_indices, end_indices, size=(k, max(start_indices.shape)), endpoint=True)

        # Correct any indices past memory size
        new_goal_indices = np.mod(new_goal_indices.flatten(), self.mem_size)

        # Perform hindsight replay
        for goal_mem_idx in new_goal_indices:
            state = self.state_memory[goal_mem_idx].copy()
            next_state = self.next_state_memory[goal_mem_idx].copy()

            # Decompose state and next_state to easily change goals
            (joint_angs, joint_ang_vels, remaining_time, released, releasing, _, NEW_launch_pt, NEW_launch_vel,
                dart_pos, dart_vel) = decompose_state(state)

            (next_joint_angs, next_joint_ang_vels, next_remaining_time, next_released, next_releasing, _, _, _,
                next_dart_pos, next_dart_vel) = decompose_state(next_state)

            # Assign new goals to the dart positions and velocities
            NEW_goal = next_dart_pos.copy()              
            if next_releasing:
                NEW_launch_pt = next_dart_pos.copy()              
                NEW_launch_vel = next_dart_vel.copy()              

            # Recombine into state and next state with new goals
            state = np.concatenate((joint_angs, joint_ang_vels, remaining_time, released, releasing, NEW_goal, NEW_launch_pt, NEW_launch_vel, dart_pos, dart_vel), axis=0) 
            next_state = np.concatenate((next_joint_angs, next_joint_ang_vels, next_remaining_time, next_released, next_releasing, NEW_goal, NEW_launch_pt, NEW_launch_vel, dart_pos, dart_vel), axis=0) 

            _, reward = reward_function(next_state)
            if self.reward_memory[goal_mem_idx] == reward: continue # Filtered HER for bias reduction

            action = self.action_memory[goal_mem_idx].copy()
            done = self.terminal_memory[goal_mem_idx].copy()

            self.store_transition(state, action, reward, next_state, done)

    def get_episode_start_index(self, num_episode_steps: int):
        return (self.mem_cntr - (num_episode_steps-1)) % self.mem_size

