import numpy as np
from utils import numpy_to_tensor

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.max_size = capacity
        self.size = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None

    def push(self, interaction):
        state, action, reward, next_state = interaction

        if self.states is None:
            # Initialize the buffer arrays with the appropriate shapes and datatypes
            self.states = np.empty((self.max_size,) + state.shape, dtype=state.dtype)
            self.actions = np.empty((self.max_size,), dtype=np.int32 if isinstance(action, int) else action.dtype)  # assuming actions are integers
            self.rewards = np.empty((self.max_size,), dtype=np.float32 if isinstance(reward, float) else reward.dtype)  # assuming rewards are floats
            self.next_states = np.empty((self.max_size,) + next_state.shape, dtype=next_state.dtype)
        
        self.states[self.size % self.max_size] = state
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_states[self.size % self.max_size] = next_state

        self.size += 1
    
    def sample(self, batch_size):
        random_indices = np.random.randint(0, self.size % self.max_size, size = batch_size)

        states = numpy_to_tensor(self.states[random_indices])
        actions = numpy_to_tensor(self.actions[random_indices]).long()
        rewards = numpy_to_tensor(self.rewards[random_indices])
        next_states = numpy_to_tensor(self.next_states[random_indices])

        return (states, actions, rewards, next_states)
    