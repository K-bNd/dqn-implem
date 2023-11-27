import typing
from collections import namedtuple
from hyperparameters import REPLAY_SIZE, MINIBATCH_SIZE, RANDOM_SEED
import numpy as np
import torch
import random

Action = int
State = torch.tensor
Memory = namedtuple("Memory",
                    ("state", "action", "reward", "new_state", "done"))


def get_tensor_from_state(state: np.ndarray, compute_device=torch.device("cuda")) -> torch.tensor:
    """
    Turn observation_space into a tensor
    """
    return torch.tensor(
        state, dtype=torch.float32, device=compute_device).unsqueeze(0).transpose(1, 3)

# class Memory(object):
#     """
#     Store memory information
#     """

#     def __init__(self, state: StateWrapper, action: int, reward: typing.SupportsFloat,
#                  new_state: StateWrapper, done: bool) -> None:
#         self.state = state
#         self.action = action
#         self.reward = reward
#         self.new_state = new_state
#         self.done = done


class ReplayMemory(object):
    """
    Store state memory array
    """

    def __init__(self, n=REPLAY_SIZE, minibatch_size=MINIBATCH_SIZE, compute_device=torch.device("cuda")):
        self.replay_size = n
        self.replay_memory: list[Memory] = []
        self.minibatch_size = minibatch_size
        random.seed = RANDOM_SEED
        self.compute_device = compute_device
    
    def __len__(self):
        return len(self.replay_memory)
    
    def get_sample(self) -> list[Memory]:
        """
        Get random memory batch
        """
        return random.sample(self.replay_memory, self.minibatch_size)

    def update_memory(self, state: State, action: int, reward: typing.SupportsFloat,
                      new_state: State, done: bool) -> None:
        """
        Update memory with new status
        """
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)
        
        reward = torch.tensor([reward], device=self.compute_device)
        action = torch.tensor([[action]], device=self.compute_device)
        self.replay_memory.append(
            Memory(state, action, reward, new_state, done)
        )
