import typing
from collections import namedtuple
from hyperparameters import REPLAY_SIZE, MINIBATCH_SIZE, RANDOM_SEED
import numpy as np
import torch
import random
import cv2

Action = int
State = torch.tensor
Memory = namedtuple("Memory",
                    ("state", "action", "reward", "new_state", "done"))


def preprocess_frame(frame):
    """
    Turn frame into grayscale and resize it to 84x84 image
    """
    # Extract the maximum value for each pixel color value over the current frame and the previous frame
    max_values = np.maximum(frame[1:], frame[:-1])

    # Extract the Y channel (luminance) from the RGB frame
    grayscale_frame = max_values[:, :, 0]

    # Rescale the grayscale frame to 84x84 pixels
    resized_frame = cv2.resize(grayscale_frame, (84, 84))

    return resized_frame


def get_tensor_from_state(state: np.ndarray, compute_device=torch.device("cuda")) -> torch.tensor:
    """
    Turn observation_space into a tensor
    """
    return torch.tensor(
        preprocess_frame(state), dtype=torch.float32, device=compute_device).unsqueeze(0).reshape(1, 1, 84, 84)


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
