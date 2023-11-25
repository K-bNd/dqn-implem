from hyperparameters import REPLAY_SIZE, MINIBATCH_SIZE
import numpy as np
import typing


class ReplayMemory:
    """
    Store state memory
    """

    def __init__(self, n=REPLAY_SIZE, minibatch_size=MINIBATCH_SIZE):
        self.replay_size = n
        self.replay_memory = []
        self.minibatch_size = minibatch_size
        self.rng = np.random.default_rng(seed=42)

    def get_sample(self) -> np.ndarray:
        """
        Get n random memory from source
        """
        return self.rng.choice(self.replay_memory, self.minibatch_size)

    def update_memory(self, state: np.ndarray, action: int, new_state: np.ndarray, reward: typing.SupportsFloat) -> None:
        """
        Update memory with new status
        """
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)
        self.replay_memory.append({
            "state": state,
            "action": action,
            "new_state": new_state,
            "reward": reward
        })
