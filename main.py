import gymnasium
import wandb
import torch
from hyperparameters import MAX_EPOCHS
from dqn import DQN


env = gymnasium.make("ALE/Breakout-v5", render_mode="rgb_array")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQN(env, compute_device=device)

wandb.init(
    project="dqn-atari",
    config={
        "epochs": MAX_EPOCHS,
        "architecture": "DQN",
    }
)
agent.play_and_train()
env.close()

wandb.finish()
