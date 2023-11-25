import gymnasium
from rl_paper.dqn import DQN
import wandb
from hyperparameters import MAX_EPOCHS

wandb.init(
    project="dqn-atari",
    config={
        "epochs": MAX_EPOCHS,
        "architecture": "DQN",
    }
)
env = gymnasium.make("ALE/Breakout-v5")
env.reset(seed=42)

agent = DQN(env)
agent.play_and_train()
env.close()

wandb.finish()