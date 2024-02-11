import gymnasium
import wandb
import torch
from models.dqn import DQN
import argparse
import yaml
from utils import Configuration


def train(config_dict: dict, env_name: str, name: str):
    """Training function"""

    config = Configuration(config_dict)
    env = gymnasium.make(env_name, render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(env, config=config, compute_device=device)

    wandb.init(
        project="dqn-atari",
        config={"architecture": "DQN", "config": config_dict, "monitor_gym": True},
        name=name,
    )

    agent.play_and_train()
    env.close()

    wandb.finish()

    agent.save_model_state(f"model_weights/dqn_{name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN")

    parser.add_argument(
        "--name", type=str, default="test", help="Name of the training run"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML file")
    parser.add_argument(
        "--env_name",
        type=str,
        default="ALE/Boxing-v5",
        help="Name of the Gymnasium environment",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        yml_dict = yaml.load(f, Loader=yaml.FullLoader)

    train(yml_dict, args.env_name, args.name)
