import gymnasium
import torch
from models.dqn import DQN
import argparse
import yaml


def eval_model(
    env_name: str = "ALE/Boxing-v5", model_path: str = "model_weights/model.pt"
):
    """Inference function"""
    env = gymnasium.make(env_name, render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(env, compute_device=device)

    agent.load_model_state(model_path)
    agent.evaluate(max_action=3600)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN")

    parser.add_argument(
        "--model_path",
        type=str,
        default="model_weights/model.pt",
        help="Trained DQN model",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="ALE/Boxing-v5",
        help="Name of the Gymnasium environment",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        yml_dict = yaml.load(f, Loader=yaml.FullLoader)

    eval_model(yml_dict, args.env_name)
