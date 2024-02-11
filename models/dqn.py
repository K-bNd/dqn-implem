import random
from math import exp
import torch
import numpy as np

from gymnasium import Env
from models.cnn import CNN
from train import Configuration
from utils import Action, State, Memory, ReplayMemory, get_tensor_from_state

import wandb


class DQN:
    """
    Deep QLearning Implementation
    """

    def __init__(
        self,
        env: Env,
        compute_device: torch.device,
        config: Configuration = None,
        soft_update=False,
    ):
        """
        DQN Implementation

        """

        random.seed = config.random_seed
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.config = config
        self.timestep = 0
        self.compute_device = compute_device
        self.model = CNN(env.action_space.n).to(compute_device)
        self.target_model = CNN(env.action_space.n).to(compute_device)
        # copying model state to target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.soft_update = soft_update
        self.replay_memory = ReplayMemory(self.config, compute_device=compute_device)

    def update(self) -> None:
        """
        Update the model weights based on the replay memory
        """
        if len(self.replay_memory) < self.config.minibatch_size:
            return
        memory_batch: list[Memory] = self.replay_memory.get_sample()
        batch = Memory(*zip(*memory_batch))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.new_state)),
            device=self.compute_device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(
            self.config.minibatch_size, device=self.compute_device
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_model.forward(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.config.gamma
        ) + reward_batch

        self.model.backward(state_action_values, expected_state_action_values)

    def target_update(self) -> None:
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        based on https://arxiv.org/pdf/1509.02971.pdf
        """
        target_net_state_dict = self.model.state_dict()
        policy_net_state_dict = self.target_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.config.target_learning_rate + target_net_state_dict[key] * (
                1 - self.config.target_learning_rate
            )
        self.target_model.load_state_dict(target_net_state_dict)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (the cnn model).
        """
        return self.model.forward(state).max(1).indices.view(1, 1)

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.
        """
        action = self.action_space.sample()

        epsilon = self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * exp(-1.0 * self.timestep / self.config.epsilon_decay_steps)

        if random.random() > epsilon:
            action = self.get_best_action(state)
        if self.timestep < self.config.epsilon_decay_steps:
            self.timestep += 1

        return action

    def save_model_state(self, filename: str) -> None:
        """
        Save model state to disk
        """
        torch.save(self.model.state_dict(), filename)

    def load_model_state(self, filename: str) -> None:
        """
        Load model state from filename
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.compute_device)

    def play_and_train(self, epochs=None, max_action=None) -> None:
        """
        Trains the model and logs both game_reward and total_reward at the end of training.
        """
        if epochs is None:
            epochs = self.config.max_epochs
        if max_action is None:
            max_action = self.config.max_action

        state, _ = self.env.reset(seed=self.config.random_seed)
        state = get_tensor_from_state(state, self.compute_device)
        for epoch in range(epochs):
            game_reward = []
            for t in range(max_action):
                action = self.get_action(state)
                new_state, reward, term, trunc, _ = self.env.step(action.item())
                new_state = get_tensor_from_state(new_state, self.compute_device)
                # put new state in replay_memory
                self.replay_memory.update_memory(
                    state, action, reward, new_state, term or trunc
                )

                game_reward.append(reward)
                self.update()

                if self.soft_update:
                    self.target_update()
                # update model every UPDATE_FREQUENCY
                elif not t % self.config.update_frequency:
                    self.target_model.load_state_dict(self.model.state_dict())

                if term or trunc:
                    print(
                        f"epoch: {epoch} / {epochs}, average_score: {np.mean(game_reward)}, total_score: {np.sum(game_reward)}"
                    )
                    wandb.log({"average_game_score": np.mean(game_reward)})
                    wandb.log({"total_game_score": np.sum(game_reward)})
                    state, _ = self.env.reset(seed=self.config.random_seed)
                    state = get_tensor_from_state(state, self.compute_device)
                    break
                state = new_state

    def evaluate(self, max_action=None) -> list[float]:
        """
        Evaluate the model by playing a game without training
        """
        if max_action is None:
            max_action = self.config.max_action
        state, _ = self.env.reset()
        state = get_tensor_from_state(state, self.compute_device)
        game_reward = []
        for t in range(max_action):
            action = self.get_best_action(state)
            new_state, reward, term, trunc, _ = self.env.step(action.item())
            new_state = get_tensor_from_state(new_state, self.compute_device)
            game_reward.append(reward)
            if term or trunc:
                self.env.reset()
                print(
                    f"lost at step = {t}, average_score: {np.mean(game_reward)}, total_score: {np.sum(game_reward)}"
                )
                break
            state = new_state
        return game_reward
