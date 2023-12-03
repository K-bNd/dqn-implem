import random
import typing
from math import exp
import torch

from gymnasium import Env
from hyperparameters import MAX_EPOCHS, MAX_ACTION, REPLAY_SIZE, GAMMA, \
    UPDATE_FREQUENCY, RANDOM_SEED, MINIBATCH_SIZE, TAU
from model import CNN
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
        replay_size: int = REPLAY_SIZE,
        gamma: float = GAMMA,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 10000,
        soft_update=False
    ):
        """
        DQN Implementation

        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.timestep = 0
        self.compute_device = compute_device
        self.model = CNN(env.action_space.n).to(compute_device)
        self.target_model = CNN(env.action_space.n).to(compute_device)
        # copying model state to target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_learning_rate = TAU
        self.soft_update = soft_update
        self.replay_memory = ReplayMemory(replay_size)
        random.seed = RANDOM_SEED

    def update(self) -> None:
        """
        Update the model weights based on the replay memory
        """
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return
        memory_batch: list[Memory] = self.replay_memory.get_sample()
        batch = Memory(*zip(*memory_batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.new_state)), device=self.compute_device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.new_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model.forward(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(
            MINIBATCH_SIZE, device=self.compute_device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model.forward(
                non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

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
            target_net_state_dict[key] = policy_net_state_dict[key]*self.target_learning_rate + \
                target_net_state_dict[key]*(1-self.target_learning_rate)
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

        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            exp(-1. * self.timestep / self.epsilon_decay_steps)

        if random.random() > epsilon:
            action = self.get_best_action(state)
        if self.timestep < self.epsilon_decay_steps:
            self.timestep += 1

        return action

    def save_model_state(self, filename) -> None:
        """
        Save model state to disk
        """
        torch.save(self.model.state_dict(), filename)

    def load_model_state(self, filename) -> None:
        """
        Load model state from filename
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.compute_device)

    def play_and_train(self, epochs=MAX_EPOCHS, t_max=MAX_ACTION) -> None:
        """
        Trains the model and logs both game_reward and total_reward at the end of training.
        """
        state, _ = self.env.reset(seed=RANDOM_SEED)
        state = get_tensor_from_state(state, self.compute_device)
        for epoch in range(epochs):
            game_reward: typing.SupportsFloat = 0.0
            for t in range(t_max):
                action = self.get_action(state)
                new_state, reward, term, trunc, _ = self.env.step(
                    action.item())
                new_state = get_tensor_from_state(
                    new_state, self.compute_device)
                # put new state in replay_memory
                self.replay_memory.update_memory(
                    state, action, reward, new_state, term or trunc)

                game_reward += reward
                self.update()

                if self.soft_update:
                    self.target_update()
                # update model every UPDATE_FREQUENCY
                elif not t % UPDATE_FREQUENCY:
                    self.target_model.load_state_dict(self.model.state_dict())

                if term or trunc:
                    print(f"epoch: {epoch} / {epochs}, score: {game_reward}")
                    wandb.log({"game_reward": game_reward})
                    state, _ = self.env.reset(seed=RANDOM_SEED)
                    state = get_tensor_from_state(state, self.compute_device)
                    break
                state = new_state
