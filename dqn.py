import random
import typing
from copy import deepcopy

import numpy as np
from gymnasium import Env
import wandb

from hyperparameters import MAX_EPOCHS, MAX_ACTION, REPLAY_SIZE, DISCOUNT_FACTOR, UPDATE_FREQUENCY
from model import CNN
from replay_memory import ReplayMemory

Action = int
State = np.ndarray


class DQN:
    """
    Deep QLearning Implementation
    """

    def __init__(
        self,
        env: Env,
        replay_size: int = REPLAY_SIZE,
        learning_rate: float = 1.0,
        gamma: float = DISCOUNT_FACTOR,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 10000
    ):
        """
        DQN Implementation

        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.timestep = 0
        self.model = CNN(env.action_space.n)
        self.target_model = deepcopy(self.model)
        self.replay_memory = ReplayMemory(replay_size)

    def update(
        self, state: State, action: Action, reward: typing.SupportsFloat, next_state: State
    ) -> None:
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = r + gamma * max_a' Q(s', a')
           TD_error(s', a) = TD_target(s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s', a)
        """
        q_value = 0.0
        target = reward + self.gamma * self.get_value(next_state)
        error = target - self.get_qvalue(state, action)
        q_value = self.get_qvalue(state, action) + self.learning_rate * error

        self.set_qvalue(state, action, q_value)
        # TODO: update model parameters

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (the cnn model).
        """
        return np.argmax(self.model.forward(state))

    def reset(self):
        """
        Reset epsilon to the start value.
        """
        self.epsilon = self.epsilon_start
        self.timestep = 0

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greey. Namely, with probability self.epsilon, we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        action = self.action_space.sample()

        epsilon = self.epsilon_start - \
            (self.timestep / self.epsilon_decay_steps) * \
            (self.epsilon_start - self.epsilon_end)
        if random.random() > epsilon:
            action = self.get_best_action(state)
        if self.timestep < self.epsilon_decay_steps:
            self.timestep += 1

        return action

    def play_and_train(self, epochs=MAX_EPOCHS, t_max=MAX_ACTION) -> None:
        """
        Trains the model and logs both game_reward and total_reward at the end of training.
        """
        total_reward: typing.SupportsFloat = 0.0
        state, _ = self.env.reset()

        for _ in range(epochs):
            game_reward: typing.SupportsFloat = 0.0
            for t in range(t_max):
                action = self.get_action(state)
                next_s, reward, done, trunc, _ = self.env.step(action)
                # Train agent for state s

                # update Q parameters every UPDATE_FREQUENCY
                if not t % UPDATE_FREQUENCY:
                    self.update(s, action, reward, next_s)
                s = next_s
                game_reward += reward
                if done:
                    s, _ = self.env.reset()
                    break
            total_reward += game_reward
            wandb.log({"game_reward": game_reward})

        wandb.log({"total_reward": total_reward})
