import logging

import gym
import numpy as np
from gym import spaces

logger = logging.getLogger(__name__)


class Env4RLClassification(gym.Env):
    metadata = {"render.modes": ["none"]}

    def __init__(self, train_loader, test_loader):
        super(Env4RLClassification, self).__init__()
        self.episode_over = False
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.episode_over = np.array([False] * len(self.current_indices))
        self.true_labels = np.take(self.y, self.current_indices, axis=0).ravel()
        if self.output_shape:
            return np.take(self.X, self.current_indices, axis=0).reshape(
                -1, *self.output_shape
            )
        else:
            return np.take(self.X, self.current_indices, axis=0)

    def step(self, action):
        # Update actions
        self.action = action
        # Take rewards for current actions
        reward = self._get_reward()
        # Update indices

        last_element = self.current_indices[-1]
        if (max(self.current_indices) + self.batch_size) > len(self.X):
            self.episode_over = np.array([True] * len(self.current_indices))
            # greater
            if last_element == max(self.current_indices):
                self.current_indices += self.batch_size
                dif = max(self.current_indices) - len(self.X)
                self.current_indices[
                len(self.current_indices) - dif - 1: len(self.current_indices)
                ] = list(range(dif + 1))
            else:
                self.current_indices = np.arange(
                    last_element + 1, last_element + 1 + self.batch_size, dtype=np.int32
                )
        else:
            self.current_indices += self.batch_size

        # Update states for next step
        self.true_labels = np.take(self.y, self.current_indices, axis=0).ravel()

        if self.output_shape:
            self.status = np.take(self.X, self.current_indices, axis=0).reshape(
                -1, *self.output_shape
            )
        else:
            self.status = np.take(self.X, self.current_indices, axis=0)

        return self.status, reward, self.episode_over, {}

    def _get_reward(self):
        return (self.action == self.true_labels) * 1

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        raise NotImplementedError
