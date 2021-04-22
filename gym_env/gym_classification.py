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
        self.true_labels = None
        self.action = None
        # self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        # self.train_loader = train_loader
        # self.test_loader = test_loader
        # self.x, self.y = next(iter(self.train_loader))

    def reset(self):
        # self.x, self.y = next(iter(self.train_loader))
        # return self.x
        return None

    def step(self, action):
        self.action = action
        rewards = self._get_reward()
        return None, rewards, False, None

    def save_step_targets(self, true_labels):
        self.true_labels = true_labels

    def _get_reward(self):
        return (self.action == self.true_labels) * 1

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        raise NotImplementedError
