# -*- coding: utf-8 -*-
from environments.actions import ActionSpace
from bitmex.bitmex_data_reader import BitmexDataReader
import datetime
import gym
import numpy as np
import random


class BaseBitmexEnvironment(gym.Env):

    def __init__(self, batch_size, balance, memory_size=0, max_episodes=0):

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.max_episodes = max_episodes
        self.initial_balance = balance

        self._btc_chop_size = 10000

        self.action_space = ActionSpace(3)
        self.observation_space = (self.batch_size, )

        self._price_memory = BitmexDataReader(memory_size=self.memory_size)

        #
        self._balance = self.initial_balance
        self._btc_held = 0
        self._buy_price = 0
        self._episodes = 0
        self._action_memory = np.array([])
        self._observation_memory = np.array([])

    def step(self, action):

        info = {}
        done = False
        reward = None

        # Increment episode count
        self._episodes += 1

        # NEUTRAL
        # =====================================================================
        if action == 0:
            pass

        # LONG
        # =====================================================================
        if action == 1:
            pass

        # SHORT
        # =====================================================================
        if action == 2:
            pass

        # Calculate reward
        # =====================================================================
        if reward is None:
            reward = 0

        # Calculate DONE
        # =====================================================================
        if self.max_episodes > 0:
            if self._episodes == self.max_episodes:
                done = True

        # Add done statement here

        # Get next observation
        # =====================================================================
        observation = self._random_observation
        self._observation_memory = np.concatenate(
            [
                self._observation_memory,
                [observation, ],
            ],
            axis=0
        )

        return (observation.tolist(), reward, done, info)

    def reset(self):
        self._balance = self.initial_balance
        self._btc_held = 0
        self._buy_price = 0
        self._episodes = 0
        self._action_memory = np.array([])
        self._observation_memory = np.array([self._random_observation, ])

    @property
    def model_name(self):
        return f'bitmex-{datetime.datetime.now()}'

    @property
    def _random_observation(self):
        limit = self.batch_size
        _price_memory_length = self._price_memory.data.size
        skip = random.randint(0, _price_memory_length - limit)
        arr = self._price_memory.paginate(skip, skip + limit).tolist()
        return np.array(list(map(lambda x: x['close'], arr)))
