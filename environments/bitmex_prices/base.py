from environments.actions import ActionSpace
from bitmex.bitmex_data_reader import BitmexDataReader
import datetime
import gym
import numpy as np
import random


class Environment(gym.Env):

    def __init__(
        self,
        batch_size,
        balance,
        price_memory_size=0,
        max_episode_count=0,
        ):

        self.batch_size = batch_size
        self.price_memory_size = price_memory_size
        self.max_episode_count = max_episode_count
        self.initial_balance = balance

        self._btc_chop_size = 10000

        self.action_space = ActionSpace(3)
        self.observation_space = (self.batch_size, )

        self._price_memory = BitmexDataReader(
            memory_size=self.price_memory_size)

    def step(self, action):

        info = {}
        done = False
        reward = None

        # Increment episode count
        self._episodes += 1

        # Add action to action memory
        self._action_memory = np.concatenate([
            self._action_memory,
            np.array([action, ])
        ])

        # Previous observation for price
        previous_observation = self._observation_memory[-1].tolist()

        price = previous_observation[-1] / self._btc_chop_size

        # NEUTRAL
        # =====================================================================
        if action == 0:
            reward = -1

        # LONG
        # =====================================================================
        if action == 1:
            if self._balance > price:
                can_buy_x = int(self._balance / price)

                if can_buy_x > 0:
                    self._btc_held = can_buy_x
                    self._balance -= (can_buy_x * price)
                    self._buy_price = price

            else:
                reward = -1

        # SHORT
        # =====================================================================
        if action == 2:
            if self._btc_held > 0:
                self._balance += (self._btc_held * price)
                self._btc_held = 0
            else:
                reward = -1

        # Calculate reward
        # =====================================================================
        if reward is None:
            reward = self._balance - self.initial_balance

        # Calculate DONE
        # =====================================================================
        if self.max_episode_count > 0:
            # Reached max episode count
            if self._episodes == self.max_episode_count:
                done = True

        # Balance increased, "Sold everything!"
        if self._balance > self.initial_balance and self._btc_held == 0:
            done = True

        # Meta data for display
        if done is True:
            info['episodes'] = self._episodes
            info['balance'] = self._balance
            info['action_memory'] = np.unique(
                self._action_memory, return_counts=True)

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
