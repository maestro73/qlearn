from environments.actions import ActionSpace
import gym
import numpy as np
import random


class Environment(gym.Env):

    def __init__(self, batch_size, balance, random_observations=True):

        self._init_balance = balance

        self._batch_size = batch_size
        self._observation_max = 100
        self._observation_min = 0

        self._buy_price = 0

        self.action_space = ActionSpace(3)
        self.observation_space = (self._batch_size, )

    def step(self, action):

        info = {}
        done = False
        observation = self._random_observation
        price = observation[-1]

        # INIT NET WORTH
        start_net_worth = self._balance + (self._goods_held * price)

        # NEUTRAL
        if action == 0:
            pass

        # LONG
        if action == 1:
            if self._balance > price:
                can_buy_x = int(self._balance / price)

                if can_buy_x > 0:
                    self._goods_held = can_buy_x
                    self._balance -= (can_buy_x * price)
                    self._buy_price = price

        # SHORT
        if action == 2:
            if self._goods_held > 0:
                self._balance += (self._goods_held * price)
                self._goods_held = 0

        self._actions += 1

        # END NET WORTH
        end_net_worth = self._balance + (self._goods_held * self._buy_price)

        reward = (end_net_worth - start_net_worth) / self._actions

        if self._balance < 0:
            done = True

        if self._balance < price / 2 and self._goods_held == 0:
            reward = self._balance - self._init_balance
            done = True

        if self._balance > self._init_balance:
            done = True

        if done is True:
            info['actions'] = self._actions
            info['balance'] = self._balance
            info['goods_held'] = self._goods_held

        return (observation, reward, done, info)

    def reset(self):
        self._balance = self._init_balance
        self._goods_held = 0
        self._actions = 0
        return self._random_observation

    @property
    def model_name(self):
        return f'random-{datetime.datetime.now()}'

    @property
    def _random_price(self):
        return np.random.uniform(
            self._observation_min,
            self._observation_max,
        )

    @property
    def _random_seed(self):
        return np.random.uniform(
            self._observation_min,
            self._observation_max,
            self._batch_size
        )

    @property
    def _random_observation(self):
        observation = self._random_seed.tolist()
        observation.append(self._random_price)
        return observation[-self._batch_size:]
