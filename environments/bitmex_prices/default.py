# -*- coding: utf-8 -*-
from environments.bitmex_prices.base import BaseBitmexEnvironment
import numpy as np


class Environment(BaseBitmexEnvironment):

    def step(self, action):

        """
        Sell everything, buy as much as posible!
        Doing wrong thing is bad!
        """

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

        # Balance increased and sold everything
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
