# # Source: https://github.com/samre12/gym-cryptotrading/

import gym
import numpy as np


class ActionSpace(gym.Space):

    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(ActionSpace, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "ActionSpace(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, ActionSpace) and self.n == other.n
