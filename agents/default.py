# -*- coding: utf-8 -*-
"""
Tutorial material from:
https://github.com/philtabor/Youtube-Code-Repository/
"""
from models.utils import build_dqn
from replay_buffers.default import ReplayBuffer
from tensorflow.keras.models import load_model
from settings import MODEL_ROOT
import numpy as np


class Agent():
    def __init__(
        self, learning_rate, gamma, action_count, epsilon, batch_size,
        input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000,
            model_name='dqn_model'):

        self.action_space = [i for i in range(action_count)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_name = model_name
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(learning_rate, action_count, batch_size)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + (
            self.gamma * np.max(q_next, axis=1) * dones)

        self.q_eval.train_on_batch(states, q_target)

        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.eps_min

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = f'{MODEL_ROOT}{self.model_name}.h5'
        self.q_eval.save(file_name)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
