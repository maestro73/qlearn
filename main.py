# -*- coding: utf-8 -*-
from agents.default import Agent
from environments.bitmex_prices.default import Environment
from utils import plotLearning
from settings import MODEL_ROOT, VISUALIZATION_ROOT
import datetime
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    batch_size = 64  # Prices per observation
    max_episodes = 100  # Max episode count, 0 - run until done
    memory_size = 60000  # 0 - All available data
    balance = 100

    gamma = 0.9  # 0.09 tutorial example
    epsilon = 1.0
    epsilon_end = 0.1  # 0.01 tutorial example
    learning_rate = 0.01  # 0.001 Tutorial example

    scenarios = 500

    env = Environment(
        batch_size=batch_size,
        balance=balance,
        memory_size=memory_size,
        max_episodes=max_episodes,
    )

    agent = Agent(
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate,
        input_dims=env.observation_space,
        action_count=env.action_space.n,
        mem_size=1000000,
        batch_size=batch_size,
        epsilon_end=epsilon_end,
        model_name=env.model_name
    )

    score_memory = []
    eps_memory = []

    _done = 0

    for i in range(scenarios):

        env.reset()

        done = False
        reward = 0
        observation = env._random_observation

        # Tutorial material
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(
                observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        score = info['balance']

        eps_memory.append(agent.epsilon)
        score_memory.append(score)

        if score > balance:
            _done += 1
            print('Scenario: ', i,)
            print('Episodes:', info['episodes'], info['action_memory'])
            # print('Reward', reward)
            print('Score:', score)
            print('AVG:', np.average(score_memory))
            print('----------------------------------------------------------')

    print('Done count:', _done, 'Done %:', (_done*100) / scenarios)

    positive_outcome = _done
    positive_outcome_percentile = (_done*100) / scenarios

    filename_ext = f'{datetime.datetime.now()}-bitmex'
    filename_ext += f'-Scenarios{scenarios}'
    filename_ext += f'-Episodes{max_episodes}'
    filename_ext += f'-BatchSize{batch_size}'
    filename_ext += f'-Gamma{gamma}'
    filename_ext += f'-Epsilon{epsilon}'
    filename_ext += f'-EpsilonEnd{epsilon_end}'
    filename_ext += f'-LearningRate{learning_rate}'
    filename_ext += f'-PositiveOutcomes{positive_outcome}'
    filename_ext += f'-PositiveOutcomePerc{positive_outcome_percentile}'

    chart_name = VISUALIZATION_ROOT + filename_ext + '.png'

    # Tutorial material
    plotLearning(range(1, scenarios + 1), score_memory, eps_memory, chart_name)

    model_name = MODEL_ROOT + filename_ext + '.h5'

    agent.save_model(model_name)
