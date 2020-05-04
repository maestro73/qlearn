from agents.default import Agent
from environments.bitmex_prices.base import Environment
from utils import plotLearning
from settings import VISUALIZATION_ROOT
import gym
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    batch_size = 16  # Prices per observation
    max_episode_count = 50  # Max episode count, 0 - run until done
    price_memory_size = batch_size * 4 * 1000  # 0 - All data
    balance = 100

    env = Environment(
        batch_size=batch_size,
        balance=balance,
        price_memory_size=price_memory_size,
        max_episode_count=max_episode_count,
    )

    learning_rate = 0.001  # 0.001 Tutorial example
    scenarios = 10000

    agent = Agent(
        gamma=0.9,  # 0.09 tutorial example
        epsilon=1.0,
        learning_rate=learning_rate,
        input_dims=env.observation_space,
        action_count=env.action_space.n,
        mem_size=1000000,
        batch_size=batch_size,
        epsilon_end=0.1,  # 0.01 tutorial example
        model_name=env.model_name
    )

    score_memory = []
    eps_memory = []

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
            print('Scenario: ', i,)
            print('Episodes:', info['episodes'], info['action_memory'])
            # print('Reward', reward)
            print('Score:', score)
            print('AVG:', np.average(score_memory))
            print('----------------------------------------------------------')

    filename = f'{VISUALIZATION_ROOT}{env.model_name}.png'  # Tutorial material
    x = [i+1 for i in range(scenarios)]  # Tutorial material
    plotLearning(x, score_memory, eps_memory, filename)  # Tutorial material

    agent.save_model()
