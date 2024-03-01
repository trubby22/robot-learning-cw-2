##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.linalg as LA

import numpy as np

import matplotlib.pyplot as plt

# Imports from this project
import constants
import configuration
from graphics import PathToDraw


ACTION_SIZE = 4
GAMMA = 0.9
epsilon = 0.9
alpha = 0.1

class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

        self.Q = np.random.rand(constants.WORLD_SIZE ** 2, ACTION_SIZE)
        self.V = np.zeros(constants.WORLD_SIZE ** 2)
        self.policy = np.zeros((constants.WORLD_SIZE ** 2, ACTION_SIZE))
        self.values = [V]
        self.total_rewards = []

        self.returns = [[[] for j in range(ACTION_SIZE)] for i in range(constants.WORLD_SIZE ** 2)]

        self.num_episodes = 2_000
        self.max_steps = 1_000
        self.rng = np.random.default_rng()

        self.i = 0
        self.j = 0

    def get_next_action_type(self, state, money_remaining):
        if len(self.next_actions) > 0:
            res = self.next_actions.pop()
            return res
        else:
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        best_a = np.argmax(self.Q[state, :])
        b_policy = np.zeros(ACTION_SIZE)
        for k in range(ACTION_SIZE):
            if k == best_a:
                b_policy[k] = 1 - epsilon + epsilon / ACTION_SIZE
            else:
                b_policy[k] = epsilon / ACTION_SIZE

        r = self.rng.random()
        act = None
        for k in range(ACTION_SIZE):
            p = b_policy[k]
            if p >= r:
                act = k
                break
            r -= p

        assert(act is not None)
        return act

    def get_next_action_testing(self, state):
        pass

    def process_transition(self, state, action, next_state, money_remaining):
        
        prv_state = state
        t, state, reward, done = env.step(act)
        total_reward += reward
        Q[prv_state, act] = Q[prv_state, act] + alpha * (reward + GAMMA * max(Q[state, :]) - Q[prv_state, act])

        if done:
            break

    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
       pass

    def dynamics_model(self, state, action):
        next_state = state + action
        return next_state

    def reward(self, state):
        dist_to_goal = Robot.dist(state, self.goal_state)
        if dist_to_goal < constants.TEST_DISTANCE_THRESHOLD:
            return 100
        else:
            return -dist_to_goal

    def dist(x, y):
       return np.linalg.norm(x=x-y, ord=2)

    def discretise_space(self, state):
        cell_x = int(state[0])
        cell_y = int(state[1])
        return np.array([cell_x, cell_y])

    


class TD_agent(object):
  def solve(self, env):
    """
    Solve a given Maze environment using Temporal Difference learning
    input: env {Maze object} -- Maze to solve
    output:
      - policy {np.array} -- Optimal policy found to solve the given Maze environment
      - values {list of np.array} -- List of successive value functions for each episode
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode
    """

    Q = np.random.rand(constants.WORLD_SIZE ** 2, ACTION_SIZE)
    V = np.zeros(constants.WORLD_SIZE ** 2)
    policy = np.zeros((constants.WORLD_SIZE ** 2, ACTION_SIZE))
    values = [V]
    total_rewards = []

    epsilon = 0.9
    alpha = 0.1
    returns = [[[] for j in range(ACTION_SIZE)] for i in range(constants.WORLD_SIZE ** 2)]

    num_episodes = 2_000
    max_steps = 1_000
    rng = np.random.default_rng()
    i = 0

    while i < num_episodes:
      t, state, reward, done = env.reset()
      total_reward = reward
      for j in range(max_steps):
        best_a = np.argmax(Q[state, :])
        b_policy = np.zeros(ACTION_SIZE)
        for k in range(ACTION_SIZE):
          if k == best_a:
            b_policy[k] = 1 - epsilon + epsilon / ACTION_SIZE
          else:
            b_policy[k] = epsilon / ACTION_SIZE

        r = rng.random()
        act = None
        for k in range(ACTION_SIZE):
          p = b_policy[k]
          if p >= r:
            act = k
            break
          r -= p

        assert(act is not None)
        prv_state = state
        t, state, reward, done = env.step(act)
        total_reward += reward
        Q[prv_state, act] = Q[prv_state, act] + alpha * (reward + GAMMA * max(Q[state, :]) - Q[prv_state, act])

        if done:
          break

      total_rewards.append(total_reward)

      V = np.zeros(constants.WORLD_SIZE ** 2)
      for j in range(constants.WORLD_SIZE ** 2):
        best_a = np.argmax(policy[j, :])
        V[j] = Q[j, best_a]

      values.append(V)

      i += 1

    for i in range(constants.WORLD_SIZE ** 2):
      best_a = np.argmax(Q[i, :])
      policy[i, best_a] = 1

    return policy, values, total_rewards