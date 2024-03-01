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
        self.values = [self.V]
        self.total_rewards = []
        self.total_reward = 0

        self.returns = [[[] for j in range(ACTION_SIZE)] for i in range(constants.WORLD_SIZE ** 2)]

        self.num_episodes = 2_000
        self.max_steps = 1_000
        self.rng = np.random.default_rng()

        self.i = 0
        self.j = 0

        self.next_action = None

    def get_next_action_type(self, state, money_remaining):
        if self.next_action is not None:
            res = self.next_action
            self.next_action = None
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
        
        reward = self.reward(next_state)
        in_goal_state = self.reached_goal(self.quantise_space(state))

        self.total_reward += reward
        self.Q[state, action] = self.Q[state, action] + alpha * (reward + GAMMA * max(self.Q[next_state, :]) - self.Q[state, action])

        if in_goal_state or self.j == self.max_steps:
            self.next_action = 'reset'
            self.total_rewards.append(self.total_reward)
            self.total_reward = 0
            V = np.zeros(constants.WORLD_SIZE ** 2)
            for k in range(constants.WORLD_SIZE ** 2):
                best_a = np.argmax(self.policy[k, :])
                V[k] = self.Q[k, best_a]

            self.values.append(V)
            self.i += 1

    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
       pass

    def dynamics_model(self, state, action):
        next_state = state + action
        return next_state

    def reward(self, quantised_state):
        goal = self.quantise_space(self.goal_state)
        dist_to_goal = self.dist(quantised_state, goal)
        if dist_to_goal < constants.TEST_DISTANCE_THRESHOLD:
            return 100
        else:
            return -dist_to_goal

    def dist(self, x, y):
       return np.linalg.norm(x=x-y, ord=2)

    def quantise_space(self, state):
        cell_x = int(state[0])
        cell_y = int(state[1])
        return np.array([cell_x, cell_y]).reshape((2, 1))
    
    def unquantise_space(self, state):
        return state + 0.5
    
    def quantise_action(self, action):
        action = np.reshape(action, (2,))
        # 0 - up
        # 1 - right
        # 2 - down
        # 3 - left
        # angle is relative to ray from origin to position (1, 0), it goes anti-clockwise
        a = np.arctan2(action[1], action[0])
        # - np.pi <= a < np.pi
        a += np.pi
        # 0 <= a < 2 * np.pi
        a /= np.pi
        # 0 <= a < 2
        a *= 2
        # 0 <= a < 4
        res = int(a)
        if res == 4:
            res = 3
        return res

    def unquantise_action(self, quantised_a):
        actions = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        action = actions[quantised_a]
        action = np.array(action, dtype=np.float32)
        return action.reshape((2, 1))

    def reached_goal(self, quantised_state):
       goal = self.quantise_space(self.goal_state)
       return quantised_state == goal
    
    def calc_policy(self):
        for i in range(constants.WORLD_SIZE ** 2):
            best_a = np.argmax(self.Q[i, :])
            self.policy[i, best_a] = 1
































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