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

        self.path = []

        self.Q = np.random.rand(constants.WORLD_SIZE ** 2, ACTION_SIZE)
        self.V = np.zeros(constants.WORLD_SIZE ** 2)
        self.policy = np.zeros((constants.WORLD_SIZE ** 2, ACTION_SIZE))
        self.final_policy = np.zeros((constants.WORLD_SIZE ** 2, ACTION_SIZE))
        self.values = [self.V]
        self.total_rewards = []
        self.total_reward = 0

        self.returns = [[[] for j in range(ACTION_SIZE)] for i in range(constants.WORLD_SIZE ** 2)]

        self.num_episodes = 2
        self.max_steps = 2
        self.rng = np.random.default_rng()

        self.i = 0
        self.j = 0

        self.next_actions = ['demo']

    def get_next_action_type(self, state, money_remaining):
        if len(self.next_actions) > 0:
            res = self.next_actions.pop()
            return res
        else:
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        state_ix = self.quantised_space_to_ix(self.quantise_space(state))
        best_a = np.argmax(self.Q[state_ix, :])
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
        unquantised_action = self.unquantise_action(act)
        return unquantised_action

    def get_next_action_testing(self, state):
        state_ix = self.quantised_space_to_ix(self.quantise_space(state))
        one_hot_actions = self.final_policy[state_ix, :]
        quantised_action = int(np.argmax(one_hot_actions))
        unquantised_action = self.unquantise_action(quantised_action)
        return unquantised_action

    def process_transition(self, state, action, next_state, money_remaining):
        
        reward = self.reward(next_state)
        quantised_state = self.quantise_space(state)
        in_goal_state = self.reached_goal(quantised_state)
        state_ix = self.quantised_space_to_ix(quantised_state)
        next_state_ix = self.quantised_space_to_ix(self.quantise_space(next_state))
        action_ix = self.quantise_action(action)

        self.total_reward += reward
        self.Q[state_ix, action_ix] = self.Q[state_ix, action_ix] + alpha * (reward + GAMMA * max(self.Q[next_state_ix, :]) - self.Q[state_ix, action_ix])

        self.path.append(state)

        if not self.demo:
            self.j += 1

        if in_goal_state or self.j == self.max_steps:
            self.next_actions.append('reset')

            self.total_rewards.append(self.total_reward)
            self.total_reward = 0

            path_to_draw = PathToDraw(path=self.path, colour=[0, 0, 255], width=1)
            self.paths_to_draw.append(path_to_draw)
            self.path = []

            V = np.zeros(constants.WORLD_SIZE ** 2)
            for k in range(constants.WORLD_SIZE ** 2):
                best_a = np.argmax(self.policy[k, :])
                V[k] = self.Q[k, best_a]

                best_a = np.argmax(self.Q[k, :])
                self.final_policy[k, best_a] = 1

            self.values.append(V)

            if not self.demo:
                self.j = 0
                self.i += 1

    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # path_to_draw = PathToDraw(path=demonstration_states, colour=[0, 0, 255], width=2)
        # self.paths_to_draw.append(path_to_draw)
        n = demonstration_states.shape[0]
        self.demo = True
        for i in range(n - 1):
            state = demonstration_states[i]
            action = demonstration_actions[i]
            next_state = demonstration_states[i + 1]
            self.process_transition(state, action, next_state, money_remaining)
        self.demo = False

        print('demo has been processed')

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
    
    def quantised_space_to_ix(self, state):
        cell_x = int(state[0])
        cell_y = int(state[1])
        return constants.WORLD_SIZE * cell_x + cell_y

    def ix_to_quantised_space(self, n):
        cell_x = n // constants.WORLD_SIZE
        cell_y = n % constants.WORLD_SIZE
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
        a /= 2
        # 0 <= a < 1
        a *= ACTION_SIZE
        res = int(a)
        if res == ACTION_SIZE:
            res = ACTION_SIZE - 1
        return res

    def unquantise_action(self, quantised_a):
        actions = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        action = actions[quantised_a]
        action = np.array(action, dtype=np.float32)
        return action.reshape((2, 1))

    def reached_goal(self, quantised_state):
       goal = self.quantise_space(self.goal_state)
       return (quantised_state == goal).all()
    
    # def calc_final_policy(self):
    #     for i in range(constants.WORLD_SIZE ** 2):
    #         best_a = np.argmax(self.Q[i, :])
    #         self.final_policy[i, best_a] = 1
































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