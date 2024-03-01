##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np

import torch.linalg as LA

# Imports from this project
import constants
import configuration
from graphics import PathToDraw


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []
        self.rng = np.random.default_rng()

        self.num_iterations = 3
        self.num_paths = 3
        self.path_length = 500
        self.num_elites = 2

        self.planning_actions = np.zeros([self.num_iterations, self.num_paths, self.path_length, 2, 1], dtype=np.float32)
        self.planning_paths = np.zeros([self.num_iterations, self.num_paths, self.path_length, 2, 1], dtype=np.float32)
        self.planning_path_rewards = np.zeros([self.num_iterations, self.num_paths])
        self.planning_mean_actions = np.zeros([self.num_iterations, self.path_length, 2, 1], dtype=np.float32)
        self.variance_actions = np.zeros([self.num_iterations, self.path_length, 2, 1], dtype=np.float32)
        self.next_actions = self.rng.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, [self.num_paths, self.path_length, 2, 1])
        for i in range(self.next_actions.shape[0]):
            for j in range(self.next_actions.shape[1]):
                self.next_actions[i, j] = self.make_max_action(self.next_actions[i, j])

        self.max_action = constants.ROBOT_MAX_ACTION
        self.planned_actions = np.zeros([self.path_length, 2, 1], dtype=np.float32)

        self.i = 0
        self.j = 0
        self.k = 0

        self.p = 0

        self.next_action = None
        self.training = True

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if self.next_action is not None:
            res = self.next_action
            self.next_action = None
            return res
        elif self.training:
            return 'step'
        else:
            return 'demo'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        return self.next_actions[self.j, self.k]

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        res = self.planned_actions[self.p]
        self.p += 1
        return res

    # Function that processes a transition
    def process_transition(self, prev_state, action, state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens

        actions = self.planning_actions
        paths = self.planning_paths
        rewards = self.planning_path_rewards
        mean_actions = self.planning_mean_actions

        actions[self.i, self.j, self.k] = action
        paths[self.i, self.j, self.k] = state

        self.k += 1

        if self.k == self.path_length:
            self.k = 0

            rewards[self.i, self.j] = self.compute_reward(paths[self.i, self.j])

            path_to_draw = PathToDraw(paths[self.i, self.j], colour=(255, 255, 255), width=1)
            self.j += 1
            self.next_action = 'reset'
            self.paths_to_draw.append(path_to_draw)
        
        if self.j == self.num_paths:
            self.j = 0

            elite_ixs = np.argsort(rewards[self.i])[ - self.num_elites : ]
            mean_actions[self.i] = np.average(actions[self.i][elite_ixs], axis=0)
            self.variance_actions[self.i] = np.std(actions[self.i][elite_ixs], axis=0)
            next_actions = self.rng.normal(
                loc=mean_actions[self.i], 
                scale=self.variance_actions[self.i], 
                size=[self.num_paths, self.path_length, 2, 1]
                )
            next_actions = np.clip(a=next_actions, a_min=-self.max_action, a_max=self.max_action)

            self.i += 1
            self.next_action = 'reset'
        
        if self.i == self.num_iterations:
            self.planned_actions = mean_actions[-1]
            print('money_remaining', money_remaining)
            self.training = False


    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received
        # Currently, nothing happens
        pass

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state

    def compute_reward(self, path):
        # reward = np.random.uniform(0, 1)

        reward = -Robot.f(path[-1].squeeze(), self.goal_state)
        # assert(not math.isnan(reward))

        return reward
    
    def f(a, b):
        return np.linalg.norm(x=(a - b), ord=2, axis=0)

    def make_max_action(self, x):
        mag = np.linalg.norm(x=x.squeeze(), ord=2)
        x = x / mag
        x *= constants.ROBOT_MAX_ACTION
        return x.reshape((2, 1))