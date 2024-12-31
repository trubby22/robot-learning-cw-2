##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.linalg as LA
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from matplotlib import pyplot as plt

# Imports from this project
import constants
import configuration
from graphics import PathToDraw

class MyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.SELU(),
            nn.Linear(in_features=10, out_features=10),
            nn.SELU(),
            nn.Linear(in_features=10, out_features=2),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []
        self.net = MyNet()
        self.planning_states = np.zeros((5, constants.DEMOS_CEM_PATH_LENGTH, 2, 1))
        self.planning_actions = np.zeros((5, constants.DEMOS_CEM_PATH_LENGTH, 2, 1))
        self.demos_consumed = 0
        self.testing = False
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=1e-2)

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if money_remaining > 30:
            print('demo', 'money', money_remaining)
            return 'demo'
        else:
            return 'reset'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        """
        this gets called 1st when I request step
        """
        random_action = np.random.uniform([-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
        return random_action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        if not self.testing:
            self.test_mode()
            self.testing = True

        state = torch.from_numpy(state.squeeze()).to(torch.float32)
        res = constants.ROBOT_MAX_ACTION * self.net(state)
        return res.numpy().reshape((2, 1))

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens
        """
        this gets called 2nd when I request step
        """
        pass

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received
        # Currently, nothing happens
        """
        this gets called when I request demo
        """
        # start_ix = self.demos_consumed * constants.DEMOS_CEM_PATH_LENGTH
        # end_ix = (self.demos_consumed + 1) * constants.DEMOS_CEM_PATH_LENGTH
        self.planning_states[self.demos_consumed] = demonstration_states.reshape((constants.DEMOS_CEM_PATH_LENGTH, 2, 1))
        self.planning_actions[self.demos_consumed] = demonstration_actions.reshape((constants.DEMOS_CEM_PATH_LENGTH, 2, 1))
        print(self.demos_consumed)
        self.demos_consumed += 1
        if self.demos_consumed == 4:
            self.train_network()

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state

    def train_network(self):
        print('training')

        losses = []
        iterations = []

        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

        buffer_states = torch.from_numpy(self.planning_states[ : self.demos_consumed].reshape((self.demos_consumed * constants.DEMOS_CEM_PATH_LENGTH, 2))).to(torch.float32)
        buffer_actions = torch.from_numpy(self.planning_actions[ : self.demos_consumed].reshape((self.demos_consumed * constants.DEMOS_CEM_PATH_LENGTH, 2))).to(torch.float32) / constants.ROBOT_MAX_ACTION

        num_iterations = 1_000
        batch_size = 64
        for training_iteration in range(num_iterations):
            self.optimiser.zero_grad()

            minibatch_indices = np.random.choice(buffer_states.shape[0], batch_size)
            minibatch_indices = torch.from_numpy(minibatch_indices)

            minibatch_inputs = buffer_states[minibatch_indices]
            minibatch_labels = buffer_actions[minibatch_indices]

            network_prediction = self.net.forward(minibatch_inputs)
            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels)
            loss.backward()
            self.optimiser.step()
            loss_value = loss.item()
            
            if training_iteration % 10 == 0:
                print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
            losses.append(loss_value)
            iterations.append(training_iteration)

        ax.plot(iterations, losses, color='blue')
        plt.yscale('log')
        # plt.show()
        # plt.pause(0.1)
        fig.savefig(fr"loss_curve.png")
    
    def test_mode(self):
        print('test mode on')
        self.net = self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
