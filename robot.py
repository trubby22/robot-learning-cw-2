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

# TODO - replace deque /w some more basic implementation - maybe a circular buffer list where I keep track of the index

_rng = np.random.default_rng()

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.max_size = size
        self.buffer = [None] * size
        self.size = 0
        self.head = 0

    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type

        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        self.buffer[self.head] = transition
        self.head += 1
        self.size = max(self.size, self.head)
        self.head %= self.max_size
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer

        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        batch_size = min(batch_size, self.size)
        if self.size == self.max_size:
            buffer = self.buffer
        else:
            buffer = self.buffer[ : self.head]
        ixs = _rng.choice(self.size, size=batch_size, replace=False).tolist()
        return [buffer[i] for i in ixs]


class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])

    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN

        Returns:
            outputted value by the DQN
        """
        x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN

    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN

    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return int(_rng.integers(0, num_actions - 1))

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter

    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss

    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates

    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []

        self.memory = ReplayBuffer(1_000)
        layers = [2, 50, 4]
        self.policy_net = DQN([x for x in layers])
        self.target_net = DQN([x for x in layers])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-2)

        self.episode_durations = []
        self.losses = []

        self.loc_ix = 0
        self.glob_ix = 0

        self.epsilon_start = 1.0
        self.epsilon_end = 0.25
        # self.epsilon_test = 0.1
        self.epsilon_diff = self.epsilon_end - self.epsilon_start
        self.epsilon_min = 1_000
        self.epsilon_max = 5_000

        self.num_episodes = 10
        self.max_episode = 1_000

        self.batch_size = 8

        self.next_action = 'demo'

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if self.next_action is not None:
            res = self.next_action
            self.next_action = None
            return res
        else:
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        epsilon = self.calc_epsilon()
        state = torch.from_numpy(state).reshape(-1).float()
        quantised_action = epsilon_greedy(epsilon, self.policy_net, state)
        return Robot.unquantise_action(quantised_action)

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        epsilon = self.epsilon_test
        state = torch.from_numpy(state).reshape(-1).float()
        quantised_action = greedy_action(self.policy_net, state)
        return Robot.unquantise_action(quantised_action)

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens

        reward = self.reward(next_state)
        done = self.reached_goal(next_state)

        reward = torch.tensor([reward])

        state = torch.from_numpy(state).reshape(-1).float()
        next_state = torch.from_numpy(next_state).reshape(-1).float()

        action = Robot.quantise_action(action)
        action = torch.tensor([action]).reshape(-1)

        self.memory.push([state, action, next_state, reward, torch.tensor([done])])

        # Perform one step of the optimization (on the policy network)
        if len(self.memory.buffer) > self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
            # Compute loss
            mse_loss = loss(self.policy_net, self.target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
            self.losses.append(mse_loss)
            # Optimize the model
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()

        self.loc_ix += 1
        if done or self.loc_ix >= self.max_episode:
            self.episode_durations.append(self.loc_ix)
            self.loc_ix = 0
            self.next_action = 'reset'
            update_target(self.target_net, self.policy_net)
        self.glob_ix += 1

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received
        # Currently, nothing happens
        path_to_draw = PathToDraw(path=demonstration_states, colour=[0, 0, 255], width=1)
        self.paths_to_draw.append(path_to_draw)
        n = demonstration_states.shape[0]
        for i in range(n - 1):
            state = demonstration_states[i]
            action = demonstration_actions[i]
            next_state = demonstration_states[i + 1]
            self.process_transition(state, action, next_state, money_remaining)
        
        update_target(self.target_net, self.policy_net)

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state

    def reward(self, state: np.ndarray):
        state = torch.from_numpy(state).float()
        goal_state = torch.from_numpy(self.goal_state).float()
        dist = LA.vector_norm(state - goal_state)
        res = dist
        if dist < constants.TEST_DISTANCE_THRESHOLD:
            res += 1_000
        return res

    def reached_goal(self, state: np.ndarray):
        state = torch.from_numpy(state).float()
        goal_state = torch.from_numpy(self.goal_state).float()
        dist = LA.vector_norm(state - goal_state)
        return dist < constants.TEST_DISTANCE_THRESHOLD

    def quantise_action(action):
        action = np.reshape(action, (2,))
        # angle is relative to ray from origin to position (1, 0), it goes anti-clockwise
        # 0 - up
        # 1 - right
        # 2 - down
        # 3 - left
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

    def unquantise_action(quantised_a):
        actions = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        action = actions[quantised_a]
        action = np.array(action)
        action *= constants.ROBOT_MAX_ACTION
        return action.reshape((2, 1))

    def calc_epsilon(self):
        if self.glob_ix < self.epsilon_min:
            epsilon = self.epsilon_start
        elif self.glob_ix < self.epsilon_max:
            epsilon = self.epsilon_start + (self.glob_ix - self.epsilon_min + 1) / (self.num_episodes - self.epsilon_min) * self.epsilon_diff
        else:
            epsilon = self.epsilon_end
        return epsilon
