import gym
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

## This script applies deep Q-learning for solving a reinforcement learning problem
## As function approximator for the state-action Q-function a deep neural network will be used
# http://incompleteideas.net/book/first/ebook/node65.html traditional
# Paper: https://arxiv.org/abs/1312.5602
# Slides: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Talks_files/deep_rl.pdf

class DQN(nn.Module):
    def __init__(self, env,
                 discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001):
        super(DQN, self).__init__()
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        ## Build DQN network
        self.dqn = self.build_DQN()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        #elf.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()


    def build_DQN(self):
        """
        Build a deep Q-Network as function approximator for the state-action function using
        PyTorch Sequential API
        """
        model = nn.Sequential(
            nn.Linear(in_features=self.n_states, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_actions),
        )
        print("Deep Q-Network: \n", model)

        if self.device == "cuda":
            model = model.cuda()

        return model

    def build_experience_memory(self, state, action, reward, next_state, done):
        """
        This function creates a memory dataset.
        It stores the transitions that the agent observes, allowing to reuse the data later.
        Sampling from it randomly, the transitions that build up a batch are decorrelated.
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        This function computes the optimal action according to the Q-Learning algorithm.
        Instead of sampling from a policy distribution the Q learning selects that action, that has maximal value
        This implementation includes an epsilon-greedy selection where the 'optimal' action will
        be selected with a specific probability epsilon and the rest with 1-epsilon
        :param state:
        :return: action [python number]
        """

        if np.random.rand() <= self.epsilon:
            ### Exploration
            action = np.random.randint(low=0, high=self.n_actions-1)
        else:
            ### Exploitation: Just take the maximum of the action state-action Q function
            # Compute predicted approximation of Q-values
            state_ = torch.Tensor(state).to(self.device)

            q_values = self.dqn(state_)
            action = torch.argmax(q_values).item()
        return action

    def train_agent(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        ## sample a minibatch
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = torch.Tensor(state).to(self.device)
            next_state = torch.Tensor(next_state).to(self.device)

            if not done:
                q_pred = self.dqn(next_state)
                q_pred_max = q_pred.max().item()
                target = (reward + self.discount_factor*q_pred_max)
            target_f = self.dqn(state)
            target_f[action] = target

            loss = self.loss(input=self.dqn(state), target=target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        ## Decay the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




if __name__ == "__main__":

    ## Create parameters and init openAI environment and DQN-agent
    max_episodes = 1000
    max_timesteps = 100
    env = gym.make("CartPole-v1")
    dqn_agent = DQN(env,
                 discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.005)
    done = False
    batch_size = 32

    cum_rewards = []
    ## Training loop via episodes
    for episode in range(max_episodes):
        rewards_ep = 0
        state = env.reset()
        ## avoid infinite loop. Otherwise while not done
        ## loop samples a trajectory with max_timesteps. If not finished it will break
        for time in range(max_timesteps):
            action = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ## Punish if the system is done with negative reward since we want to have very long timesteps
            reward = reward if not done else -5
            rewards_ep += reward
            ## Append the transitions into agent memory
            dqn_agent.build_experience_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, done at timestep: {}, cumulative undiscounted reward: {} epsilon:{:.2}".format(
                    episode, max_episodes, time, rewards_ep, dqn_agent.epsilon))
                cum_rewards.append(rewards_ep)
                break

            ## Training part if the memory is big enough
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.train_agent(batch_size)
