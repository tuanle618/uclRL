import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class reinforce(nn.Module):

    def __init__(self, env, discount_factor=1.0):
        """
        Initializes an agent which uses a neural network as function approximator for the policy
        :param env: openAI environment
        :param discount_factor:
        """
        super(reinforce, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.discount_factor = discount_factor

        # PyTorch building blocks
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0], out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=env.action_space.n)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Computes a forward pass of input (states) into probability values for each action
        :return:
        """
        x = self.fc1(x).to(device=self.device)
        x = self.relu(x).to(device=self.device)
        x = self.fc2(x).to(device=self.device)
        x = self.relu(x).to(device=self.device)
        x = self.fc3(x).to(device=self.device)
        x = self.softmax(x).to(device=self.device)

        return x

    def get_action(self, state):
        """

        :param state:
        :return:
        """
        state = torch.Tensor(state).to(device=self.device)
        state = torch.unsqueeze(state, 0)
        probs = self.forward(state)
        probs = torch.unsqueeze(probs, 0)
        action = probs.multinomial(num_samples=1)
        action = action[0]
        return action.item()

    def pi(self, state, action):
        """

        :param state:
        :param action:
        :return:
        """
        state = torch.Tensor([state]).to(device=self.device)
        probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        return probs[action]

    def update_weight(self, states, actions, rewards, optimizer):
        """

        :param states:
        :param actions:
        :param rewards:
        :return:
        """
        return_ = torch.Tensor([0]).to(device=self.device)
        # For each step of the episode t = T-1, ..., 0:
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            ## Acumulate total reward
            return_ = torch.Tensor([r_tt]).to(device=self.device) + self.discount_factor*return_
            loss = (-1.0)*return_*torch.log(self.pi(s_t, a_t))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
    else:
        torch.manual_seed(999)

    max_episodes = 10000
    max_timesteps = 150
    env = gym.make("CartPole-v1")
    agent = reinforce(env, discount_factor=0.95)
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    rewards_history = []
    timesteps_history = []

    for i_episode in range(max_episodes):

        state = env.reset()

        states = []
        actions = []
        rewards = [0]  # no reward at t = 0

        for timestep in range(max_timesteps):
            action = agent.get_action(state)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = env.step(action)

            rewards.append(reward)

            if done:
                rewards_history.append(np.sum(rewards))
                timesteps_history.append(timestep + 1)
                print("Episode {} finished after {} timesteps with total reward {}.".format(i_episode, timestep + 1,
                                                                                            np.sum(rewards)))
                break

        agent.update_weight(states, actions, rewards, optimizer)

    env.close()