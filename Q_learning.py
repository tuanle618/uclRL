# -*- coding: utf-8 -*-
"""
@author: Tuan Le
@email: tuanle@hotmail.de
"""

from gridworld import GridworldEnv
from windy_gridworld import WindyGridworldEnv
import numpy as np

class Agent:    
    def __init__(self, env, discount_factor=1.0):
        """
        Initialized an Agent to solve a MDP via model free off policy Q-Learning
        Params:
            env [OpenAI env.]
                env.P represents the transition probabilities of the environment:
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done). Note that next_state is an integer
                    env.nS is a number of states in the environment. 
                    env.nA is a number of actions in the environment.
            discount_factor [float]: 
                Value between 0.0 and 1.0 for stressing importance of future rewards. 
                If discount_factor close to 0 the agent is short-sighted and only cares about immediate rewards
                If discount_factor close to 1 the agent is far-sighted and also cares about future rewards
                Default is 1.0
            epsilon [float]:
                error term for convergence. During the algorithms the difference of last iteration and current iterations
                state-value functions will be computed and compared to the epsilon value.
        """
        self.envDim = env.shape
        self.nS = env.nS
        self.nA = env.nA
        self.discount_factor = discount_factor
        self.env = env
    
    
    def epsilon_greedy_action_policy(self, Q, state, epsilon):
        """
        Epsilon greedy exploration:
            - All self.nA actions are tried with non-zero probability.
            - With probability 1-epsilon choose the greedy action
            - With probability epsilon choose an action at random
            functional: policy(a|s) = epsilon/nA  + 1 - epsilon, iif. a is argmax of Q(s,a)
                        else: epsilon/nA otherwise
        Returns:
            A numpy vector for action probability to sample given a state
        """
        
        A = np.ones(self.nA, dtype=float) * epsilon / self.nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
    
        return A
    
    def init_state(self):
        ## Initializes a random state but not terminal state
        non_terminal_states = np.arange(self.nS)
        non_terminal_states = [state for state in non_terminal_states if state not in self.env.terminalStates]
        init_state = int(np.random.choice(a=non_terminal_states, size=1))
        
        return init_state
    
    
    def Q_learning(self, num_iter=1000, epsilon=0.10, alpha=0.20, discount_factor=None):
        """
        This function computes the optimal action-values Q(s,a) using off-policy learning.
        The algorithm can be found on: http://incompleteideas.net/book/ebook/node65.html
        """
        
        if discount_factor is None:
            discount_factor = self.discount_factor
            
        ## Initialize state-action Q function 
        Q = np.zeros(shape=(self.nS, self.nA))
        ## Loop forever
        for _ in range(num_iter):
            ## Initialize state S
            state = self.init_state()
            ## Repeat (for each step of episode) until S is terminal
            while True:
                ## Choose epsilon greedy action
                action_probs = self.epsilon_greedy_action_policy(Q, state, epsilon)
                ## Sample action with the given action probabilities
                action = int(np.random.choice(a=self.nA, size=1, p=action_probs))
                ## Take action and observe transitions/step
                [(prob, next_state, reward, done)] = self.env.P[state][action]
                ## Do the state-action update.
                # First compute components
                optim_pi = np.max(a=Q[next_state])
                estimate = reward + discount_factor*optim_pi
                delta = estimate - Q[state, action]
                # Do the update
                Q[state, action] += alpha*delta
                if done:
                    break
                state = next_state
        
        return Q

def main():
    np.random.seed(26)
    env = GridworldEnv(shape=[4,4])
    agent = Agent(env)

    Q_learning_gridworld = agent.Q_learning(num_iter=1000,
                                            epsilon=0.10, alpha=0.20,
                                            discount_factor=0.30)    
    
    print("Optimal Q-Function after 1000 iterations:")
    print(np.round(Q_learning_gridworld,2))
    
    env2 = WindyGridworldEnv()
    agent2 = Agent(env2)
    Q_learning_windyworld = agent2.Q_learning(num_iter=1000,
                                            epsilon=0.10, alpha=0.20,
                                            discount_factor=0.30)   
    
    print("Optimal Q-Function after 1000 iterations:")
    print(np.round(Q_learning_windyworld,2))
    
if __name__ == "__main__":
    main()
    
                