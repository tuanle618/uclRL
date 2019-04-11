# -*- coding: utf-8 -*-
"""
@author: Tuan Le
@email: tuanle@hotmail.de

"""

from gridworld import GridworldEnv
import numpy as np

class Agent:    
    def __init__(self, env, discount_factor=1.0, epsilon=1e-6):
        """
        Initialized an Agent to solve a MDP via model free Temporal Difference Learning method.
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
        self.epsilon = epsilon
        self.vFnc= np.zeros(self.nS)
        
    def init_state(self):
        ## Initializes a random state but not terminal state
        init_state = np.random.choice(a=np.arange(start=1, stop=self.env.terminalStates[1]-1, step=1))
        return init_state
        
    def get_action(self, state, policy):
        ## Samples a random action. This is uniformly distributed. 0=Up 1=Right, 2=Down, 3=Left
        # Note this can be modified by a specific policy. In this case all actions are uniformly distributed
        action = np.random.choice(self.nA, p=policy[state])
        return action
        
    def td_zero(self, policy=None, num_iter=10000, alpha=0.1, discount_factor=None):
        
        if policy is None:
            ### Uniformly distributed action matrix given states. Shape: nS x nA 
            policy = self.env.isap
        
        ## Initialize a state value function V arbitrarily
        v_fnc = np.zeros(self.nS)
        if discount_factor is None:
            discount_factor = self.discount_factor
        
        ## Loop forever / for each episode:
        for _ in range(num_iter):
            ## Init a random state
            state = self.init_state()
            ## Loop for each state in episode:
            while True:
                ## Select a random action given a state
                action = self.get_action(state, policy)
                ## Get transitions: Take action and observe rewards and next state
                [(prob, next_state, reward, done)] = self.env.P[state][action]
                if done:
                    break
                ## Modify the value function by estimating the new value for state
                v_fnc[state] += alpha*(reward + discount_factor*v_fnc[next_state] - v_fnc[state])
                state = next_state
            
        return v_fnc

    

def main():
    np.random.seed(26)
    env = GridworldEnv(shape=[4,4])
    agent = Agent(env)
    td_zero_res = agent.td_zero(discount_factor=0.25, alpha=0.10)    
    print("Result TD(0) Value function:")
    print(td_zero_res.reshape((env.shape)))
    
if __name__ == "__main__":
    main()
    


