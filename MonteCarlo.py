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
        Initialized an Agent to solve a MDP via model free Monte-Carlo Policy Evaluation method.
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
        
    def generate_episode(self):
        # Initialize random first non-terminal state
        state = np.random.choice(a=np.arange(start=1, stop=self.env.terminalStates[1]-1, step=1))
        # Initialize episode list of sequences: S_i,A_i,R_i+1
        episode = []
        # Initialize done variable to be false
        done = False
        # write sequence into episode list until final state will be reached
        while True:
            if done:
                return episode
            # select a random action given the current state [note this is uniformly distributed]
            action = np.random.choice(self.nA, p=self.env.isap[state])
            # from the current state get transition probability information from environment
            [(prob, next_state, reward, done)] = self.env.P[state][action]
            # append episode with step-dictionary
            episode.append({"state":state, "action":action, "reward":reward, "next_state":next_state})
            # for the while loop define state to be the next state
            state = next_state
            
            
    def monte_carlo(self, num_iter=10000, discount_factor=None, first_visit=True):
        """
        
        """
        ## Initialize a state value function V arbitrarily
        v_fnc = np.zeros(self.nS)
        ## Initialize return lists for each state as dictionary
        returns = {i :list() for i in range(self.nS)}
        if discount_factor is None:
            discount_factor = self.discount_factor
        
        ## Loop forever (to get many monte carlo samples)
        for _ in range(num_iter):
            ## Initialize return G (total discounted reward) to be 0
            G = 0
            ## Initialize an episode
            episode = self.generate_episode()
            ## Reverse episode to begin with last time step
            reversed_episode = episode[::-1]
            for t, step in enumerate(reversed_episode):
                G = discount_factor*G + step["reward"]
                if first_visit:
                    # Make sure its first-visit:
                    if step["state"] not in [x["state"] for x in episode[:t]]:                   
                        returns[step["state"]].append(G)
                        new_value = np.average(returns[step["state"]])
                        v_fnc[step["state"]] = new_value
                else: #every-visit
                    returns[step["state"]].append(G)
                    new_value = np.average(returns[step["state"]])
                    v_fnc[step["state"]] = new_value
                                
        return v_fnc

        

    

def main():
    np.random.seed(26)
    env = GridworldEnv(shape=[4,4])
    agent = Agent(env)
    ## Sample one episode
    episode = agent.generate_episode()
    print("Example: Sample episode for Monte Carlo:")
    print(episode)
    ## Do First-Visit-Monte-Carlo
    first_visit_MC_value_fnc = agent.monte_carlo(first_visit=True,
                                                 discount_factor=1.0,
                                                 num_iter=10000)
    first_visit_MC_value_fnc = np.round(first_visit_MC_value_fnc, 2)
    print("Result first-visit Monte Carlo:")
    print(first_visit_MC_value_fnc.reshape((env.shape)))

    ## Do Every-Visit Monte-Carlo
    every_visit_MC_value_fnc = agent.monte_carlo(first_visit=False,
                                                 discount_factor=1.0,
                                                 num_iter=10000)
    every_visit_MC_value_fnc = np.round(every_visit_MC_value_fnc, 2)
    print("Result every-visit Monte Carlo:")
    print(every_visit_MC_value_fnc.reshape((env.shape)))
    
if __name__ == "__main__":
    main()
    

