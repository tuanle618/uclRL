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
    
    def generate_episode(self, policy):
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
            action = np.random.choice(self.nA, p=policy[state])
            # from the current state get transition probability information from environment
            [(prob, next_state, reward, done)] = self.env.P[state][action]
            # append episode with step-dictionary
            episode.append({"state":state, "action":action, "reward":reward, "next_state":next_state})
            # for the while loop define state to be the next state
            state = next_state
        
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
    
    @staticmethod
    def compute_lambda_return(episode, _lambda, discount_factor, v_fnc):
        ## Initialize the step return vector
        G_n = np.zeros(len(episode))
        ## for each step k compute the k-step return
        for k, step in enumerate(episode):
            G_n[k] = step["reward"] + discount_factor*v_fnc[step["next_state"]]
        
        ## Get lambda weights vector to enhance numpy vectorization
        lambda_weights = np.array([_lambda**(n-1) for n in np.arange(start=1, stop=len(episode)+1)])
        G_lambda = (1-_lambda)* np.matmul(G_n, lambda_weights)
        
        return G_lambda
            
        
        
    def td_lambda(self, policy=None, num_iter=10000,
                  _lambda=0.5, discount_factor=None, alpha=0.1,
                  backward=True):
        """
        Algorithm:
        forward: http://incompleteideas.net/book/ebook/node74.html
        backward: http://incompleteideas.net/book/ebook/node75.html
        """
        if policy is None:
            ### Uniformly distributed action matrix given states. Shape: nS x nAS
            policy = self.env.isap
        
        if discount_factor is None:
            discount_factor = self.discount_factor
            
        ## Initialize value function
        v_fnc = np.zeros(self.nS)
        
        ## Do TD(Lambda):
        if backward:
            ## Initialize eligibility trace 
            e_trace = np.zeros(self.nS)
            ## Repeat for each episode:
            for _ in range(num_iter):
                ## Initialize a random non-terminal state
                state = self.init_state()
                ## Repeat for each step in episode:
                while True:
                    ## Sample action given a policy
                    action = self.get_action(state, policy)
                    ## Get transitions
                    [(prob, next_state, reward, done)] = self.env.P[state][action]
                    ## check if the next_state is terminal. If so, break and iterate again for new episode
                    if done:
                        break
                    ## Compute the TD(0) error [get current error: one step ahead]
                    td_error = reward + discount_factor*v_fnc[next_state] - v_fnc[state]
                    ## Increment eligibility trace for current state [frequency]
                    e_trace[state] += 1
                    
                    ## For all states enhance the very past. this can be vectorized
                    #for s in range(self.nS):
                    #    v_fnc[s] += alpha*td_error*e_trace[s]
                    #    e_trace[s] *= _lambda*discount_factor
                    
                    ## Vectorized form of updating the value function and modifying the eligibility trace wrt to recency
                    v_fnc += alpha*td_error*e_trace
                    e_trace *=  _lambda*discount_factor
                    
                    state = next_state
        else: #forward TD(lambda)
            ## Loop forever
            for _ in range(num_iter):
                ## sample a full trajectory / episode
                episode = self.generate_episode(policy)
                ## compute the lambda-return
                G_lambda = self.compute_lambda_return(episode, _lambda, discount_factor, v_fnc)

                ## Iterate through steps and apply update
                for _, step in enumerate(episode):
                    v_fnc[step["state"]] += alpha*(G_lambda - v_fnc[step["state"]])
                
            
        return v_fnc
                
def main():
    np.random.seed(26)
    env = GridworldEnv(shape=[4,4])
    agent = Agent(env)
    td_zero_res = agent.td_zero(discount_factor=0.25, alpha=0.10)    
    print("Result TD(0) Value function:")
    print(td_zero_res.reshape((env.shape)))
    
    td_lambda_res_bw = agent.td_lambda(discount_factor=0.25, alpha=0.1, _lambda=0.5, backward=True)
    print("Result backward TD(Lambda=0.5) Value function:")
    print(td_lambda_res_bw.reshape((env.shape)))
    
    td_lambda_res_fw = agent.td_lambda(discount_factor=0.25, alpha=0.1, _lambda=0.5, backward=False)
    print("Result forward TD(Lambda=0.5) Value function:")
    print(td_lambda_res_fw.reshape((env.shape)))
    
if __name__ == "__main__":
    main()
    