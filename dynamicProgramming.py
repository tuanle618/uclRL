# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:10:13 2019

@author: Tuan

"""

from gridworld import GridworldEnv
import numpy as np

class Agent:    
    def __init__(self, env, discount_factor=1.0, epsilon=1e-6):
        """
        Initialized an Agent to solve a MDP via dynamic programming.
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
        
    def policy_evaluation(self, policy):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.
        
        Params:
            policy: [S, A] shaped matrix representing the policy.
            For a state (row) there is a probability vector of actions.
            
        Returns:
            Vector of length env.nS representing the value function.
        """
        
        # Start with a random (all 0) value function
        V_old = self.vFnc
    
        while True:
            #new value function
            V_new = np.zeros(self.nS)
            #stopping condition
            delta = 0
    
            #loop over state space
            for s in range(self.nS):
    
                #To accumelate bellman expectation eqn
                v_fnc = 0
                #get probability distribution over actions. This is a row vector of dimension 1xnA
                action_probs = policy[s]
    
                #loop over possible actions
                for a in range(self.nA):
    
                    #get transitions from environment
                    [(prob, next_state, reward, done)] = self.env.P[s][a]
                    #apply bellman expectation equation
                    v_fnc += action_probs[a] * (reward + self.discount_factor * V_old[next_state])
    
                #get the biggest difference over state space
                delta = max(delta, abs(v_fnc - V_old[s]))
    
                #update state-value
                V_new[s] = v_fnc
    
            #the new value function
            self.vFnc = V_old = V_new
    
            #if true value function checking via stoppping criterion
            if(delta < self.epsilon):
                break
    
        return np.array(V_old)
    
    def one_step_lookahead(self, s, value_fnc):
        """
        For a given policy and current state s compute the q-function state-action values for each action
        """
        actions = np.zeros(self.nA)

        for a in range(self.nA):
            [(prob, next_state, reward, done)] = self.env.P[s][a]
            # Compute q value [•state-action fnc]
            actions[a] = prob * (reward + self.discount_factor * value_fnc[next_state])
            
        return actions
    
    def policy_improvement(self, policy):
        """
        Policy Improvement Algorithm. 
        Iteratively evaluates and improves a policy until an optimal policy is found.
    
        Returns:
            A tuple (policy, V). 
            policy is the optimal policy, a matrix of shape [S, A] where each state s
            contains a valid probability distribution over actions.
            V is the value function for the optimal policy.
        """
        
        ## Do policy improvement:
        
        while True:
            
            # Evaluate the current policy using self.policy_evaluation function. 
            # This leads to the result of the policy_evaluation matrix:
            ## [[  0. -14. -20. -22.]
            ##  [-14. -18. -20. -20.]
            ##  [-20. -20. -18. -14.]
            ##  [-22. -20. -14.   0.]]
            value_fnc = self.policy_evaluation(policy=policy)
            
            # Initialize help variable for stopping criterion
            policy_stable = True
            
            # Loop over state space to get best actions:
            for s in range(self.nS):
                # Perform one-step-look-ahead to get q function values
                action_values = self.one_step_lookahead(s=s, value_fnc=value_fnc)
                # Maximize over the best action wrt to the highest amount of reward [A]:
                best_action = np.argmax(action_values)
                # Choose best action on the CURRENT policy [B]:
                chosen_action = np.argmax(policy[s])
                
                # Check Bellman optimality equation whether best_action equals the chosen_action
                if(best_action != chosen_action):
                    policy_stable = False
                    
                # Act greedily and select policy w.r.t to the value function, meaning going with 100% into a certain direction
                policy[s] = np.eye(self.nA)[best_action]
                
            ## Check Bellman optimality equation after all states were seen:
            if policy_stable:
                return policy, value_fnc

                    
def main(shape=[4,4]):
    env = GridworldEnv(shape=shape)
    agent = Agent(env=env)
    
    ## Policy Evaluation
    print("Do Policy Evaluation...:")
    policy = env.isap
    print("Initial value function:")
    print(agent.vFnc.reshape((env.shape)))
    print("")
    print("Random Policy uniformly distributed")
    print(policy)
    print("")
    optimal_value_fnc = agent.policy_evaluation(policy)
    optimal_value_fnc = np.round(optimal_value_fnc)
    print("Optimal value function:")
    print(optimal_value_fnc.reshape((env.shape)))
    print("")
    
    ## Policy Improvement
    print("Do Policy Improvement...:")
    print("Start with Random Policy uniformly distributed")
    ## Initialize random policy for each state and action from environment
    policy = env.isap
    print(policy)
    print("")
    policy_improvement_res, value_fnc_optimal = agent.policy_improvement(policy)
    print("Optimal Policy Probability Distribution:")
    print(policy_improvement_res)
    print("")
    
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy_improvement_res, axis=1), env.shape))
    print("")
    
    print("Value Function:")
    print(value_fnc_optimal)
    print("")
    
    print("Reshaped Grid Value Function:")
    print(value_fnc_optimal.reshape(env.shape))
    print("")
    

if __name__ == "__main__":
    main()
