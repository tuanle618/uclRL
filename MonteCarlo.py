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
        
    def generate_episode(self, policy, state=None):
        # Initialize random first non-terminal state if not inserted
        if state is None:
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
            if policy.shape == (self.nS, self.nA): # If policy matrix is given
                action = np.random.choice(self.nA, p=policy[state])
            else: # If vector of probabilities to draw an action is given
                action = np.random.choice(self.nA, p=policy)
            
            # from the current state get transition probability information from environment
            [(prob, next_state, reward, done)] = self.env.P[state][action]
            # append episode with step-dictionary
            episode.append({"state":state, "action":action, "reward":reward, "next_state":next_state})
            # for the while loop define state to be the next state
            state = next_state
            
            
    def monte_carlo_prediction(self, policy=None, num_iter=10000, discount_factor=None, first_visit=True):
        """
        
        """
        if policy is None:
            ### Uniformly distributed action matrix given states. Shape: nS x nA 
            policy = self.env.isap
            
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
            episode = self.generate_episode(policy)
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
    
    def epsilon_greedy_action_policy(self, Q, state, epsilon=0.1):
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
    
    def monte_carlo_control(self, policy=None, num_iter=10000,
                            discount_factor=None, epsilon_method=True,
                            epsilon=0.1, on_policy=True):
        """
        Monte-Carlo ES [with Exploring Starts] Control.
        Algorithm taken from: http://incompleteideas.net/book/ebook/node53.html
        
        Monte-Carlo epsilon greedy control:
        http://incompleteideas.net/book/ebook/node54.html
        """
        
        if discount_factor is None:
            discount_factor = self.discount_factor
            
        ## Initialize Q-Table (action value function)
        Q = np.zeros(shape=(self.nS, self.nA))
        ## Initialize policy arbitrarily to be uniformly distributed
        policy = self.env.isap
        ## Initialize empty returns list for state-action
        returns = {(i, j):list() for i in range(self.nS) for j in range(self.nA)}
        

        ## Loop forever
        for _ in range(num_iter):
            ## Generate an episode:
            # First sample a init state
            state = np.random.choice(a=np.arange(start=1, stop=self.env.terminalStates[1]-1, step=1))
            if epsilon_method:
                control_policy = self.epsilon_greedy_action_policy(Q, state)
            else: # take uniformly distributed policy. "With exploring starts Algorithm"
                control_policy = policy
                
            ## Generate an episode
            episode = self.generate_episode(policy=control_policy, state=state)
            
            G = 0
            ##For each pair s,a appearing in the episode
            for t, step in enumerate(episode):
                s, a = step["state"], step["action"]
                ## Compute total return
                G = discount_factor*G + step["reward"]
                ## Append to State-Action return list
                returns[s, a].append(G)
                ## Compute estimate of State-Action function Q
                Q[s,a] = np.mean(a=returns[s,a])
            ## Algo 5.3 or 5.4 from http://incompleteideas.net/book/ebook/node53.html
            ## http://incompleteideas.net/book/ebook/node54.html
            ## Get unique states in episode
            unique_states = set([step["state"] for step in episode])
            for state in unique_states:
                optim_action = np.argmax(Q[state])
                for action_iterator in range(len(policy[state])):
                    if not epsilon_method:
                        policy[state, action_iterator] = 1 if action_iterator == optim_action else 0
                    else:
                        policy[state, action_iterator] = 1 - epsilon + epsilon/len(policy[state]) if action_iterator == optim_action else epsilon/len(policy[state])

                                    
        return Q, policy
        

    

def main():
    np.random.seed(26)
    env = GridworldEnv(shape=[4,4])
    agent = Agent(env)
    ## Sample one episode
    episode = agent.generate_episode(policy=agent.env.isap)
    print("Example: Sample episode for Monte Carlo:")
    print(episode)
    ## Do First-Visit-Monte-Carlo Prediction
    first_visit_MC_value_fnc = agent.monte_carlo_prediction(first_visit=True,
                                                 discount_factor=1.0,
                                                 num_iter=1000)
    first_visit_MC_value_fnc = np.round(first_visit_MC_value_fnc, 2)
    print("Result first-visit Monte Carlo:")
    print(first_visit_MC_value_fnc.reshape((env.shape)))

    ## Do Every-Visit Monte-Carlo Prediction
    every_visit_MC_value_fnc = agent.monte_carlo_prediction(first_visit=False,
                                                 discount_factor=1.0,
                                                 num_iter=1000)
    every_visit_MC_value_fnc = np.round(every_visit_MC_value_fnc, 2)
    print("Result every-visit Monte Carlo:")
    print(every_visit_MC_value_fnc.reshape((env.shape)))
    
    ## Do Every-Visit Monte-Carlo Control with Exploring Starts (no epsilon greedy method)
    Q_control_no_epsilon, policy_control_no_epsilon = agent.monte_carlo_control(policy=None, num_iter=200,
                                                                                discount_factor=None, epsilon_method=False,
                                                                                epsilon=0.1, on_policy=True)
    Q_control_no_epsilon = np.round(Q_control_no_epsilon, 2)
    policy_control_no_epsilon = np.round(policy_control_no_epsilon, 2)
    print("Result every-visit Monte Carlo Control Q-Function:")
    print(Q_control_no_epsilon)
    print("Result every-visit Monte Carlo Control optimal policy:")
    print(policy_control_no_epsilon)
    
    ## Do Every-Visit Monte-Carlo Control epsilon greedy method:
    Q_control_eps_greedy, policy_control_eps_greedy = agent.monte_carlo_control(policy=None, num_iter=500,
                                                                                discount_factor=None, epsilon_method=True,
                                                                                epsilon=0.1, on_policy=True)
    Q_control_eps_greedy = np.round(Q_control_eps_greedy, 2)
    policy_control_eps_greedy = np.round(policy_control_eps_greedy, 2)
    print("Result every-visit Monte Carlo Control Q-Function:")
    print(Q_control_eps_greedy)
    print("Result every-visit Monte Carlo Control optimal policy:")
    print(policy_control_eps_greedy)
    
if __name__ == "__main__":
    main()
    

