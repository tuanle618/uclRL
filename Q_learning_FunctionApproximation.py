# -*- coding: utf-8 -*-
"""
@author: Tuan
@email: tuanle@hotmail.de
"""

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

## Models
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

## Create MountainCar environment from gym library
env = gym.envs.make("MountainCar-v0")

## Check out the environment
# Action space
print("The action space is:", env.action_space)
# State space
print("The state space is:", env.observation_space)

## Create a 'dataset' by producing an state-episode of k steps
k = 10000
observation_examples = np.array([env.observation_space.sample() for x in range(k)])

print("Shape of states is: {}".format(observation_examples.shape))
## first state: position
## second state: velocity
## For details have a look at source code: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

## Analyse the observation examples
print("Mean values for state 1 and 2:", np.average(observation_examples, axis=0))
print("Min value for state 1 and 2:", np.min(observation_examples, axis=0))
print("Max value for state 1 and 2:", np.max(observation_examples, axis=0))

## Feature preprocessing: Scale the observation samples
# Initialize the scaler 
scaler = sklearn.preprocessing.StandardScaler()
# Fit the scaler with dataset
scaler.fit(observation_examples)

# For the features we use radial basis functions sampler. 
# Note that each state consists of 2 points. For the feature representation 
# we include 4 Features [4 times a RBF sample method where per feature n_components=100 samples are drawn with MC]
# for detailed version have a look at: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space. 
# Each feature map rbf1 until rbf4 will be concatenated using the FeatureUnion after transforming
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

##### Extra:
## Scaled dataset
scaled_dataset = scaler.fit_transform(observation_examples)
## In order to get the new dataset (with concatenated feature maps)
feature_map = featurizer.fit_transform(scaled_dataset)
## Info about feature map
print("Shape of feature map:", feature_map.shape)

## Delete scaled_dataset and feature_map to save memory
del scaled_dataset, feature_map
import gc
gc.collect()

##### Build Estimator class:
class Estimator():
    """
    Value Function approximator v_hat(s, w).
    s=state
    w=weights of the model
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            #Linear model fitted by minimizing a regularized empirical loss with SGD
            model = SGDRegressor(loss="squared_loss", penalty="l2",
                                 alpha=0.0001, fit_intercept=True,
                                 learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # with partial fit one epoch of SGD will be performed on  given samples.
            # Here fit an the feature map of an initial random state with 0
            model.partial_fit(X=[self.featurize_state(env.reset())], y=[0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        ## Scale the new data point (state) with the Scaler instance
        scaled = scaler.transform([state]) #should have shape (1,2)
        ## Compute feature map of the new data point with the featurizer
        featurized = featurizer.transform(scaled) #should have shape (1, 400)
        ## Get only the vector instead as matrix
        featurized = featurized[0]
        return featurized
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        ## Compute features from state s
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        ## Compute features from state s
        features = self.featurize_state(s)
        ## Perform one SGD update for the given sample [features] and [y]
        self.models[a].partial_fit([features], [y])
        

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Computation for epsilon greedy probabilites:
        functional: policy(a|s) = epsilon/nA  + 1 - epsilon, iif. a is argmax of Q(s,a) [[greedy-exploitation]]
        else: epsilon/nA otherwise [[exploration]]
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes=1000, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = {"episode_length": np.zeros(num_episodes),
             "episode_rewards": np.zeros(num_episodes)}
    
    
    for i_episode in range(num_episodes):
        
        # The policy we're following
        # For each iteration we include a decay in the epsilon parameter
        policy = make_epsilon_greedy_policy(
            estimator = estimator,
            epsilon = epsilon * epsilon_decay**i_episode,
            nA = env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats["episode_rewards"][i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                ## Get epsilon greedy draw probabilities for actions
                action_probs = policy(state)
                ## Draw one action with epsilon greedy probabilities
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
    
            # Update statistics
            stats["episode_rewards"][i_episode] += reward
            stats["episode_length"][i_episode] = t
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            estimator.update(s=state, a=action, y=td_target)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state
    
    return stats



estimator = Estimator()
stats = q_learning(env, estimator, 100, epsilon=0.0)
