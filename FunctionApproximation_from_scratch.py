# -*- coding: utf-8 -*-
"""
@author: Tuan
"""

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.kernel_approximation import RBFSampler

## Create MountainCar environment from gym library
env = gym.envs.make("MountainCar-v0")

## Check out the environment
# Action space
print("The action space is:", env.action_space)
print("There are {} number of actions".format(env.action_space.n+1))
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
# 10000 x 400

## Delete scaled_dataset and feature_map to save memory
#del scaled_dataset, feature_map
#import gc
#gc.collect()

class Estimator():
    """
    
    """
    def __init__(self, n_features, featurizer, scaler, lr=0.05):
        ## for each action create a model
        self.W = [np.random.randn(n_features) for _ in range(env.action_space.n)]
        
        self.featurizer = featurizer
        self.scaler = scaler
        self.lr = lr
    
    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        features = self.featurizer.transform(scaled)
        return features[0]

    
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
        features = self.featurize_state(s)
        if not a:
            prediction = []
            for w in self.W:
                model_prediction = np.matmul(w, features)
                prediction.append(model_prediction)
            return np.array(prediction)
        else:
            return np.array(np.matmul(self.W[a], features))
        
    def compute_loss(self, v, v_pred):
        loss = 0.5*(v-v_pred)**2
        loss = np.sum(loss)
        return loss
    
    def l2_gradient(self, v, v_pred, features):
        """
        negative gradient from the objective function. 
        note the prediction v_pred is computed via a linear model
        """
        grad = (v-v_pred)*features
        return -grad
    
    def update(self, s, a, v_pred):
        """
        Performs a gradient update on one sample
        """
        ## Compute the current prediction
        v = self.predict(s, a)
        ## Note v_pred is the td-target for next iteration
        grad = self.l2_gradient(v, v_pred)
        ## Perform update
        self.W[a] += self.lr*grad
        
        
estimator = Estimator(n_features=400, featurizer=featurizer, scaler=scaler)

