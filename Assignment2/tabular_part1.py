import gym
import numpy as np
from keras import layers, models
import random
import matplotlib.pyplot as plt

from collections import defaultdict
class History:

    def __init__(self,max_len):
        self.list = []
        self.max_len = max_len
    
    def reset(self):
        self.list = []
    def add(self,obs):
        if len(self.list) == self.max_len:
            self.list.pop(0)
        self.list.append(obs)
    
    def get_data(self):
        return self.list

    def size(self):
        return len(self.list)

class BaseFramework:

    def __init__(self, 
                env_name, 
                exploration="eps-greedy",
                init_temperature=1000, 
                lambda_=1, 
                epsilon=0.5, 
                decay=.95, 
                alpha=0.5, 
                gamma=.7,
                flatten_state = False):
           
        self.env = gym.make(env_name).env
        
        self.flatten_state = flatten_state
        self.is_indexed = False

        self.state_size = self.env.observation_space.shape
        if len(self.state_size) == 0:
            self.state_size = self.env.observation_space.n
            self.is_indexed = True
        else:
            if flatten_state:
                self.state_size = np.array(self.state_size).sum()
            else:
                self.state_size = self.state_size
        
        self.n_actions = self.env.action_space.n
        self.init_action_value_f()
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.wins = 0
        self.lambda_ = lambda_
        self.hist = History(lambda_+1)
        self.exploration = exploration

        self.init_temperature = init_temperature
        self.temperature = init_temperature
        self.min_temperature = 1
        
    def reset(self):
        self.init_action_value_f()
    
    def get_action(self, obs, soft=True):
        if self.mode == "train":
            if self.exploration == "eps-greedy":
                if soft and random.random() < self.eps:
                    return self.env.action_space.sample()
                return np.argmax(self.get_action_state_value(obs))
            elif self.exploration == "softmax":
                z = np.exp(self.get_action_state_value(obs)/self.temperature)
                p = z / np.sum(z)
                return np.random.choice(self.n_actions, 1, p=p)[0]
            else:
                raise Exception("Unknown Exploration Method!")
        elif self.mode == "test":
            return np.argmax(self.get_action_state_value(obs))
        else:
            raise Exception("Unknown Mode!")
    
    def update_lr(self):
        if self.mode == "train":
            if self.exploration == "eps-greedy":
                self.eps *= self.decay
            elif self.exploration == "softmax" and self.temperature*self.decay>self.min_temperature:
                self.temperature *= self.decay
    
    def prepare(self,*vargs):
        out_array = []

        for arg in vargs[0]:
            out_array.append(np.array(arg))
        return out_array

    def run(self,n_episodes,max_steps,render = False,mode="train",reset=True):
        self.mode = mode
        self.env._max_episode_steps = max_steps

        scores = []
        best_score = 0
        for i_episode in range(n_episodes):
            if reset:
                self.hist.reset()
            prev_observation = np.array(self.env.reset())

            step = 0
            score = 0
            while True:
                if render:
                    self.env.render()

                action = self.get_action(prev_observation)
                observation, reward, done, _ = self.env.step(action)
                observation = np.array(observation)
                self.hist.add((prev_observation, reward, action))

                prev_observation = observation
                score+=reward
                
                if self.mode == "train" and self.hist.size() == self.lambda_+1:
                    self.evaluate()

                if done or (max_steps is not None and step == max_steps):
                    print(f"Episode {i_episode} - Max Score {best_score} - Score {score} - Step {step}")
                    print("Episode finished after {} timesteps".format(step + 1))
                    break
                step += 1
            self.update_lr()
            
            
            if best_score< score:
                best_score = score
            scores.append(score)
        return scores

    def get_action_state_value(self,obs):
        return NotImplementedError()
    
    def set_action_state_value(self,obs,value):
        return NotImplementedError()
    
    def init_action_value_f(self):
        return NotImplementedError()

    def evaluate(self):
        return NotImplementedError()

class BaseDiscrete(BaseFramework):
    def get_action_state_value(self,obs):
        return self.q_function[hash(str(obs))]
    
    def set_action_state_value(self,obs,value):
        self.q_function[hash(str(obs))] = value
    
    def init_action_value_f(self):
        self.q_function =  defaultdict(lambda :np.random.rand(self.n_actions))

class SARSA(BaseDiscrete):
    def evaluate(self):
        hist = self.hist.get_data()
        start_estiamte_q = None
        start_action = None
        start_obs = None
        G = 0
        for i, state in enumerate(self.hist.get_data()):
            observation, reward, action = state
            if i == 0:
                start_obs = observation.copy()
                start_action = action
                start_estiamte_q = self.get_action_state_value(start_obs)
                G += self.gamma**i* reward
            elif i<len(hist)-1:
                G += self.gamma**i* reward
            else:
                end_action = action
                end_estimate_q = self.get_action_state_value(observation)
                G += self.gamma**i *end_estimate_q[end_action]
        
        start_estiamte_q[start_action] += self.alpha * (G - start_estiamte_q[start_action])
        self.set_action_state_value(start_obs,start_estiamte_q)

if __name__ == "__main__":
    env = SARSA('Taxi-v2',
                    exploration="softmax",
                    init_temperature=1000,
                    decay=1-1e-2, 
                    alpha=0.1, 
                    gamma=1
    )
    
    env.run(n_episodes=1000, max_steps=200)
