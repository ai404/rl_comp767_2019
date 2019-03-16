from collections import defaultdict
import numpy as np
from numpy.random import choice,random,rand
import gym

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
                epsilon=0.5, 
                decay=1, 
                alpha=0.5, 
                gamma=.7):
        
        self.env = gym.make(env_name)

        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        
        self.init_eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.exploration = exploration
        self.init_temperature = init_temperature
        self.min_temperature = 1
        self.reset()

    def is_off(self):
        return True
    
    def reset(self):
        self.temperature = self.init_temperature
        self.eps = self.init_eps
        self.init_action_value_f()
    
    def _get_softmax_probas(self,obs,use_temp=True):
        mx = np.max(self.get_action_state_value(obs))
        z = self.get_action_state_value(obs)- mx
        if use_temp:
            z = z/self.temperature
        
        return np.exp(z) / np.sum(np.exp(z))
    
    def get_action(self, obs):
        if self.mode == "train":
            if self.exploration == "eps-greedy":
                if random() < self.eps:
                    return self.env.action_space.sample()
            elif self.exploration == "softmax":
                #return exp.get_action(self.q_function,obs)
                p = self._get_softmax_probas(obs)
                return choice(self.n_actions, 1, p=p)[0]
            else:
                raise Exception("Unknown Exploration Method!")
        return np.argmax(self.get_action_state_value(obs))
    
    def update_lr(self):
        if self.mode == "train":
            if self.exploration == "eps-greedy":
                self.eps *= self.decay
            elif self.exploration == "softmax" and self.temperature*self.decay>self.min_temperature:
                self.temperature *= self.decay
    
    def run(self,n_episodes,render = False,mode="train",verbose=1):
        assert mode in ["train","test"],"Unknown Mode!"
        self.mode = mode

        scores = []
        best_score = 0
        for i_episode in range(n_episodes):
            step = 0
            score = 0
            done = False
            
            s = self.env.reset()
            a = self.get_action(s)
            
            while not done:
                if render:
                    self.env.render()

                s_prime, reward, done, _ = self.env.step(a)
                
                a_prime = None if self.is_off() else self.get_action(s_prime)
                
                if self.mode == "train":
                    self.evaluate((s,a,reward,s_prime,a_prime))

                a = self.get_action(s_prime) if self.is_off() else a_prime
                s = s_prime
                
                score+=reward
                step += 1
            
            if verbose != 0:
                print(f"Episode {i_episode} - Max Score {best_score} - Score {score} - Step {step}")
                print("Episode finished after {} timesteps".format(step + 1))
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

    def evaluate(self,params):
        return NotImplementedError()

class BaseDiscrete(BaseFramework):
    def get_action_state_value(self,obs):
        return self.q_function[obs]
    
    def set_action_state_value(self,obs,value):
        self.q_function[obs] = value
    
    def init_action_value_f(self,random=True):
        
        self.q_function = {s: rand(self.n_actions) for s in range(self.n_states)}
        #self.q_function = {s: np.zeros(self.n_actions) for s in range(self.n_states)}
        
        for location_id, (loc_row,loc_col) in enumerate(self.env.unwrapped.locs):
            state_id = self.env.unwrapped.encode(loc_row, loc_col, location_id, location_id)
            self.q_function[state_id] = np.zeros(self.n_actions)
