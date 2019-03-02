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
                n_steps=1, 
                epsilon=0.5, 
                decay=1, 
                alpha=0.5, 
                gamma=.7,
                flatten_state = False):
           
        self.env = gym.make(env_name).env
        
        self.flatten_state = flatten_state
        self.is_discrete_state = False

        self.state_size = self.env.observation_space.shape
        if len(self.state_size) == 0:
            self.state_size = self.env.observation_space.n
            self.is_discrete_state = True
        elif flatten_state:
            self.state_size = np.array(self.state_size).sum()
        
        #TODO Handle discrete vs continious actions
        # for now it only supports discrete action space

        self.n_actions = self.env.action_space.n
        self.init_eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.n_steps = n_steps
        self.hist = History(n_steps+1)
        self.exploration = exploration
        self.init_temperature = init_temperature
        self.min_temperature = 1
        self.reset()

    def reset(self):
        self.wins = 0
        self.temperature = self.init_temperature
        self.eps = self.init_eps
        self.init_action_value_f()
        self.hist.reset()
    
    def _get_softmax_probas(self,obs):
        z = np.exp(self.get_action_state_value(obs)/self.temperature)
        return z / np.sum(z)
    
    def get_action(self, obs, greedy=False):
        if not greedy and self.mode == "train":
            if self.exploration == "eps-greedy":
                if random() < self.eps:
                    return self.env.action_space.sample()
            elif self.exploration == "softmax":
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
    
    def prepare(self,*vargs):
        out_array = []

        for arg in vargs[0]:
            out_array.append(np.array(arg))
        return out_array

    def run(self,n_episodes,max_steps,render = False,greedy=False,mode="train",reset=True,verbose=1):

        assert mode in ["train","test"],"Unknown Mode!"
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

                action = self.get_action(prev_observation, greedy=greedy)
                observation, reward, done, _ = self.env.step(action)
                self.hist.add((prev_observation, reward, action))

                prev_observation = np.array(observation)
                score+=reward
                
                if self.mode == "train" and self.hist.size() == self.n_steps+1:
                    self.evaluate()

                if done or (max_steps is not None and step == max_steps):
                    if verbose != 0:
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
        self.q_function =  defaultdict(lambda :rand(self.n_actions))