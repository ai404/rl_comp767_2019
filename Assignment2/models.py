from numpy.random import choice,random
import numpy as np
from base import BaseDiscrete

class SARSA(BaseDiscrete):
    def evaluate(self):
        hist = self.hist.get_data()
        start_estiamte_q = None
        start_action = None
        start_obs = None
        G = 0
        for i, state in enumerate(hist):
            observation, reward, action = state
            if i == 0:
                start_obs = observation.copy()
                start_action = action
                start_estiamte_q = self.get_action_state_value(start_obs)
                G += reward
            elif i<len(hist)-1:
                G += self.gamma**i* reward
            else:
                end_action = action
                end_estimate_q = self.get_action_state_value(observation)
                G += self.gamma**i *end_estimate_q[end_action]
        
        start_estiamte_q[start_action] += self.alpha * (G - start_estiamte_q[start_action])
        self.set_action_state_value(start_obs,start_estiamte_q)


class QLearning(BaseDiscrete):

    def __init__(self, 
                env_name, 
                exploration="eps-greedy",
                init_temperature=1000, 
                epsilon=0.5, 
                decay=1, 
                alpha=0.5, 
                gamma=.7,
                flatten_state = False):
        super(QLearning,self).__init__(env_name, 
                exploration="eps-greedy",
                init_temperature=1000, 
                n_steps=1, 
                epsilon=0.5, 
                decay=1, 
                alpha=0.5, 
                gamma=.7,
                flatten_state = False)

    def _get_next_q(self,obs):
        return max(self.get_action_state_value(obs))
    
    def evaluate(self):
        hist = self.hist.get_data()

        observation1, reward, start_action = hist[0]
        start_estiamte_q = self.get_action_state_value(observation1)

        observation2, _, _ = hist[1]        
        G = reward + self.gamma *self._get_next_q(observation2)

        start_estiamte_q[start_action] += self.alpha * (G - start_estiamte_q[start_action])
        self.set_action_state_value(observation1,start_estiamte_q)


class ExpectedSARSA(QLearning):
    def _get_next_q(self,obs):
        return np.sum(self._get_softmax_probas(obs,use_temp=False)*self.get_action_state_value(obs))

