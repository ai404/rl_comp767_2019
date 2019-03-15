from numpy.random import choice,random
import numpy as np
from base import BaseDiscrete

class SARSA(BaseDiscrete):
    def evaluate(self,params):
        s,a,reward,s_prime,a_prime = params

        current_q = self.get_action_state_value(s)
        next_q = self.get_action_state_value(s_prime)
       
        G = reward + self.gamma * next_q[a_prime]

        current_q[a] = current_q[a] + self.alpha * (G - current_q[a])
        self.set_action_state_value(s,current_q)

class QLearning(BaseDiscrete):

    def _get_next_q(self,s):
        return max(self.get_action_state_value(s))
    
    def evaluate(self,params):
        s,a,reward,s_prime,_ = params
        
        current_q = self.get_action_state_value(s)
        G = reward + self.gamma * self._get_next_q(s_prime)

        current_q[a] = current_q[a] + self.alpha * (G - current_q[a])
        self.set_action_state_value(s,current_q)

class ExpectedSARSA(QLearning):
    def _get_next_q(self,obs):
        return np.sum(self._get_softmax_probas(obs)*self.get_action_state_value(obs))

