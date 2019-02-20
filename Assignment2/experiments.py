
#%%
from tabular_part1 import SARSA
import numpy as np

#%%
env = SARSA('Taxi-v2',
            exploration="softmax",
            init_temperature=1000,
            decay=1-1e-2, 
            alpha=0.1, 
            gamma=1
)

#%%
RUNS = 10
SEGMENTS = 100
TRAIN_EPISODES = 10
TEST_EPISODES = 1
train_scores = []
test_scores = []
for run in range(RUNS):
    for segment in range(SEGMENTS):
        train_scores+=env.run(n_episodes=TRAIN_EPISODES, max_steps=200)
        test_scores+=env.run(n_episodes=TEST_EPISODES,render=False, max_steps=200,mode="test")
    env.reset()

#%%
train_scores = np.array(train_scores).reshape(RUNS,SEGMENTS,TRAIN_EPISODES)
test_scores = np.array(test_scores).reshape(RUNS,SEGMENTS,TEST_EPISODES)

#%%
import matplotlib.pyplot as plt

averaged_test_scores = test_scores.mean(axis=0)
averaged_train_scores = train_scores.mean(axis=0)

#plt.plot(averaged_test_scores)
plt.plot(averaged_train_scores)
