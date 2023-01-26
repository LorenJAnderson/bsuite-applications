from stable_baselines3 import A2C
from stable_baselines3 import DQN

import numpy as np

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

SAVE_PATH_DQN = 'reports/reports/bsuite/rllib'
env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record('memory_size/16', save_path=SAVE_PATH_DQN, overwrite=True))
done = False
env_counter = 0
epi_counter = 0
while epi_counter < 10_000:
    env_counter += 1
    obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
    if done:
        print(env_counter, epi_counter)
        env_counter = 0
        epi_counter += 1
print(epi_counter)


DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)