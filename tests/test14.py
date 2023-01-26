from stable_baselines3 import A2C
from stable_baselines3 import DQN

import numpy as np

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

SAVE_PATH_DQN = 'reports'
env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record('memory_size/16', save_path=SAVE_PATH_DQN, overwrite=True))
model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=29_000)


DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)