from stable_baselines3 import A2C
from stable_baselines3 import DQN

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

SAVE_PATH_DQN = 'reports/bsuite/rllib'
env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record('cartpole/0', save_path=SAVE_PATH_DQN, overwrite=True))
print(env.action_space, env.observation_space)
model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=59_999)

DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)