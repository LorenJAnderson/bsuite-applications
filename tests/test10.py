from stable_baselines3 import A2C
from stable_baselines3 import DQN

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

SAVE_PATH_DQN = 'reports/bsuite/rllib'
# env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record('memory_len/2', save_path=SAVE_PATH_DQN, overwrite=True))
from custom_envs.deep_sea_test.environment import DeepSea
env = DeepSea(3)
model = A2C("MlpPolicy", env, verbose=1, ent_coef=0.01, tensorboard_log="./tensorboard/")
# model = DQN("MlpPolicy", env, verbose=1, exploration_initial_eps=1.0, exploration_final_eps=0.0, exploration_fraction=0.5)
model.learn(total_timesteps=100_000)

# DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
# BSUITE_SCORE = summary_analysis.bsuite_score(DF)
# print(BSUITE_SCORE)