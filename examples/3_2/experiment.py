from stable_baselines3 import A2C

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load


# for i, lr in enumerate([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
for i, lr in enumerate([1e-3, 1e-4, 1e-5]):
    save_path = 'reports/stable_baselines3/' + str(i) + '/'
    for ii in range(4):
        env_num = ii + 12
        env_name = 'bandit_noise/' + str(env_num)
        env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=lr)
        model.learn(total_timesteps=100_000)

# DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
# BSUITE_SCORE = summary_analysis.bsuite_score(DF)
# print(BSUITE_SCORE)