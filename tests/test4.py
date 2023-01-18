import bsuite
import dm_env

from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis


SAVE_PATH_DQN = 'reports/bsuite/rllib'
raw_env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_DQN, overwrite=True)
env = gym_wrapper.GymFromDMEnv(raw_env)

for episode in range(env.bsuite_num_episodes):
    state = env.reset()
    done = False
    while not done:
      state, reward, done, info = env.step(0)


from bsuite.logging import csv_load
DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)