import bsuite
from bsuite import sweep
import gym
from bsuite.utils import gym_wrapper

print(sweep.SWEEP)
print(sweep.BANDIT_NOISE)

for bsuite_id in sweep.BANDIT_NOISE:
  env = bsuite.load_from_id(bsuite_id)
  print('bsuite_id={}, settings={}, num_episodes={}'
        .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

raw_env = bsuite.load_from_id(bsuite_id='memory_len/0')
env = gym_wrapper.GymFromDMEnv(raw_env)
isinstance(env, gym.Env)

import numpy as np
SAVE_PATH_RAND = 'reports/bsuite/rand'
env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_RAND, overwrite=True)


for episode in range(env.bsuite_num_episodes):
  timestep = env.reset()
  while not timestep.last():
    action = np.random.choice(env.action_spec().num_values)
    timestep = env.step(action)

# def run_random_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
#   """Evaluates a random agent experiment on a single bsuite_id."""
#   env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
#   for episode in range(env.bsuite_num_episodes):
#     timestep = env.reset()
#     while not timestep.last():
#       action = np.random.choice(env.action_spec().num_values)
#       timestep = env.step(action)
#   return
#
# for bsuite_id in sweep.BANDIT_NOISE:
#   run_random_agent(bsuite_id)


from bsuite.logging import csv_load
DF, _ = csv_load.load_bsuite('reports/bsuite/full')
print(DF.size)
from bsuite.experiments import summary_analysis
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)
print(type(BSUITE_SCORE))
print(DF.head())