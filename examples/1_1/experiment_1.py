from stable_baselines3 import A2C

import bsuite
from bsuite.utils import gym_wrapper

import pandas as pd

SAVE_PATH = 'data3/experiment_1'

from multiprocessing import Pool

from library.bsuite_sweep import sweep_list


def log_experiment(run):
    env_name, time_steps, fixed, csv_name = run
    base_env = bsuite.load_and_record(env_name, save_path=SAVE_PATH, overwrite=True)
    env = gym_wrapper.GymFromDMEnv(base_env)
    model = A2C("MlpPolicy", env, verbose=0, learning_rate=0.001, ent_coef=0.01)
    if fixed:
        model.learn(total_timesteps=time_steps)
    else:
        # TODO determine way to stop early once experiment completes X episodes
        model.learn(total_timesteps=time_steps)


with Pool(2) as p:
    p.map(log_experiment, sweep_list())

print('finished')