from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

import bsuite
from bsuite.utils import gym_wrapper

for i in [0, 1, 2, 3, 4, 5, 7, 10, 13, 16]:
    env_name = 'memory_size/' + str(i)

    save_path = 'data/memory_size_basic/' + str(i)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30_000)

    save_path = 'data/memory_size_rnn/' + str(i)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
    model = RecurrentPPO("MlpLstmPolicy", env, n_steps=2048, batch_size=64, verbose=0)
    model.learn(total_timesteps=30_000)

for i, time_steps in [(0, 20_000), (1, 30_000), (2, 40_000), (3, 50_000),
                      (5, 70_000), (7, 90_000), (10, 120_000), (13, 210_000),
                      (17, 510_000), (22, 101_000)]:
    env_name = 'memory_len/' + str(i)

    save_path = 'data/memory_len_basic/' + str(i)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=time_steps)

    save_path = 'data/memory_len_rnn/' + str(i)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
    model = RecurrentPPO("MlpLstmPolicy", env, n_steps=2048, batch_size=64, verbose=0)
    model.learn(total_timesteps=time_steps)
