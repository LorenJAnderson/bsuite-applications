from sb3_contrib import RecurrentPPO

from stable_baselines3 import PPO

import bsuite
from bsuite.utils import gym_wrapper

save_path_1 = 'reports/memory1'
save_path_2 = 'reports/memory2'

for i in range(1):
    print(i, 'ppo_regular')
    env_name = 'memory_size/' + str(i)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path_1, overwrite=True))
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30_000)

    # print(i, 'ppo_rnn')
    # env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path_2, overwrite=True))
    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=0)
    # model.learn(total_timesteps=i*10_000 + 20_000)
