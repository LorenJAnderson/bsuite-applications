from stable_baselines3 import A2C

import bsuite
from bsuite.utils import gym_wrapper

for i, lr in enumerate([1e-3]):
    save_path = 'reports/stable_baselines3/' + str(i) + '/'
    env_name = 'bandit_noise/' + str(12)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record(env_name, save_path=save_path, overwrite=True))
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=lr)
    model.learn(total_timesteps=100_000)
