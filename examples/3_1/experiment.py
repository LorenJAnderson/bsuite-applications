from stable_baselines3 import A2C

from custom_envs.deep_sea_test.environment import DeepSea


for i in range(10):
    env = DeepSea(3)
    model = A2C("MlpPolicy", env, verbose=1, seed=i, tensorboard_log="./tensorboard/ent_0_00")
    model.learn(total_timesteps=100_000)

for i in range(10):
    env = DeepSea(3)
    model = A2C("MlpPolicy", env, ent_coef=0.01, verbose=1, seed=i, tensorboard_log="./tensorboard/ent_0_01")
    model.learn(total_timesteps=100_000)
