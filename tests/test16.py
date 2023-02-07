'Determining the length of an episode for Catch'

import bsuite
from bsuite.utils import gym_wrapper
import gym

raw_env = bsuite.load_from_id(bsuite_id='catch/0')
env = gym_wrapper.GymFromDMEnv(raw_env)
isinstance(env, gym.Env)

env.reset()
done = False
counter = 0
while not done:
    obs, rew, done, info = env.step(0)
    counter += 1
print(counter)
