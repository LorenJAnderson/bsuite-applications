import gym


class SkipWrapper(gym.Env):
    def __init__(self, env):
        self.base_env = env

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        done = False
        rew_total = 0
        for _ in range(4):
            obs, rew, done, info = self.base_env.step(action)
            rew_total += rew
            if done:
                break
        return obs, rew_total, done, info
