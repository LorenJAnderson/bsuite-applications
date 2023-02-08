import gym


class LifeWrapper(gym.Env):
    def __init__(self, env):
        self.count_resets = 0
        self.base_env = env
        self.obs = None
        self.rew = None
        self.done = None
        self.info = None

    def reset(self):
        if (self.count_resets % 5) == 0:
            return self.hard_reset()
        else:
            return self.soft_reset()

    def soft_reset(self):
        obs = self.base_env.reset()
        return obs, self.rew, self.done, self.info

    def hard_reset(self):
        return self.base_env.reset()

    def step(self, action):
        self.obs, self.rew, self.done, self.info = self.base_env.step(action)
        if self.done:
            self.count_resets += 1
            return self.reset()
        else:
            return self.obs, self.rew, self.done, self.info
