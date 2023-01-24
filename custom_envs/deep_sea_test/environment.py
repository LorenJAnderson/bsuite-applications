import gym
import numpy as np


class DeepSea(gym.Env):
    def __init__(self, n=5, random=0.0):
        self.N = n
        self.RANDOM = random
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.MultiDiscrete([n+1, n+1])
        self.time = None
        self.state = None

    def reset(self):
        self.time = 0
        self.state = np.array([0, 0])
        return self.state

    def step(self, action):
        if np.random.uniform(0, 1) < self.RANDOM:
            action = np.random.randint(2)
        self.time += 1
        if action == 0:
            new_loc = max(0, self.state[1] - 1)
        elif action == 1:
            new_loc = min(self.N, self.state[1] + 1)
        self.state = np.array([self.time, new_loc])
        done = self.time == self.N
        if done:
            if self.state[1] == self.N:
                reward = self.N * 10
            elif action == 0:
                reward = 0
            else:
                reward = -1

        elif action == 0:
            reward = 0
        else:
            reward = -1
        return self.state, reward, done, {}


if __name__ == "__main__":
    env = DeepSea(5)
    env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(np.random.randint(2))
        print(state, reward, done, info)
