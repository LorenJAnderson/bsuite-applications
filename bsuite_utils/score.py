import numpy as np


def score_deep_sea(rews, episodes, max_score):
    episode_len = len(rews) / episodes
    avg_episode = np.sum(rews) / (len(rews) / episodes)
    exp_regret = (max_score - avg_episode)
    rand_regret = max_score