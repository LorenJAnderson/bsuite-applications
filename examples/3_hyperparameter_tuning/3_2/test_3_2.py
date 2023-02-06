from stable_baselines3 import A2C

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

# [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for i, x in enumerate([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    save_path = 'reports/stable_baselines3/' + str(i) + '/'
    DF, _ = csv_load.load_bsuite(save_path)
    BSUITE_SCORE = summary_analysis.bsuite_score(DF)
    print(x, BSUITE_SCORE)