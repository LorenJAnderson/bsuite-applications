from stable_baselines3 import A2C

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis

from bsuite.logging import csv_load

for i in range(5):
    vf_coef = 0.1 + 0.2 * i
    save_path = 'reports/stable_baselines3/' + str(i) + '/'
    DF, _ = csv_load.load_bsuite(save_path)
    BSUITE_SCORE = summary_analysis.bsuite_score(DF)
    print(vf_coef, BSUITE_SCORE)