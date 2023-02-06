from stable_baselines3 import A2C
from stable_baselines3 import DQN

import bsuite
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis
from bsuite.logging import sqlite_load

import warnings

from bsuite.experiments import summary_analysis
from bsuite.logging import csv_load
from bsuite.logging import sqlite_load

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotnine as gg



from bsuite.logging import csv_load

# SAVE_PATH_DQN = 'data/experiment_1'
# DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
# BSUITE_SCORE = summary_analysis.bsuite_score(DF)
# BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, ['basic'])
# __radar_fig__ = summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY)

experiments = {'vanilla': 'reports/memory1', 'rnn': 'reports/memory2'}
experiments = {'dqn': 'data3/experiment_1'}
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
# summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)
print(BSUITE_SUMMARY)

