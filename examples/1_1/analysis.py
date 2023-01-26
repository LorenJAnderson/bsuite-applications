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

experiments = {'dqn': 'data/experiment_1', 'dqn2': 'data/experiment_1'}
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
# summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)



import matplotlib.pyplot as plt
import numpy as np

labels=['Siege', 'Initiation', 'Crowd_control', 'Wave_clear', 'Objective_damage']
markers = [0, 1, 2, 3, 4, 5]
str_markers = ["0", "1", "2", "3", "4", "5"]

def make_radar_chart(name, stats, attribute_labels=labels,
                     plot_markers=markers, plot_str_markers=str_markers):

    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    print(angles)
    angles = angles[0:5]
    print(angles)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    plt.yticks(markers)
    ax.set_title(name)
    ax.grid(True)

    # fig.savefig("static/images/%s.png" % name)

    return plt.show()

make_radar_chart("Agni", [2,3,4,4,5]) # example

