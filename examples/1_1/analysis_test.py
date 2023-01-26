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


from bsuite.utils import plotting


def _tag_pretify(tag):
    return tag.replace('_', ' ').title()

def _radar(
        df: pd.DataFrame, ax: plt.Axes, label: str, all_tags,
        color: str, alpha: float = 0.2, edge_alpha: float = 0.85, zorder: int = 2,
        edge_style: str = '-'):
      """Plot utility for generating the underlying radar plot."""
      tmp = df.groupby('tag').mean().reset_index()

      values = []
      for curr_tag in all_tags:
        score = 0.
        selected = tmp[tmp['tag'] == curr_tag]
        if len(selected) == 1:
          score = float(selected['score'])
        else:
          print('{} bsuite scores found for tag {!r} with setting {!r}. '
                'Replacing with zero.'.format(len(selected), curr_tag, label))
        values.append(score)
      values = np.maximum(values, 0.05)  # don't let radar collapse to 0.
      values = np.concatenate((values, [values[0]]))

      angles = np.linspace(0, 2*np.pi, len(all_tags), endpoint=False)
      angles = np.concatenate((angles, [angles[0]]))

      ax.plot(angles, values, '-', linewidth=5, label=label,
              c=color, alpha=edge_alpha, zorder=zorder, linestyle=edge_style)
      ax.fill(angles, values, alpha=alpha, color=color, zorder=zorder)
      # TODO(iosband): Necessary for some change in matplotlib code...
      axis_angles = angles[:-1] * 180/np.pi
      ax.set_thetagrids(
          axis_angles, map(_tag_pretify, all_tags), fontsize=18)

      # To avoid text on top of gridlines, we flip horizontalalignment
      # based on label location
      text_angles = np.rad2deg(angles)
      for label, angle in zip(ax.get_xticklabels()[:-1], text_angles[:-1]):
        if 90 <= angle <= 270:
          label.set_horizontalalignment('right')
        else:
          label.set_horizontalalignment('left')


def bsuite_radar_plot(summary_data: pd.DataFrame,
                          sweep_vars):
      """Output a radar plot of bsuite data from bsuite_summary by tag."""
      fig = plt.figure(figsize=(8, 8), facecolor='white')

      ax = fig.add_subplot(111, polar=True)
      try:
        ax.set_axis_bgcolor('white')
      except AttributeError:
        ax.set_facecolor('white')
      all_tags = sorted(summary_data['tag'].unique())

      if sweep_vars is None:
        summary_data['agent'] = 'agent'
      elif len(sweep_vars) == 1:
        summary_data['agent'] = summary_data[sweep_vars[0]].astype(str)
      else:
        summary_data['agent'] = (summary_data[sweep_vars].astype(str)
                                 .apply(lambda x: x.name + '=' + x, axis=0)
                                 .apply(lambda x: '\n'.join(x), axis=1)  # pylint:disable=unnecessary-lambda
                                )
      if len(summary_data.agent.unique()) > 5:
        print('WARNING: We do not recommend radar plot for more than 5 agents.')

      # Creating radar plot background by hand, reusing the _radar call
      # it will give a slight illusion of being "3D" as inner part will be
      # darker than the outer
      thetas = np.linspace(0, 2*np.pi, 100)
      ax.fill(thetas, [0.25,] * 100, color='k', alpha=0.05)
      ax.fill(thetas, [0.5,] * 100, color='k', alpha=0.05)
      ax.fill(thetas, [0.75,] * 100, color='k', alpha=0.03)
      ax.fill(thetas, [1.,] * 100, color='k', alpha=0.01)

      palette = lambda x: plotting.CATEGORICAL_COLOURS[x]
      if sweep_vars:
        sweep_data_ = summary_data.groupby('agent')
        for aid, (agent, sweep_df) in enumerate(sweep_data_):
          _radar(sweep_df, ax, agent, all_tags, color=palette(aid))
        if len(sweep_vars) == 1:
          label = sweep_vars[0]
          if label == 'experiment':
            label = 'agent'  # rename if actually each individual agent
          legend = ax.legend(loc=(1.1, 0.), ncol=1, title=label)
          ax.get_legend().get_title().set_fontsize('20')
          ax.get_legend().get_title().set_fontname('serif')
          ax.get_legend().get_title().set_color('k')
          ax.get_legend().get_title().set_alpha(0.75)
          legend._legend_box.align = 'left'  # pylint:disable=protected-access
        else:
          legend = ax.legend(loc=(1.1, 0.), ncol=1,)
        plt.setp(legend.texts, fontname='serif')
        frame = legend.get_frame()
        frame.set_color('white')
        for text in legend.get_texts():
          text.set_color('grey')
      else:
        _radar(summary_data, ax, '', all_tags, color=palette(0))

      # Changing internal lines to be dotted and semi transparent
      for line in ax.xaxis.get_gridlines():
        line.set_color('grey')
        line.set_alpha(0.95)
        line.set_linestyle(':')
        line.set_linewidth(2)

      for line in ax.yaxis.get_gridlines():
        line.set_color('grey')
        line.set_alpha(0.95)
        line.set_linestyle(':')
        line.set_linewidth(2)

      plt.xticks(color='grey', fontname='serif')
      ax.set_rlabel_position(0)
      plt.yticks(
          [0, 0.25, 0.5, 0.75, 1],
          ['', '.25', '.5', '.75', '1'],
          color='k', alpha=0.75, fontsize=16, fontname='serif')
      # For some reason axis labels are behind plot by default ...
      ax.set_axisbelow(False)
      plt.show()

experiments = {'dqn': 'data/experiment_1', 'dqn2': 'data2/experiment_2'}
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
print(BSUITE_SUMMARY)
bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)


