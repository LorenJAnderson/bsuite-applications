from bsuite.experiments import summary_analysis
from bsuite.logging import csv_load

experiments = {'basic': 'data/memory_len_basic/22',
               'rnn': 'data/memory_len_rnn/22'}
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
print(BSUITE_SCORE)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
# summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)
print(BSUITE_SUMMARY)