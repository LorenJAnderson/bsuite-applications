from bsuite.experiments import summary_analysis
from bsuite.logging import csv_load

experiments = {'vanilla': 'reports/memory1', 'rnn': 'reports/memory2'}
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
# summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)
print(BSUITE_SUMMARY)